"""Run manifests (workbook NFR8: every reported number traceable to code, data and config).

A manifest records, for one experiment run: the configuration and seed, the git revision (read
with ``git rev-parse`` only; nothing in the repository is written), Python and package versions,
the tokenizer and embedder actually used, SHA-256 hashes of the data files read, model ids,
CATS_* environment settings (secrets redacted), timestamps, outcome and the hashes of the
result files written. It is saved as ``results/manifests/<name>.json`` next to
``results/<name>.json``.

    with run_manifest("bench_final", config=cfg, seed=0, enrichment=Path("data/enrichment")) as m:
        s = build_system(cfg, seed=0, enrichment=Path("data/enrichment"))
        m.set_system(s)                        # tokenizer, embedder, registry, calibration
        res = run_baselines(s)
        m.add_output(save("bench_final", res))
    # -> results/manifests/bench_final.json, status "ok" (or "error" with the exception)

``verify_data(manifest)`` re-hashes the data files and lists what changed since the run.
Command line: ``python -m cats.manifest snapshot|show|verify NAME``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .build import DATA, ROOT

MANIFESTS = ROOT / "results" / "manifests"
SCHEMA = 1
PACKAGES = (
    "cats", "numpy", "scipy", "scikit-learn", "pydantic", "tokenizers", "tiktoken", "wordllama",
    "sentence-transformers", "anthropic", "httpx", "mcp", "fastapi", "uvicorn", "opentelemetry-sdk",
    "langfuse", "pyarrow", "hydra-core", "omegaconf", "psycopg", "qdrant-client", "faiss-cpu", "langgraph",
)
_SECRET = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|AUTH|CREDENTIAL|DSN|DATABASE_URL", re.IGNORECASE)
_ENV_PREFIXES = ("CATS_", "OTEL_", "HYDRA_", "TIKTOKEN_", "LANGFUSE_HOST", "PYTHONHASHSEED")


class Manifest(BaseModel):
    """One experiment run, as written to ``results/manifests/<name>.json``."""
    schema_version: int = SCHEMA
    name: str
    experiment: str | None = None
    status: str = "running"  # running | ok | error
    error: str | None = None
    created_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    git: dict[str, Any] = Field(default_factory=dict)
    python: dict[str, Any] = Field(default_factory=dict)
    platform: dict[str, Any] = Field(default_factory=dict)
    packages: dict[str, str | None] = Field(default_factory=dict)
    tokenizer: str | None = None
    embedder: str | None = None
    models: dict[str, str] = Field(default_factory=dict)
    data: dict[str, str] = Field(default_factory=dict)  # repo-relative path -> sha256
    data_digest: str = ""  # sha256 over the sorted (path, sha256) pairs
    outputs: dict[str, str] = Field(default_factory=dict)  # result files written -> sha256
    system: dict[str, Any] = Field(default_factory=dict)  # registry stats, split sizes, calibration
    command: list[str] = Field(default_factory=list)
    cwd: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------------- collectors
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(args: list[str], root: Path) -> str | None:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # never take locks or refresh the index
    try:
        out = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, timeout=5,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _git_from_files(root: Path) -> dict[str, Any]:
    """Read HEAD without the git binary (plain file reads)."""
    gitdir = root / ".git"
    if gitdir.is_file():  # worktree: "gitdir: <path>"
        text = gitdir.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = (root / text.split(":", 1)[1].strip()).resolve()
    head = gitdir / "HEAD"
    if not head.exists():
        return {}
    ref = head.read_text().strip()
    if not ref.startswith("ref:"):
        return {"rev": ref, "branch": None}
    name = ref.split(":", 1)[1].strip()
    path = gitdir / name
    if path.exists():
        return {"rev": path.read_text().strip(), "branch": name.removeprefix("refs/heads/")}
    packed = gitdir / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == name:
                return {"rev": parts[0], "branch": name.removeprefix("refs/heads/")}
    return {"rev": None, "branch": name.removeprefix("refs/heads/")}


def git_info(root: Path = ROOT) -> dict[str, Any]:
    """Git revision and branch of ``root``, read-only (``git rev-parse``, else .git files)."""
    rev = _git(["rev-parse", "HEAD"], root)
    if rev:
        return {"available": True, "rev": rev, "short": rev[:12],
                "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root), "source": "git rev-parse"}
    info = _git_from_files(root)
    if info.get("rev"):
        return {"available": True, "short": info["rev"][:12], "source": ".git files", **info}
    return {"available": False, "rev": None, "short": None, "branch": info.get("branch"), "source": None}


def package_versions(names: Iterable[str] = PACKAGES) -> dict[str, str | None]:
    """Installed version of each distribution (None when not installed)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def hash_paths(paths: Iterable[Path | str], root: Path = ROOT) -> dict[str, str]:
    """SHA-256 of every file under ``paths`` (files or directories, recursively), keyed by path
    relative to ``root``. Hidden files and ``__pycache__`` are skipped; missing paths are ignored."""
    out: dict[str, str] = {}
    for p in map(Path, paths):
        if p.is_file():
            out[_rel(p, root)] = hash_file(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                parts = f.relative_to(p).parts
                if f.is_file() and "__pycache__" not in parts and not any(x.startswith(".") for x in parts):
                    out[_rel(f, root)] = hash_file(f)
    return dict(sorted(out.items()))


def data_digest(hashes: Mapping[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(f"{k}\0{hashes[k]}\n".encode())
    return h.hexdigest()


def default_data_paths(enrichment: Path | str | None = None, jev: Path | str | None = None,
                       llm_router: Path | str | None = None, extra: Iterable[Path | str] = ()) -> list[Path]:
    """The files a standard CATS run reads: corpus and tasks, plus enrichment / Jev / LLM-router
    answers when used."""
    paths = [DATA / "corpus", DATA / "tasks"]
    paths += [Path(p) for p in (enrichment, jev, llm_router) if p]
    return paths + [Path(p) for p in extra]


def _env() -> dict[str, str]:
    out = {}
    for k, v in sorted(os.environ.items()):
        if k.startswith(_ENV_PREFIXES):
            out[k] = "<redacted>" if _SECRET.search(k) else v
    return out


def to_plain(obj: Any) -> Any:
    """Config objects (dataclass, pydantic, OmegaConf, Path, ...) as JSON-safe plain data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_plain(dataclasses.asdict(obj))
    if isinstance(obj, BaseModel):
        return to_plain(obj.model_dump(mode="json"))
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, DictConfig | ListConfig):
            return to_plain(OmegaConf.to_container(obj, resolve=True))
    except ImportError:
        pass
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return str(obj)
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    return str(obj)


def describe_system(system: Any) -> dict[str, Any]:
    """Tokenizer, embedder, registry stats, split sizes and calibration of a built system
    (``cats.build.System`` or a bare ``CATS``)."""
    cats = getattr(system, "cats", system)
    reg = getattr(system, "registry", None) or getattr(cats, "registry", None)
    index = getattr(system, "index", None) or getattr(cats, "index", None)
    out: dict[str, Any] = {}
    if reg is not None:
        out["registry"] = reg.stats()
        out["registry_version"] = reg.version
    if index is not None:
        out["embedder"] = getattr(index.embedder, "name", None)
    if all(hasattr(system, k) for k in ("train", "calib", "test")):
        out["split"] = {"train": len(system.train), "calib": len(system.calib), "test": len(system.test)}
    report = getattr(cats, "calibration_report", None)
    if report:
        out["calibration"] = to_plain(report)
    return out


def collect(name: str, *, config: Any = None, seed: int | None = None, experiment: str | None = None,
            data: Iterable[Path | str] | None = None, enrichment: Path | str | None = None,
            jev: Path | str | None = None, llm_router: Path | str | None = None,
            models: Mapping[str, str] | None = None, system: Any = None, tokenizer: str | None = None,
            embedder: str | None = None, notes: Mapping[str, Any] | None = None, root: Path = ROOT) -> Manifest:
    """Build a manifest for the current process. ``data`` overrides the default data paths."""
    paths = list(data) if data is not None else default_data_paths(enrichment, jev, llm_router)
    hashes = hash_paths(paths, root)
    m = Manifest(
        name=name, experiment=experiment, created_at=_now(), seed=seed, config=to_plain(config or {}),
        git=git_info(root),
        python={"version": platform.python_version(), "implementation": platform.python_implementation(),
                "executable": sys.executable},
        platform={"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
                  "cpu_count": os.cpu_count()},
        packages=package_versions(), models=dict(models or {}), data=hashes, data_digest=data_digest(hashes),
        command=list(sys.argv), cwd=os.getcwd(), env=_env(), notes=to_plain(dict(notes or {})),
    )
    if tokenizer is None and system is None:
        try:
            from .tokens import default_counter

            tokenizer = default_counter().backend
        except (ImportError, OSError, ValueError, RuntimeError):
            tokenizer = None
    m.tokenizer = tokenizer
    m.embedder = embedder or os.environ.get("CATS_EMBEDDER") or None
    if system is not None:
        _apply_system(m, system)
    return m


def _apply_system(m: Manifest, system: Any) -> None:
    m.system = describe_system(system)
    m.tokenizer = m.system.get("registry", {}).get("tokenizer", m.tokenizer)
    m.embedder = m.system.get("embedder", m.embedder)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=,+-]+", "_", name).strip("_") or "run"


def manifest_path(name: str, out_dir: Path | None = None) -> Path:
    return Path(out_dir or MANIFESTS) / f"{_safe_name(name)}.json"


def write_manifest(m: Manifest, out_dir: Path | None = None) -> Path:
    path = manifest_path(m.name, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m.model_dump(mode="json"), indent=1, default=str))
    tmp.replace(path)
    return path


def load_manifest(name_or_path: str | Path, out_dir: Path | None = None) -> Manifest:
    p = Path(name_or_path)
    if not p.exists():
        p = manifest_path(str(name_or_path), out_dir)
    return Manifest.model_validate_json(p.read_text())


def verify_data(m: Manifest, root: Path = ROOT) -> dict[str, list[str]]:
    """Re-hash the manifest's data files: which changed, went missing, or are new in the same dirs."""
    changed, missing = [], []
    for rel, digest in m.data.items():
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.exists():
            missing.append(rel)
        elif hash_file(p) != digest:
            changed.append(rel)
    dirs = {str(Path(rel).parent) for rel in m.data}
    now = hash_paths([(Path(d) if Path(d).is_absolute() else root / d) for d in dirs], root) if dirs else {}
    added = sorted(k for k in now if k not in m.data and str(Path(k).parent) in dirs)
    return {"changed": changed, "missing": missing, "added": added}


class ManifestRecorder:
    """Handle yielded by ``run_manifest``: add outputs, notes, the built system or model ids."""

    def __init__(self, manifest: Manifest, out_dir: Path | None, root: Path):
        self.manifest, self.out_dir, self.root = manifest, out_dir, root
        self.path = manifest_path(manifest.name, out_dir)

    def add_output(self, path: Path | str) -> None:
        p = Path(path)
        if p.is_file():
            self.manifest.outputs[_rel(p, self.root)] = hash_file(p)

    def note(self, **kv: Any) -> None:
        self.manifest.notes.update(to_plain(kv))

    def set_system(self, system: Any) -> None:
        _apply_system(self.manifest, system)

    def set_models(self, **models: str) -> None:
        self.manifest.models.update(models)

    def write(self) -> Path:
        self.path = write_manifest(self.manifest, self.out_dir)
        return self.path


@contextmanager
def run_manifest(name: str, *, out_dir: Path | None = None, write_on_enter: bool = True,
                 root: Path = ROOT, **kw: Any) -> Iterator[ManifestRecorder]:
    """Wrap an experiment: the manifest is written on entry (status "running") and again on exit
    with the outcome ("ok", or "error" plus the exception; the exception propagates), duration
    and output hashes. Keyword arguments go to ``collect``."""
    rec = ManifestRecorder(collect(name, root=root, **kw), out_dir, root)
    t0 = time.perf_counter()
    if write_on_enter:
        rec.write()
    try:
        yield rec
    except BaseException as e:
        rec.manifest.status = "error"
        rec.manifest.error = "".join(traceback.format_exception_only(type(e), e)).strip()
        raise
    else:
        rec.manifest.status = "ok"
    finally:
        rec.manifest.finished_at = _now()
        rec.manifest.duration_s = round(time.perf_counter() - t0, 3)
        rec.write()


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="cats-manifest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="write a manifest of the current environment and data")
    s.add_argument("name")
    s.add_argument("--enrich", default=None)
    s.add_argument("--seed", type=int, default=None)
    for cmd in ("show", "verify"):
        sub.add_parser(cmd).add_argument("name")
    for sp in sub.choices.values():
        sp.add_argument("--dir", default=None, help=f"manifest directory (default {MANIFESTS})")
    a = p.parse_args(argv)
    out = Path(a.dir) if a.dir else None
    if a.cmd == "snapshot":
        print(write_manifest(collect(a.name, seed=a.seed, enrichment=a.enrich), out))
    elif a.cmd == "show":
        print(load_manifest(a.name, out).model_dump_json(indent=1))
    else:
        print(json.dumps(verify_data(load_manifest(a.name, out)), indent=1))


if __name__ == "__main__":
    main()
