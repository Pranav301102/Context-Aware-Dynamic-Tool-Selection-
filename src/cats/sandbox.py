"""Code sandbox: run a Python snippet in a resource-limited subprocess (workbook Code Execution agent).

Each run gets a fresh temporary working directory (deleted afterwards), a minimal environment
that inherits nothing from the parent (no API keys or other secrets), POSIX rlimits on CPU time,
address space, file size and core dumps, a wall-clock timeout that kills the whole process
group, and capped stdout/stderr (the rest is drained and discarded, so a chatty snippet can
neither block on a full pipe nor exhaust the parent's memory).

The limits are set by a tiny bootstrap inside the child before the snippet runs, rather than
with ``preexec_fn``, so the sandbox is safe to call from threaded servers (the demo app).

This is resource isolation, not a security boundary: the snippet runs as the calling user and
can read whatever that user can (and root can raise its own hard limits). For untrusted code
in production, put the same call inside a container, gVisor or nsjail.

    >>> run_python("print(6 * 7)").stdout
    '42\\n'

CLI: ``python -m cats.sandbox snippet.py [--cpu 5 --memory-mb 512 --timeout 10]`` (``-`` reads stdin).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO

# Runs inside the child: apply the limits, then execute main.py as __main__.
_BOOTSTRAP = """\
import json, resource, runpy, sys
lim = json.loads(sys.argv[1])
def _set(kind, soft, hard=None):
    if soft is None:
        return
    try:
        resource.setrlimit(kind, (soft, soft if hard is None else hard))
    except (ValueError, OSError):
        pass
_set(resource.RLIMIT_CORE, 0)
_set(resource.RLIMIT_CPU, lim["cpu"], lim["cpu"] + 1)
_set(resource.RLIMIT_AS, lim["mem"])
_set(resource.RLIMIT_FSIZE, lim["fsize"])
if lim.get("nproc") is not None:
    _set(resource.RLIMIT_NPROC, lim["nproc"])
sys.argv = ["main.py"] + lim["argv"]
del json, resource, lim, _set
runpy.run_path("main.py", run_name="__main__")
"""

SAFE_ENV_KEYS = ("LANG", "LC_ALL", "TZ")


@dataclass
class SandboxLimits:
    cpu_seconds: int = 5  # RLIMIT_CPU: SIGXCPU after this much CPU time
    memory_mb: int = 512  # RLIMIT_AS: allocations beyond this raise MemoryError
    wall_seconds: float = 10.0  # the process group is killed after this much wall-clock time
    max_output_bytes: int = 64_000  # per stream; the rest is drained and dropped
    max_file_mb: int = 16  # RLIMIT_FSIZE: largest file the snippet may write
    max_processes: int | None = None  # RLIMIT_NPROC; per *user*, so off by default


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None  # negative = killed by that signal (POSIX convention)
    timed_out: bool  # wall-clock timeout hit
    truncated: bool = False  # stdout or stderr exceeded max_output_bytes
    duration_ms: float = 0.0
    signal: str | None = None  # e.g. SIGXCPU (CPU limit) or SIGKILL (timeout)
    limits: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return asdict(self)


class _CappedReader(threading.Thread):
    """Reads a pipe to EOF, keeping the first ``cap`` bytes."""

    def __init__(self, stream: IO[bytes], cap: int):
        super().__init__(daemon=True)
        self.stream, self.cap = stream, cap
        self.buf = bytearray()
        self.truncated = False

    def run(self) -> None:
        try:
            while chunk := self.stream.read(8192):
                room = self.cap - len(self.buf)
                if room > 0:
                    self.buf += chunk[:room]
                if len(chunk) > max(room, 0):
                    self.truncated = True
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        return self.buf.decode("utf-8", errors="replace")


class Sandbox:
    """Runs Python snippets under :class:`SandboxLimits` (see the module docstring)."""

    def __init__(self, limits: SandboxLimits | None = None, python: str = sys.executable,
                 env: Mapping[str, str] | None = None):
        """``env`` adds variables to the child's minimal environment; nothing is inherited."""
        self.limits = limits or SandboxLimits()
        self.python = python
        self.extra_env = dict(env or {})

    def environment(self, workdir: str) -> dict[str, str]:
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir,
               "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}  # BLAS thread pools reserve memory
        env.update({k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ})
        env.update(self.extra_env)
        return env

    def run(self, code: str, *, stdin: str = "", files: Mapping[str, str | bytes] | None = None,
            argv: list[str] | None = None, limits: SandboxLimits | None = None) -> SandboxResult:
        """Run ``code`` as ``main.py`` in a fresh temp dir. ``files`` are written next to it
        (relative paths only); ``argv`` becomes ``sys.argv[1:]``."""
        lim = limits or self.limits
        spec = {"cpu": int(lim.cpu_seconds), "mem": int(lim.memory_mb) * 2**20,
                "fsize": int(lim.max_file_mb) * 2**20, "nproc": lim.max_processes, "argv": list(argv or [])}
        with tempfile.TemporaryDirectory(prefix="cats-sbx-") as workdir:
            root = Path(workdir).resolve()
            for name, content in (files or {}).items():
                target = (root / name).resolve()
                if root not in target.parents:
                    raise ValueError(f"sandbox file {name!r} escapes the working directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode() if isinstance(content, str) else content)
            (root / "main.py").write_text(code)
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                [self.python, "-E", "-s", "-c", _BOOTSTRAP, json.dumps(spec)], cwd=root,
                env=self.environment(str(root)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True, close_fds=True,
            )
            readers = [_CappedReader(proc.stdout, lim.max_output_bytes),
                       _CappedReader(proc.stderr, lim.max_output_bytes)]
            for r in readers:
                r.start()
            threading.Thread(target=_feed, args=(proc.stdin, stdin.encode()), daemon=True).start()
            timed_out = False
            try:
                proc.wait(timeout=lim.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            _kill_group(proc)  # on timeout the snippet; otherwise anything it left running in its group
            proc.wait()
            for r in readers:
                r.join(timeout=5)
            duration = (time.perf_counter() - t0) * 1000
        rc = proc.returncode
        sig = signal.Signals(-rc).name if rc is not None and rc < 0 and -rc in signal.valid_signals() else None
        return SandboxResult(stdout=readers[0].text(), stderr=readers[1].text(), exit_code=rc, timed_out=timed_out,
                             truncated=any(r.truncated for r in readers), duration_ms=round(duration, 1),
                             signal=sig, limits=asdict(lim))


def _feed(pipe: IO[bytes], data: bytes) -> None:
    try:
        if data:
            pipe.write(data)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_python(code: str, *, stdin: str = "", files: Mapping[str, str | bytes] | None = None,
               **limits) -> SandboxResult:
    """One-shot helper: ``run_python(code, cpu_seconds=2, wall_seconds=5)``."""
    return Sandbox(SandboxLimits(**limits)).run(code, stdin=stdin, files=files)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="Python file to run, or - for stdin")
    ap.add_argument("--cpu", type=int, default=5, help="CPU seconds")
    ap.add_argument("--memory-mb", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=10.0, help="wall-clock seconds")
    ap.add_argument("--max-output", type=int, default=64_000, help="bytes kept per stream")
    a = ap.parse_args(argv)
    code = sys.stdin.read() if a.file == "-" else Path(a.file).read_text()
    res = run_python(code, cpu_seconds=a.cpu, memory_mb=a.memory_mb, wall_seconds=a.timeout,
                     max_output_bytes=a.max_output)
    print(json.dumps(res.to_dict(), indent=1))
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
