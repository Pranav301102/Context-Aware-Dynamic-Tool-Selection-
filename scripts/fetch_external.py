"""Download the external benchmarks, pinned to a commit, into data/external/<name>/raw/.

Only raw.githubusercontent.com is needed. Each download is checked against the SHA-256 recorded
below, so a changed upstream file fails loudly instead of silently changing the benchmark
(workbook risk D3: pinned, mirrored benchmarks).

Usage:
  python scripts/fetch_external.py                 # every benchmark
  python scripts/fetch_external.py livemcpbench    # one benchmark
  python scripts/fetch_external.py --convert       # also write converted/ (our corpus + task format)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"

LIVEMCP_REF = "36a51a1065dda49c1e503b87edf33a1e0b331a74"  # icip-cas/LiveMCPBench main, 2025-12-18
BFCL_REF = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"  # ShishirPatil/gorilla, last change to bfcl_eval/data
BFCL_DIR = "berkeley-function-call-leaderboard/bfcl_eval/data"
BFCL_CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
                   "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
                   "live_parallel_multiple"]
BFCL_DISTRACTORS = ["irrelevance", "live_irrelevance", "live_relevance"]  # functions only, no gold

SOURCES: dict[str, dict] = {
    "livemcpbench": {
        "repo": "icip-cas/LiveMCPBench", "ref": LIVEMCP_REF, "licence": "Apache-2.0",
        "files": {
            "tools.json": "tools/LiveMCPTool/tools.json",
            "all_annotations.json": "annotated_data/all_annotations.json",
            "LICENSE": "LICENSE",
        },
    },
    "bfcl": {
        "repo": "ShishirPatil/gorilla", "ref": BFCL_REF, "licence": "Apache-2.0",
        "files": {
            **{f"BFCL_v4_{c}.json": f"{BFCL_DIR}/BFCL_v4_{c}.json" for c in BFCL_CATEGORIES + BFCL_DISTRACTORS},
            **{f"possible_answer/BFCL_v4_{c}.json": f"{BFCL_DIR}/possible_answer/BFCL_v4_{c}.json"
               for c in BFCL_CATEGORIES},
            "LICENSE": "LICENSE",
        },
    },
}


def raw_url(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, force: bool = False) -> dict:
    src = SOURCES[name]
    out = EXT / name / "raw"
    manifest_path = EXT / name / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = {}
    for local, remote in src["files"].items():
        dest = out / local
        url = raw_url(src["repo"], src["ref"], remote)
        if force or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as r:  # fixed https host
                dest.write_bytes(r.read())
        digest = sha256(dest)
        want = old.get("files", {}).get(local, {}).get("sha256")
        if want and want != digest:
            sys.exit(f"{name}/{local}: sha256 {digest} does not match the pinned {want}")
        files[local] = {"url": url, "sha256": digest, "bytes": dest.stat().st_size}
    manifest = {"name": name, "repo": src["repo"], "ref": src["ref"], "licence": src["licence"],
                "retrieved": old.get("retrieved", datetime.now(UTC).date().isoformat()), "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{name}: {len(files)} files in {out}")
    return manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("names", nargs="*", default=list(SOURCES))
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--convert", action="store_true", help="write converted/ corpus and tasks")
    a = p.parse_args(argv)
    for n in a.names:
        fetch(n, a.force)
        if a.convert:
            from cats.bench.external import load_benchmark

            bench = load_benchmark(n, EXT / n / "raw")
            bench.write(EXT / n / "converted")
            print(f"{n}: converted {bench.meta['tools']} tools, {bench.meta['tasks']} tasks")


if __name__ == "__main__":
    main()
