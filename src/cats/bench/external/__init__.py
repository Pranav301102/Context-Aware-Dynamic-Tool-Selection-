"""Adapters for open benchmarks that no Claude subagent wrote (FR9; workbook risk D3).

  livemcpbench  LiveMCPBench / LiveMCPTool: 69 MCP servers, 525 tools, 95 human-written tasks
  bfcl          BFCL v4 single-turn categories pooled into one catalog (~3,900 functions, 2,501 tasks)

``load_benchmark(name, raw_dir)`` converts the pinned raw files (scripts/fetch_external.py) into an
``ExternalBenchmark`` (branch documents + tasks); ``cats.bench.external.run`` evaluates CATS and the
baselines on it. Benchmarks that could not be obtained are listed in data/external/README.md.
"""
from __future__ import annotations

from pathlib import Path

from .common import ExternalBenchmark, auto_hierarchy, registry_from_docs

BENCHMARKS = ("livemcpbench", "bfcl")


def load_benchmark(name: str, raw_dir: str | Path, **kwargs) -> ExternalBenchmark:
    if name == "livemcpbench":
        from .livemcpbench import load
    elif name == "bfcl":
        from .bfcl import load
    else:
        raise KeyError(f"unknown benchmark {name!r}; known: {', '.join(BENCHMARKS)}")
    return load(raw_dir, **kwargs)


__all__ = ["BENCHMARKS", "ExternalBenchmark", "auto_hierarchy", "load_benchmark", "registry_from_docs"]
