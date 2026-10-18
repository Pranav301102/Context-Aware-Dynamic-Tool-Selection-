"""Token and cost ledger with a prompt-cache simulator (NFR1, NFR6, NFR8).

The cache model follows how provider prompt caches behave: a request re-uses the
longest prefix, at tool-block boundaries, that an earlier request in the cache
already sent, provided it is at least ``min_cacheable`` tokens. Cached reads are
billed at a discount and the first write at a premium. This is a simulation for
comparing strategies, not a provider bill.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .types import TraceSpan


@dataclass
class Prices:
    """USD per million tokens. Defaults: the workbook's illustrative $3/M input."""
    input: float = 3.0
    output: float = 15.0
    cache_write_mult: float = 1.25
    cache_read_mult: float = 0.10
    router_mult: float = 1.0  # router input price relative to the main model (e.g. 0.1 for a small model)


@dataclass
class RequestCost:
    input_tokens: int = 0  # uncached input
    cache_write: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    router_tokens: int = 0  # LLM routing calls, billed as input

    def usd(self, p: Prices) -> float:
        return (
            self.input_tokens * p.input
            + self.router_tokens * p.input * p.router_mult
            + self.cache_write * p.input * p.cache_write_mult
            + self.cache_read * p.input * p.cache_read_mult
            + self.output_tokens * p.output
        ) / 1e6

    def __iadd__(self, o: RequestCost) -> RequestCost:
        self.input_tokens += o.input_tokens
        self.cache_write += o.cache_write
        self.cache_read += o.cache_read
        self.output_tokens += o.output_tokens
        self.router_tokens += o.router_tokens
        return self


class PrefixCache:
    """Longest-cached-prefix model over tool blocks (hash chain per block boundary)."""

    def __init__(self, min_cacheable: int = 1024, enabled: bool = True):
        self.min_cacheable = min_cacheable
        self.enabled = enabled
        self._seen: set[str] = set()

    def charge(self, block_keys: list[str], block_tokens: list[int], suffix_tokens: int,
               output_tokens: int = 0) -> RequestCost:
        """Bill one request whose prompt is the given blocks followed by an uncached suffix."""
        cost = RequestCost(output_tokens=output_tokens)
        prefix_total = sum(block_tokens)
        if not self.enabled or prefix_total < self.min_cacheable:
            cost.input_tokens = prefix_total + suffix_tokens
            return cost
        chain, hit_tokens, running = "", 0, 0
        chains = []
        for key, tok in zip(block_keys, block_tokens):
            chain = f"{chain}|{key}"
            running += tok
            chains.append(chain)
            if chain in self._seen and running >= self.min_cacheable:
                hit_tokens = running
        cost.cache_read = hit_tokens
        cost.cache_write = prefix_total - hit_tokens
        cost.input_tokens = suffix_tokens
        self._seen.update(chains)
        return cost


@dataclass
class Tracer:
    spans: list[TraceSpan] = field(default_factory=list)

    @contextmanager
    def span(self, stage: str, **meta):
        t0 = time.perf_counter()
        rec = TraceSpan(stage=stage, latency_ms=0.0, meta=dict(meta))
        try:
            yield rec
        finally:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            self.spans.append(rec)
