"""Context analyzer: query + dialogue state -> retrieval features (Algorithm 1, line 1)."""
from __future__ import annotations

import re
from collections import Counter

from .registry import Registry
from .types import Features, Turn

_WORDS = re.compile(r"[a-z0-9]+")
_ACTIONS = (
    "send|post|email|mail|message|ping|notify|tell|text|dm|share|create|open|file|log|add|schedule|book|invite|"
    "save|upload|attach|update|write|draft|put|make|run|deploy|restart|scale|check|compare|translate|convert|"
    "summari[sz]e|export|download|refund|charge|invoice|remind|set|cancel|delete|close|assign|merge|tag|label"
)
_SPLIT = re.compile(
    rf"\s*(?:;|\band then\b|\bthen\b|\bafter that\b|\bafterwards\b|\band also\b|\bplus\b|"
    rf",?\s*\band\b(?=\s+(?:{_ACTIONS})\b)|,\s*(?=(?:{_ACTIONS})\b))\s*",
    re.IGNORECASE,
)


def decompose(query: str, max_parts: int = 3) -> list[str]:
    """Split a multi-step request into sub-requests ("check the pods and open a ticket").

    A cheap rule-based stand-in for LLM query decomposition: it splits on sequencing words
    and on "and"/commas that introduce a new action verb. Returns [query] when nothing splits.
    """
    parts = [p.strip(" ,.") for p in _SPLIT.split(query) if p and p.strip(" ,.")]
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) < 2:
        return [query]
    return parts[:max_parts]
# Words that appear in vendor/toolkit names but carry no vendor identity.
_GENERIC = frozenset(
    """api apis app cloud server mcp search web local file files data db database the of for and tool tools
    service services online official client manager store hub package packages checker interpreter runner
    notebook sandbox documents office finance maps energy economy device access""".split()
)


class ContextAnalyzer:
    """Builds the effective query from the request and recent turns, and spots vendor names.

    Vendor tokens are words from toolkit ids/names that rarely occur in other toolkits'
    tool descriptions (e.g. "slack", "hubspot"), so naming them is a strong routing signal.
    """

    def __init__(self, registry: Registry, history_turns: int = 3, history_chars: int = 400,
                 max_vendor_df: float = 0.02):
        self.registry = registry
        self.history_turns = history_turns
        self.history_chars = history_chars
        self.max_vendor_df = max_vendor_df
        self.vendors: dict[str, set[str]] = {}
        self._dirty = False
        registry.subscribe(self._on_change)
        self.refresh()

    def _on_change(self, event) -> None:
        self._dirty = True  # rebuild the vendor map lazily, on the next request

    def refresh(self) -> None:
        reg = self.registry
        df: Counter = Counter()
        by_toolkit_words: dict[str, set[str]] = {}
        for tk_id in reg.toolkits:
            words = set()
            for t in reg.toolkit_tools(tk_id):
                words |= set(_WORDS.findall(reg.tools[t].description.lower()))
            by_toolkit_words[tk_id] = words
            df.update(words)
        n_tk = max(1, len(reg.toolkits))
        vendors: dict[str, set[str]] = {}
        for tk_id, tk in reg.toolkits.items():
            for w in set(_WORDS.findall(f"{tk_id.replace('_', ' ')} {tk.name.lower()}")):
                if len(w) < 3 or w in _GENERIC:
                    continue
                other_df = df[w] - (1 if w in by_toolkit_words[tk_id] else 0)
                if other_df / n_tk <= self.max_vendor_df:
                    vendors.setdefault(w, set()).add(tk_id)
        self.vendors = vendors

    def analyze(self, query: str, history: list[Turn] | None = None) -> Features:
        if getattr(self, "_dirty", False):
            self.refresh()
            self._dirty = False
        recent = (history or [])[-self.history_turns:]
        ctx = " ".join(t.content for t in recent)[-self.history_chars:]
        # The query is repeated so it outweighs history in BM25 term frequencies.
        text = f"{query} {query} {ctx}".strip() if ctx else query
        words = set(_WORDS.findall(f"{query} {ctx}".lower()))
        hints = sorted({tk for w in words for tk in self.vendors.get(w, ()) if tk in self.registry.toolkits})
        return Features(text=text, query=query, vendor_hints=hints)
