"""Token counting for tool definitions and prompts.

Provider tokenizers differ, so every count here is a proxy (workbook Listing 8.1
uses tiktoken o200k_base the same way). Backends, first available wins:

1. tiktoken ``o200k_base`` if its encoding file is already cached locally;
2. a Hugging Face ``tokenizers`` BPE file at ``$CATS_TOKENIZER`` or
   ``~/.cache/cats/tokenizer.json`` (``scripts/fetch_tokenizer.py`` fetches one);
3. a regex approximation of BPE pre-tokenization, which tracks real BPE counts
   on JSON schemas to within roughly 15%.
"""
from __future__ import annotations

import functools
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_APPROX = re.compile(r"[A-Za-z]{1,6}|\d{1,3}|\s+|[^\sA-Za-z\d]")


class TokenCounter:
    def __init__(self, backend: str = "auto"):
        self.requested = backend
        self.backend, self._encode = self._resolve(backend)

    def __getstate__(self):
        return {"requested": self.requested}

    def __setstate__(self, state):
        self.__init__(state["requested"])

    @staticmethod
    def _tiktoken_cached() -> bool:
        d = os.environ.get("TIKTOKEN_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "data-gym-cache")
        return os.path.isdir(d) and bool(os.listdir(d))

    @classmethod
    def _resolve(cls, backend: str):
        # In auto mode tiktoken is used only when its encoding is already cached (no network fetch).
        if backend == "tiktoken" or (backend == "auto" and cls._tiktoken_cached()):
            try:
                import tiktoken

                enc = tiktoken.get_encoding("o200k_base")
                return "tiktoken:o200k_base", lambda s: len(enc.encode(s, disallowed_special=()))
            except Exception:
                if backend == "tiktoken":
                    raise
        if backend in ("auto", "hf"):
            path = Path(os.environ.get("CATS_TOKENIZER", Path.home() / ".cache/cats/tokenizer.json"))
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(str(path))
                return f"hf:{path.name}", lambda s: len(tok.encode(s).ids)
            except Exception:
                if backend == "hf":
                    raise
        return "approx-regex", lambda s: len(_APPROX.findall(s))

    def count(self, text: str) -> int:
        cache = self.__dict__.setdefault("_cache", {})
        n = cache.get(text)
        if n is None:
            if len(cache) > 200_000:
                cache.clear()
            n = cache[text] = self._encode(text)
        return n

    def count_json(self, obj: Any) -> int:
        return self.count(json.dumps(obj, separators=(", ", ": ")))

    def count_tool(self, spec: dict[str, Any]) -> int:
        """Tokens one tool definition adds to every prompt (same measure as Listing 8.1)."""
        return self.count_json(spec)


@functools.lru_cache(maxsize=1)
def default_counter() -> TokenCounter:
    return TokenCounter(os.environ.get("CATS_TOKEN_BACKEND", "auto"))
