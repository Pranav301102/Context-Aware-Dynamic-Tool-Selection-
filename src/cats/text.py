"""Text normalisation for lexical retrieval and the documents that describe tools."""
from __future__ import annotations

import re
from typing import Any

from .types import Branch, Toolkit, ToolRecord

_WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
STOPWORDS = frozenset(
    """a an and are as at be by can could do does for from get given has have how i if in into is it its
    me my of on or our please should so that the their them then there these this to us use used uses using
    was we what when where which who will with would you your yours via any all also just some""".split()
)


def stem(word: str) -> str:
    """Very light suffix stripping: enough to join list/lists/listing, query/queries."""
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + repl
    return word


def tokenize(text: str) -> list[str]:
    """Split snake_case, camelCase and punctuation; drop stopwords; stem."""
    words = _WORD.findall(text.replace("_", " "))
    out = []
    for w in words:
        w = w.lower()
        if w in STOPWORDS or len(w) < 2:
            continue
        out.append(stem(w))
    return out


def _schema_text(schema: Any, depth: int = 0) -> list[str]:
    if not isinstance(schema, dict) or depth > 3:
        return []
    parts: list[str] = []
    for name, prop in (schema.get("properties") or {}).items():
        parts.append(name)
        if isinstance(prop, dict):
            if prop.get("description"):
                parts.append(str(prop["description"]))
            if isinstance(prop.get("enum"), list):
                parts.extend(str(e) for e in prop["enum"][:20])
            parts.extend(_schema_text(prop, depth + 1))
            if isinstance(prop.get("items"), dict):
                parts.extend(_schema_text(prop["items"], depth + 1))
    return parts


def tool_document(tool: ToolRecord, toolkit: Toolkit | None = None, *, with_schema: bool = True) -> str:
    """The text a tool is indexed under (BM25 document / embedding input)."""
    parts = [tool.name.replace("_", " "), tool.description]
    if toolkit is not None:
        parts.insert(0, toolkit.name)
    if with_schema:
        parts.extend(_schema_text(tool.input_schema))
    return ". ".join(p for p in parts if p)


def toolkit_document(toolkit: Toolkit, tools: list[ToolRecord]) -> str:
    names = ", ".join(t.name.replace("_", " ") for t in tools)
    return f"{toolkit.name}. {toolkit.summary}. Tools: {names}"


def branch_document(branch: Branch, toolkits: list[Toolkit]) -> str:
    tk = "; ".join(f"{t.name}: {t.summary}" for t in toolkits)
    return f"{branch.name}. {branch.summary}. Servers: {tk}"
