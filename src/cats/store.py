"""Persistent registry store: the workbook's data model (Section 6.2) on SQLite or PostgreSQL.

Tables (one per record type of Section 6.2, plus the change feed and an embedding cache):

    branch           id, name, summary, embedding                                   Branch
    toolkit          id, branch_id, name, server, summary, embedding                Toolkit
    tool             id, toolkit_id, branch_id, name, description, input_schema,    Tool
                     bm25_document, token_cost, version, embedding
    usage_stat       tool_id, selections, calls, successful_calls                   UsageStat
    route_decision   one row per request: branch scores, sets, tools, escalations   RouteDecision
    trace_span       one row per stage of a request: tokens and latency             TraceSpan
    request_outcome  whether the request succeeded and which tools were called
    registry_event   the registry's change feed                                     RegistryEvent
    embedding_cache  (embedder, text hash) -> vector, for stateless embedders
    meta             schema version, tokenizer, embedder, registry version

The in-memory Registry stays the source of truth while a process runs. The store snapshots it
(``save_registry`` / ``load_registry``), follows it change by change (``attach``), and keeps the
logs that evaluation and calibration read back (``log_route``, ``record_outcome``). The
route_decision, trace_span, request_outcome and registry_event tables are append-only: the
database itself rejects UPDATE and DELETE on them.

    store = RegistryStore("registry.sqlite")       # or "postgresql://user@host/cats"
    store.save_registry(reg, index)                # snapshot, with BM25 documents and embeddings
    store.attach(reg, index)                       # then write through every RegistryEvent
    rid = store.log_route(selection, query=q)      # per request (also counts selections)
    store.record_outcome(rid, success=True, called=["slack.post_message"])
    reg2 = RegistryStore("registry.sqlite").load_registry()

Command line: ``python -m cats.store snapshot|info|export`` (see ``main``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .registry import Registry
from .text import tool_document
from .tokens import TokenCounter
from .types import Branch, RegistryEvent, RouteDecision, Selection, Toolkit, TraceSpan, Turn

SCHEMA_VERSION = 1
APPEND_ONLY = ("route_decision", "trace_span", "request_outcome", "registry_event")


class UsageStat(BaseModel):
    """How often a tool was put in a prompt and how often calling it worked (feeds FR16, priors)."""
    tool_id: str
    selections: int = 0
    calls: int = 0
    successful_calls: int = 0

    @property
    def success_rate(self) -> float | None:
        return self.successful_calls / self.calls if self.calls else None


class RouteRecord(BaseModel):
    """One logged request: a row of route_decision with its RouteDecision rebuilt."""
    id: int
    request_id: str
    ts: float
    strategy: str
    query: str | None = None
    request_key: str | None = None
    route: RouteDecision | None = None
    tool_ids: list[str]
    scores: list[float]
    tool_tokens: int = 0
    overhead_tokens: int = 0
    registry_version: int | None = None


# ----------------------------------------------------------------------------- dialects
@dataclass(frozen=True)
class _Dialect:
    name: str
    blob: str
    real: str
    serial_pk: str

    def sql(self, statement: str) -> str:
        # Statements are written with qmark placeholders; psycopg wants %s.
        return statement if self.name == "sqlite" else statement.replace("?", "%s")

    def append_only(self, table: str) -> list[str]:
        if self.name == "sqlite":
            return [
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
                for op in ("UPDATE", "DELETE")
            ]
        return [(f"CREATE OR REPLACE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
                 f"FOR EACH ROW EXECUTE FUNCTION cats_append_only()")]


SQLITE = _Dialect("sqlite", blob="BLOB", real="REAL", serial_pk="INTEGER PRIMARY KEY AUTOINCREMENT")
POSTGRES = _Dialect("postgres", blob="BYTEA", real="DOUBLE PRECISION", serial_pk="BIGSERIAL PRIMARY KEY")

_TABLES = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)
;;
CREATE TABLE IF NOT EXISTS branch (
  id TEXT PRIMARY KEY, seq INTEGER NOT NULL, name TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
  embedding {blob}, updated_at {real} NOT NULL)
;;
CREATE TABLE IF NOT EXISTS toolkit (
  id TEXT PRIMARY KEY, seq INTEGER NOT NULL, branch_id TEXT NOT NULL REFERENCES branch(id),
  name TEXT NOT NULL, server TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
  embedding {blob}, updated_at {real} NOT NULL)
;;
CREATE TABLE IF NOT EXISTS tool (
  id TEXT PRIMARY KEY, toolkit_id TEXT NOT NULL REFERENCES toolkit(id) ON DELETE CASCADE,
  branch_id TEXT NOT NULL, position INTEGER NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL,
  input_schema TEXT NOT NULL, bm25_document TEXT NOT NULL, token_cost INTEGER NOT NULL,
  version TEXT NOT NULL, embedding {blob}, updated_at {real} NOT NULL)
;;
CREATE INDEX IF NOT EXISTS tool_branch ON tool (branch_id)
;;
CREATE INDEX IF NOT EXISTS tool_toolkit ON tool (toolkit_id, position)
;;
CREATE TABLE IF NOT EXISTS usage_stat (
  tool_id TEXT PRIMARY KEY, selections INTEGER NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
  successful_calls INTEGER NOT NULL DEFAULT 0, last_selected {real}, last_called {real})
;;
CREATE TABLE IF NOT EXISTS route_decision (
  id {serial_pk}, request_id TEXT NOT NULL, ts {real} NOT NULL, strategy TEXT NOT NULL,
  request_key TEXT, query TEXT, has_route INTEGER NOT NULL, branch_scores TEXT NOT NULL,
  selected TEXT NOT NULL, conformal_set TEXT NOT NULL, threshold {real}, fallback INTEGER NOT NULL,
  escalations INTEGER NOT NULL, tool_ids TEXT NOT NULL, scores TEXT NOT NULL,
  tool_tokens INTEGER NOT NULL, overhead_tokens INTEGER NOT NULL, registry_version INTEGER)
;;
CREATE INDEX IF NOT EXISTS route_request ON route_decision (request_id)
;;
CREATE TABLE IF NOT EXISTS trace_span (
  id {serial_pk}, request_id TEXT NOT NULL, ts {real} NOT NULL, stage TEXT NOT NULL,
  latency_ms {real} NOT NULL, tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL, meta TEXT NOT NULL)
;;
CREATE INDEX IF NOT EXISTS span_request ON trace_span (request_id)
;;
CREATE TABLE IF NOT EXISTS request_outcome (
  id {serial_pk}, request_id TEXT NOT NULL, ts {real} NOT NULL, success INTEGER NOT NULL,
  called TEXT NOT NULL, meta TEXT NOT NULL)
;;
CREATE TABLE IF NOT EXISTS registry_event (
  id {serial_pk}, ts {real} NOT NULL, kind TEXT NOT NULL, toolkit_id TEXT NOT NULL,
  tool_ids TEXT NOT NULL, registry_version INTEGER NOT NULL)
;;
CREATE TABLE IF NOT EXISTS embedding_cache (
  embedder TEXT NOT NULL, text_hash TEXT NOT NULL, dim INTEGER NOT NULL, vec {blob} NOT NULL,
  PRIMARY KEY (embedder, text_hash))
"""

_PG_FUNCTION = """CREATE OR REPLACE FUNCTION cats_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME; END $$"""


def _blob(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def _vec(blob: Any) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(bytes(blob), dtype=np.float32).copy()


def _is_postgres(target: str) -> bool:
    return target.startswith(("postgresql://", "postgres://")) or (
        "=" in target and " " in target and "dbname" in target)  # libpq key=value DSN


class RegistryStore:
    """A Registry snapshot plus the per-request logs, in SQLite (default) or PostgreSQL.

    ``target`` is a file path, ``":memory:"``, or a PostgreSQL DSN (needs ``psycopg``).
    One store may be shared by threads; writes are serialised by a lock.
    """

    def __init__(self, target: str | Path = ":memory:"):
        target = str(target)
        self.target = target
        if _is_postgres(target):
            import psycopg

            self.dialect = POSTGRES
            # utf8 also on SQL_ASCII databases, where psycopg would otherwise return bytes for TEXT
            self.conn = psycopg.connect(target, autocommit=True, client_encoding="utf8")
        else:
            self.dialect = SQLITE
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(target, check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA foreign_keys = ON")
            if target != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")  # readers do not block the writer
        self._lock = threading.RLock()
        self._depth = 0
        self._registry: Registry | None = None
        self._index = None
        self._create_schema()

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> RegistryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _exec(self, sql: str, params: Sequence[Any] | None = None):
        cur = self.conn.cursor()
        if params is None:
            cur.execute(self.dialect.sql(sql))
        else:
            cur.execute(self.dialect.sql(sql), tuple(params))
        return cur

    def _many(self, sql: str, rows: list[Sequence[Any]]) -> None:
        if rows:
            cur = self.conn.cursor()
            cur.executemany(self.dialect.sql(sql), [tuple(r) for r in rows])
            cur.close()

    def _all(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
        cur = self._exec(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    @contextmanager
    def _tx(self) -> Iterator[None]:
        """One transaction; nested calls join the outer one."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self._depth = 1
            try:
                if self.dialect is SQLITE:
                    self.conn.execute("BEGIN")
                    try:
                        yield
                    except BaseException:
                        self.conn.execute("ROLLBACK")
                        raise
                    self.conn.execute("COMMIT")
                else:
                    with self.conn.transaction():
                        yield
            finally:
                self._depth = 0

    def _create_schema(self) -> None:
        d = self.dialect
        statements = [s.strip().format(blob=d.blob, real=d.real, serial_pk=d.serial_pk)
                      for s in _TABLES.split(";;") if s.strip()]
        if d is POSTGRES:
            statements.append(_PG_FUNCTION)
        for table in APPEND_ONLY:
            statements += d.append_only(table)
        with self._tx():
            for s in statements:
                self._exec(s).close()
            found = self.get_meta("schema_version")
            if found is None:
                self.set_meta("schema_version", SCHEMA_VERSION)
            elif int(found) != SCHEMA_VERSION:
                raise RuntimeError(f"store schema v{found} != code schema v{SCHEMA_VERSION}")

    def get_meta(self, key: str) -> str | None:
        rows = self._all("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0][0] if rows else None

    def set_meta(self, key: str, value: Any) -> None:
        with self._tx():
            self._exec("INSERT INTO meta (key, value) VALUES (?, ?) "
                       "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, str(value))).close()

    # ------------------------------------------------------------- the catalog
    def _vectors(self, index, level: str, ids: Iterable[str]) -> dict[str, np.ndarray]:
        if index is None:
            return {}
        hy = getattr(index, level)
        out = {}
        for i in ids:
            s = hy.slot.get(i)
            if s is not None and s < len(hy.vecs):
                out[i] = hy.vecs[s]
        return out

    def _write_branches(self, reg: Registry, ids: Iterable[str], index=None) -> None:
        order = {b: n for n, b in enumerate(reg.branches)}
        ids = [i for i in ids if i in reg.branches]
        vecs = self._vectors(index, "branches", ids)
        now = time.time()
        self._many(
            "INSERT INTO branch (id, seq, name, summary, embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET seq = excluded.seq, name = excluded.name, summary = excluded.summary, "
            "embedding = COALESCE(excluded.embedding, branch.embedding), updated_at = excluded.updated_at",
            [(b, order[b], reg.branches[b].name, reg.branches[b].summary, _blob(vecs.get(b)), now) for b in ids])

    def _write_toolkit(self, reg: Registry, tk_id: str, index=None) -> None:
        """Make the stored toolkit equal the registry's: row, tools, positions; drop tools it lost."""
        tk = reg.toolkits[tk_id]
        seq = list(reg.toolkits).index(tk_id)
        now = time.time()
        tk_vec = self._vectors(index, "toolkits", [tk_id]).get(tk_id)
        self._exec(
            "INSERT INTO toolkit (id, seq, branch_id, name, server, summary, embedding, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET seq = excluded.seq, "
            "branch_id = excluded.branch_id, name = excluded.name, server = excluded.server, "
            "summary = excluded.summary, embedding = COALESCE(excluded.embedding, toolkit.embedding), "
            "updated_at = excluded.updated_at",
            (tk.id, seq, tk.branch_id, tk.name, tk.server, tk.summary, _blob(tk_vec), now)).close()
        ids = reg.toolkit_tools(tk_id)
        vecs = self._vectors(index, "tools", ids)
        rows = []
        for pos, tid in enumerate(ids):
            t = reg.tools[tid]
            doc = index.tool_text(tid) if index is not None else tool_document(t, tk)
            # Schemas keep their key order (TEXT, not JSONB): the order changes the token count.
            rows.append((tid, tk_id, t.branch_id, pos, t.name, t.description, json.dumps(t.input_schema),
                         doc, t.token_cost, t.version, _blob(vecs.get(tid)), now))
        self._many(
            "INSERT INTO tool (id, toolkit_id, branch_id, position, name, description, input_schema, "
            "bm25_document, token_cost, version, embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET toolkit_id = excluded.toolkit_id, branch_id = excluded.branch_id, "
            "position = excluded.position, name = excluded.name, description = excluded.description, "
            "input_schema = excluded.input_schema, bm25_document = excluded.bm25_document, "
            "token_cost = excluded.token_cost, version = excluded.version, "
            # Saved without an index: keep the stored vector only if the tool did not change.
            "embedding = CASE WHEN excluded.embedding IS NOT NULL THEN excluded.embedding "
            "WHEN tool.version = excluded.version THEN tool.embedding ELSE NULL END, "
            "updated_at = excluded.updated_at", rows)
        stored = {r[0] for r in self._all("SELECT id FROM tool WHERE toolkit_id = ?", (tk_id,))}
        self._many("DELETE FROM tool WHERE id = ?", [(i,) for i in sorted(stored - set(ids))])

    def _write_meta(self, reg: Registry, index=None) -> None:
        self.set_meta("registry_version", reg.version)
        self.set_meta("tokenizer", reg.counter.backend)
        if index is not None:
            self.set_meta("embedder", index.embedder.name)
            self.set_meta("embedding_dim", index.tools.vecs.shape[1])

    def save_registry(self, reg: Registry, index=None) -> dict[str, int]:
        """Snapshot a Registry: the store's catalog becomes equal to it (logs are kept).

        With a built ToolIndex, BM25 documents include enrichment and the branch, toolkit
        and tool embeddings are stored too.
        """
        with self._tx():
            self._write_branches(reg, reg.branches, index)
            for tk_id in reg.toolkits:
                self._write_toolkit(reg, tk_id, index)
            stored_tk = {r[0] for r in self._all("SELECT id FROM toolkit")}
            self._many("DELETE FROM toolkit WHERE id = ?", [(i,) for i in sorted(stored_tk - set(reg.toolkits))])
            stored_b = {r[0] for r in self._all("SELECT id FROM branch")}
            self._many("DELETE FROM branch WHERE id = ?", [(i,) for i in sorted(stored_b - set(reg.branches))])
            self._write_meta(reg, index)
            self.set_meta("saved_at", time.time())
        return self.counts()

    def load_registry(self, counter: TokenCounter | None = None) -> Registry:
        """Rebuild a Registry from the store (through the public upsert path, so token costs and
        versions are recomputed; ``verify`` compares them with the stored ones)."""
        reg = Registry(counter)
        for bid, name, summary in self._all("SELECT id, name, summary FROM branch ORDER BY seq, id"):
            reg.add_branch(Branch(id=bid, name=name, summary=summary))
        tools_by_tk: dict[str, list[dict[str, Any]]] = {}
        for tk_id, name, desc, schema in self._all(
                "SELECT toolkit_id, name, description, input_schema FROM tool ORDER BY toolkit_id, position"):
            tools_by_tk.setdefault(tk_id, []).append(
                {"name": name, "description": desc, "input_schema": json.loads(schema)})
        for tk_id, bid, name, server, summary in self._all(
                "SELECT id, branch_id, name, server, summary FROM toolkit ORDER BY seq, id"):
            reg.upsert_toolkit(Toolkit(id=tk_id, name=name, branch_id=bid, server=server, summary=summary),
                               tools_by_tk.get(tk_id, []))
        stored_version = self.get_meta("registry_version")
        if stored_version is not None:
            reg.version = max(reg.version, int(stored_version))
        return reg

    def verify(self, reg: Registry) -> dict[str, list[str]]:
        """Differences between a Registry and the stored catalog (all lists empty = in sync)."""
        rows = {r[0]: r[1:] for r in self._all("SELECT id, version, token_cost FROM tool")}
        same_tokenizer = self.get_meta("tokenizer") == reg.counter.backend
        return {
            "missing_in_store": sorted(set(reg.tools) - set(rows)),
            "missing_in_registry": sorted(set(rows) - set(reg.tools)),
            "version_mismatch": sorted(i for i, t in reg.tools.items() if i in rows and rows[i][0] != t.version),
            "token_cost_mismatch": sorted(i for i, t in reg.tools.items()
                                          if same_tokenizer and i in rows and rows[i][1] != t.token_cost),
        }

    def embeddings(self, level: str = "tool") -> dict[str, np.ndarray]:
        """Stored vectors for one level: "branch", "toolkit" or "tool"."""
        if level not in ("branch", "toolkit", "tool"):
            raise ValueError(f"unknown level {level!r}")
        return {i: _vec(b) for i, b in self._all(f"SELECT id, embedding FROM {level} WHERE embedding IS NOT NULL")}

    def bm25_documents(self) -> dict[str, str]:
        return dict(self._all("SELECT id, bm25_document FROM tool"))

    # ------------------------------------------------------- following changes
    def attach(self, reg: Registry, index=None) -> RegistryStore:
        """Snapshot ``reg`` and write every later RegistryEvent through (FR11 for the store).

        Pass the ToolIndex to store BM25 documents and embeddings. Attach after ``index.build()``:
        listeners run in subscription order, so the index must re-embed before the store reads it.
        """
        self._registry, self._index = reg, index
        self.save_registry(reg, index)
        reg.subscribe(self.on_event)
        return self

    def on_event(self, event: RegistryEvent) -> None:
        reg = self._registry
        if reg is None:
            raise RuntimeError("RegistryStore.on_event needs attach(registry) first")
        with self._tx():
            self._exec("INSERT INTO registry_event (ts, kind, toolkit_id, tool_ids, registry_version) "
                       "VALUES (?, ?, ?, ?, ?)", (time.time(), event.kind, event.toolkit_id,
                                                  json.dumps(event.tool_ids), event.registry_version)).close()
            tk = reg.toolkits.get(event.toolkit_id)
            if tk is None:  # the whole toolkit is gone (its tools cascade)
                self._exec("DELETE FROM toolkit WHERE id = ?", (event.toolkit_id,)).close()
            else:
                self._write_branches(reg, [tk.branch_id], self._index)
                self._write_toolkit(reg, tk.id, self._index)
            self.set_meta("registry_version", reg.version)

    def events(self) -> list[RegistryEvent]:
        return [RegistryEvent(kind=k, toolkit_id=tk, tool_ids=json.loads(ids), registry_version=v)
                for k, tk, ids, v in self._all(
                    "SELECT kind, toolkit_id, tool_ids, registry_version FROM registry_event ORDER BY id")]

    # ------------------------------------------------------------ request logs
    def log_route(self, selection: Selection, *, request_id: str | None = None, query: str | None = None,
                  history: list[Turn] | None = None, count_usage: bool = True) -> str:
        """Append one RouteDecision and its TraceSpans; count a selection for every tool shown.

        Escalations (``CATS.escalate``) are logged the same way under the same ``request_id``.
        The query text is stored only when passed; its fingerprint (``request_key``) always is
        when a query is given, so logs can be joined with LLM-router scores without the text.
        """
        from .llm_router import request_key

        rid = request_id or uuid.uuid4().hex
        r = selection.route
        now = time.time()
        reg_version = self._registry.version if self._registry is not None else None
        with self._tx():
            self._exec(
                "INSERT INTO route_decision (request_id, ts, strategy, request_key, query, has_route, "
                "branch_scores, selected, conformal_set, threshold, fallback, escalations, tool_ids, scores, "
                "tool_tokens, overhead_tokens, registry_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, now, selection.strategy, request_key(query, history) if query is not None else None, query,
                 int(r is not None), json.dumps(r.branch_scores if r else {}), json.dumps(r.selected if r else []),
                 json.dumps(r.conformal_set if r else []), r.threshold if r else None,
                 int(bool(r and r.fallback)), r.escalations if r else 0, json.dumps(selection.tool_ids),
                 json.dumps([float(s) for s in selection.scores]), selection.tool_tokens,
                 selection.overhead_tokens, reg_version)).close()
            self.log_spans(rid, selection.spans)
            if count_usage:
                self._bump(selection.tool_ids, selections=1, ts_col="last_selected")
        return rid

    def log_spans(self, request_id: str, spans: Iterable[TraceSpan]) -> None:
        now = time.time()
        with self._tx():
            self._many("INSERT INTO trace_span (request_id, ts, stage, latency_ms, tokens_in, tokens_out, meta) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [(request_id, now, s.stage, float(s.latency_ms), s.tokens_in, s.tokens_out,
                         json.dumps(s.meta, default=str)) for s in spans])

    def record_outcome(self, request_id: str, success: bool, called: Sequence[str] = (),
                       meta: dict[str, Any] | None = None) -> None:
        """Append how a request ended; every called tool gets a call (and a success if it worked)."""
        with self._tx():
            self._exec("INSERT INTO request_outcome (request_id, ts, success, called, meta) VALUES (?, ?, ?, ?, ?)",
                       (request_id, time.time(), int(success), json.dumps(list(called)),
                        json.dumps(meta or {}, default=str))).close()
            for t in called:
                self.record_call(t, success)

    def record_call(self, tool_id: str, success: bool) -> None:
        self._bump([tool_id], calls=1, successful_calls=int(success), ts_col="last_called")

    def record_selection(self, tool_ids: Iterable[str]) -> None:
        self._bump(tool_ids, selections=1, ts_col="last_selected")

    def _bump(self, tool_ids: Iterable[str], selections: int = 0, calls: int = 0, successful_calls: int = 0,
              ts_col: str = "last_selected") -> None:
        now = time.time()
        with self._tx():
            self._many(
                f"INSERT INTO usage_stat (tool_id, selections, calls, successful_calls, {ts_col}) "
                f"VALUES (?, ?, ?, ?, ?) ON CONFLICT (tool_id) DO UPDATE SET "
                f"selections = usage_stat.selections + excluded.selections, "
                f"calls = usage_stat.calls + excluded.calls, "
                f"successful_calls = usage_stat.successful_calls + excluded.successful_calls, "
                f"{ts_col} = excluded.{ts_col}",
                [(t, selections, calls, successful_calls, now) for t in dict.fromkeys(tool_ids)])

    # ----------------------------------------------------------------- reading
    def usage(self, tool_id: str | None = None) -> dict[str, UsageStat]:
        sql = "SELECT tool_id, selections, calls, successful_calls FROM usage_stat"
        rows = self._all(sql + " WHERE tool_id = ?", (tool_id,)) if tool_id else self._all(sql)
        return {r[0]: UsageStat(tool_id=r[0], selections=r[1], calls=r[2], successful_calls=r[3]) for r in rows}

    def routes(self, request_id: str | None = None, limit: int | None = None) -> list[RouteRecord]:
        sql = ("SELECT id, request_id, ts, strategy, query, request_key, has_route, branch_scores, selected, "
               "conformal_set, threshold, fallback, escalations, tool_ids, scores, tool_tokens, overhead_tokens, "
               "registry_version FROM route_decision")
        params: tuple = ()
        if request_id is not None:
            sql, params = sql + " WHERE request_id = ?", (request_id,)
        sql += " ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        out = []
        for (rid, req, ts, strat, q, key, has_route, bs, sel, cs, thr, fb, esc, tids, sc, tt, ot,
             rv) in self._all(sql, params):
            route = RouteDecision(branch_scores=json.loads(bs), selected=json.loads(sel),
                                  conformal_set=json.loads(cs), threshold=thr if thr is not None else 0.0,
                                  fallback=bool(fb), escalations=esc) if has_route else None
            out.append(RouteRecord(id=rid, request_id=req, ts=ts, strategy=strat, query=q, request_key=key,
                                   route=route, tool_ids=json.loads(tids), scores=json.loads(sc),
                                   tool_tokens=tt, overhead_tokens=ot, registry_version=rv))
        return out

    def spans(self, request_id: str | None = None) -> list[TraceSpan]:
        sql = "SELECT stage, latency_ms, tokens_in, tokens_out, meta FROM trace_span"
        rows = (self._all(sql + " WHERE request_id = ? ORDER BY id", (request_id,)) if request_id
                else self._all(sql + " ORDER BY id"))
        return [TraceSpan(stage=s, latency_ms=lat, tokens_in=ti, tokens_out=to, meta=json.loads(m))
                for s, lat, ti, to, m in rows]

    def outcomes(self, request_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT request_id, ts, success, called, meta FROM request_outcome"
        rows = (self._all(sql + " WHERE request_id = ? ORDER BY id", (request_id,)) if request_id
                else self._all(sql + " ORDER BY id"))
        return [{"request_id": r, "ts": ts, "success": bool(ok), "called": json.loads(c), "meta": json.loads(m)}
                for r, ts, ok, c, m in rows]

    def counts(self) -> dict[str, int]:
        tables = ["branch", "toolkit", "tool", "usage_stat", *APPEND_ONLY, "embedding_cache"]
        return {t: int(self._all(f"SELECT COUNT(*) FROM {t}")[0][0]) for t in tables}

    def info(self) -> dict[str, Any]:
        meta = dict(self._all("SELECT key, value FROM meta"))
        tokens = self._all("SELECT COALESCE(SUM(token_cost), 0) FROM tool")[0][0]
        return {"backend": self.dialect.name, "target": self.target if self.dialect is SQLITE else "postgresql",
                "meta": meta, "rows": self.counts(), "tool_tokens": int(tokens)}

    # ------------------------------------------------------- embedding cache
    def cached_vectors(self, embedder: str, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 500):
            chunk = list(hashes[start:start + 500])
            marks = ", ".join("?" * len(chunk))
            for h, b in self._all(f"SELECT text_hash, vec FROM embedding_cache WHERE embedder = ? "
                                  f"AND text_hash IN ({marks})", (embedder, *chunk)):
                out[h] = _vec(b)
        return out

    def cache_vectors(self, embedder: str, items: dict[str, np.ndarray]) -> None:
        with self._tx():
            self._many("INSERT INTO embedding_cache (embedder, text_hash, dim, vec) VALUES (?, ?, ?, ?) "
                       "ON CONFLICT (embedder, text_hash) DO NOTHING",
                       [(embedder, h, int(v.shape[0]), _blob(v)) for h, v in items.items()])


class CachedEmbedder:
    """Wrap an Embedder so vectors already in a RegistryStore are not recomputed.

    Worth it for slow models (BGE-M3 on CPU embeds a 10,000-tool catalog in minutes). Only
    stateless embedders are cached: LSA's vectors depend on the corpus it was fitted on, so
    for it every call goes to the base embedder.
    """

    def __init__(self, base, store: RegistryStore, stateless: bool | None = None):
        self.base, self.store = base, store
        self.name = base.name
        self.dim = base.dim
        self.stateless = (not base.name.startswith("lsa")) if stateless is None else stateless
        self.hits = self.misses = 0

    def fit(self, corpus: list[str]) -> None:
        self.base.fit(corpus)
        self.dim = self.base.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.stateless:
            return self.base.embed(texts)
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        found = self.store.cached_vectors(self.name, list(dict.fromkeys(keys)))
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            fresh = self.base.embed([texts[i] for i in todo])
            new = {keys[i]: v for i, v in zip(todo, fresh)}
            self.store.cache_vectors(self.name, new)
            found.update(new)
        self.hits += len(keys) - len(todo)
        self.misses += len(todo)
        return np.stack([found[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, self.dim), np.float32)


# --------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    """python -m cats.store snapshot --db PATH [--corpus DIR]... [--enrich DIR] [--embed]
    python -m cats.store info --db PATH
    python -m cats.store export --db PATH --out DIR      (back to data/SPEC.md corpus files)"""
    p = argparse.ArgumentParser(prog="python -m cats.store", description="CATS registry store")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="load corpus files and save them (optionally with embeddings)")
    s.add_argument("--db", required=True, help="SQLite path or postgresql:// DSN")
    s.add_argument("--corpus", action="append", default=None, help="corpus dir, repeatable (default: data/corpus)")
    s.add_argument("--enrich", default=None, help="enrichment dir, appended to BM25 documents")
    s.add_argument("--embed", action="store_true", help="build the index and store embeddings")
    i = sub.add_parser("info", help="row counts and metadata")
    i.add_argument("--db", required=True)
    e = sub.add_parser("export", help="write the stored catalog as corpus JSON files")
    e.add_argument("--db", required=True)
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)

    store = RegistryStore(a.db)
    if a.cmd == "snapshot":
        from .build import build_registry

        reg = build_registry([Path(d) for d in a.corpus] if a.corpus else None)
        index = None
        if a.embed or a.enrich:
            from .index import ToolIndex, make_embedder

            index = ToolIndex(reg, CachedEmbedder(make_embedder(), store))
            if a.enrich:
                from .enrich import load_enrichment

                index.enrichment = load_enrichment(Path(a.enrich), reg)
            index.build()
        print(json.dumps(store.save_registry(reg, index)))
    elif a.cmd == "info":
        print(json.dumps(store.info(), indent=1))
    else:
        store.load_registry().to_corpus(a.out)
        print(f"wrote corpus files to {a.out}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
