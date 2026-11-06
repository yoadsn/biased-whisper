"""SQLite-backed corpus of (audio, reference text) utterances."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..config import AUDIO_DIR, CORPUS_DIR, ensure_dirs
from ..hebrew_numbers import digit_chars

DB_PATH = CORPUS_DIR / "corpus.db"
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  source_key TEXT NOT NULL,
  text TEXT NOT NULL,
  duration REAL,
  audio_path TEXT,
  start REAL, end REAL,
  ref_digits INTEGER DEFAULT 0,
  meta TEXT DEFAULT '{}',
  batch TEXT,
  created REAL,
  UNIQUE(source, source_key)
);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_samples_digits ON samples(ref_digits);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "c", None)
    if c is None:
        ensure_dirs()
        c = sqlite3.connect(DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        c.executescript(SCHEMA)
        _local.c = c
    return c


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["meta"] = json.loads(d.get("meta") or "{}")
    return d


def add_sample(
    source: str,
    source_key: str,
    text: str,
    audio_path: str | Path,
    duration: float,
    start: float | None = None,
    end: float | None = None,
    meta: dict | None = None,
    batch: str | None = None,
) -> int | None:
    """Insert; returns new id or None if (source, source_key) already exists."""
    c = conn()
    try:
        with c:
            cur = c.execute(
                "INSERT INTO samples(source,source_key,text,duration,audio_path,start,end,ref_digits,meta,batch,created)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (source, source_key, text, duration, str(audio_path), start, end, digit_chars(text),
                 json.dumps(meta or {}, ensure_ascii=False), batch, time.time()),
            )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def exists(source: str, source_key: str) -> bool:
    return conn().execute("SELECT 1 FROM samples WHERE source=? AND source_key=?", (source, source_key)).fetchone() is not None


def get(sample_id: int) -> dict | None:
    r = conn().execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
    return _row(r) if r else None


def query(
    source: str | None = None,
    q: str | None = None,
    digits_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    ids: list[int] | None = None,
    order: str = "id",
    exclude_ids: list[int] | None = None,
) -> list[dict]:
    where, args = [], []
    if source:
        where.append("source=?"); args.append(source)
    if q:
        where.append("text LIKE ?"); args.append(f"%{q}%")
    if digits_only:
        where.append("ref_digits>0")
    if ids:
        where.append(f"id IN ({','.join('?' * len(ids))})"); args += ids
    if exclude_ids:
        where.append(f"id NOT IN ({','.join('?' * len(exclude_ids))})"); args += exclude_ids
    sql = "SELECT * FROM samples"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY " + ("RANDOM()" if order == "random" else "id") + " LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [_row(r) for r in conn().execute(sql, args).fetchall()]


def count(source: str | None = None, digits_only: bool = False) -> int:
    sql, args = "SELECT COUNT(*) FROM samples WHERE 1=1", []
    if source:
        sql += " AND source=?"; args.append(source)
    if digits_only:
        sql += " AND ref_digits>0"
    return conn().execute(sql, args).fetchone()[0]


def stats() -> dict:
    rows = conn().execute(
        "SELECT source, COUNT(*) n, SUM(ref_digits>0) with_digits, SUM(duration) secs FROM samples GROUP BY source"
    ).fetchall()
    return {"sources": [dict(r) for r in rows], "total": count()}


def update_text(sample_id: int, text: str) -> None:
    c = conn()
    c.execute("UPDATE samples SET text=?, ref_digits=? WHERE id=?", (text, digit_chars(text), sample_id))
    c.commit()


def update_references(updates: list[tuple[int, str, dict]]) -> None:
    """Commit one short repair batch; never hold a write transaction while scanning."""
    if not updates:
        return
    c = conn()
    values = [(text, digit_chars(text), json.dumps(meta, ensure_ascii=False), sample_id)
              for sample_id, text, meta in updates]
    with c:
        c.executemany("UPDATE samples SET text=?, ref_digits=?, meta=? WHERE id=?", values)


def delete(sample_ids: list[int]) -> int:
    c = conn()
    for r in c.execute(f"SELECT audio_path FROM samples WHERE id IN ({','.join('?' * len(sample_ids))})", sample_ids):
        try:
            Path(r[0]).unlink(missing_ok=True)
        except OSError:
            pass
    cur = c.execute(f"DELETE FROM samples WHERE id IN ({','.join('?' * len(sample_ids))})", sample_ids)
    c.commit()
    return cur.rowcount


def audio_path_for(source: str, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{source}__{key}")
    return AUDIO_DIR / f"{safe}.wav"
