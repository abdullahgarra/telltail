"""On-disk passage store: passage_id -> text, as a single SQLite file.

The research code read passage text from MySQL (`fetch_texts_mysql`). We replace that with
a local SQLite file — one file on disk, opened with Python's stdlib ``sqlite3`` (no server,
no credentials, nothing installed), with an index so lookups don't scan the whole corpus.

Nothing here is shipped: the repo ships only the builder (`tools/build_passage_store.py`),
a few KB. The user runs it once; it downloads MS MARCO and writes the store to their own
``$TELLTAIL_DATA_DIR/passages.sqlite``, exactly like the FAISS indices. No passage dump is
ever committed.

The store is needed ONLY by the topic-level / response-only attack (`opt/retrieve.py`) and
the demo corpus builder — NOT by TM1/TM2 (those score ranks, not text).

    from telltail.passage_store import get_passages
    texts = get_passages(["123", "456"])     # -> {"123": "...", "456": "..."}
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TABLE = "passages"
DEFAULT_FILENAME = "passages.sqlite"
_VAR_LIMIT = 900  # stay under SQLite's 999 bound-variable limit


def store_path(path: Optional[Path] = None) -> Path:
    """Resolve the store path (arg wins; else ``$TELLTAIL_DATA_DIR/passages.sqlite``)."""
    if path is not None:
        return Path(path)
    from telltail.paths import data_dir
    return data_dir() / DEFAULT_FILENAME


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"passage store not found: {path}\n"
            f"The topic-level / response-only attack and the demo need it (TM1/TM2 do not).\n"
            f"Build it once:\n"
            f"    python -m tools.build_passage_store --out {path}")
    return path


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def get_passages(ids: Iterable[str], db_path: Optional[Path] = None) -> Dict[str, str]:
    """Return ``{passage_id: passage_text}`` for the ids present in the store.

    The single lookup helper used by both ``opt/retrieve.py`` and the demo corpus builder.
    """
    ids = [str(i) for i in ids]
    if not ids:
        return {}
    path = _require(store_path(db_path))
    out: Dict[str, str] = {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        for s in range(0, len(ids), _VAR_LIMIT):
            chunk = ids[s:s + _VAR_LIMIT]
            q = (f"SELECT passage_id, passage_text FROM {TABLE} "
                 f"WHERE passage_id IN ({','.join('?' * len(chunk))})")
            for pid, txt in cur.execute(q, chunk):
                out[str(pid)] = txt or ""
    finally:
        conn.close()
    return out


def count(db_path: Optional[Path] = None) -> int:
    path = _require(store_path(db_path))
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    finally:
        conn.close()


def sample_ids(n: int, db_path: Optional[Path] = None) -> List[str]:
    path = _require(store_path(db_path))
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [str(r[0]) for r in
                conn.execute(f"SELECT passage_id FROM {TABLE} LIMIT ?", (n,)).fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Write (used by tools/build_passage_store.py)
# --------------------------------------------------------------------------- #
def create_store(db_path: Path) -> sqlite3.Connection:
    """Create (overwrite) the store and return a writable connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(f"CREATE TABLE {TABLE} (passage_id TEXT PRIMARY KEY, passage_text TEXT)")
    return conn


def insert_batch(conn: sqlite3.Connection, rows: List[Tuple[str, str]]) -> None:
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE}(passage_id, passage_text) VALUES (?, ?)", rows)
