"""Build the local passage store (passage_id -> text) as a single SQLite file.

Its own step — NOT part of index building — but it iterates the corpus through the SAME
loader as `indexing/build_faiss.py` (`telltail.corpus.iter_corpus`), so the store's ids and
the indices' docids share one id space and cannot drift.

CPU + disk only (no GPU, no model). Downloads MS MARCO and writes
`$TELLTAIL_DATA_DIR/passages.sqlite` (~3 GB) onto your own disk, exactly like the indices.
Needed ONLY by the topic-level / response-only attack (`opt/retrieve.py`) and the demo —
TM1/TM2 score ranks, not text, so they don't need it.

Usage:
    # full corpus (run once; reports size and build time)
    python -m tools.build_passage_store --out $TELLTAIL_DATA_DIR/passages.sqlite

    # smoke test: first N passages, round-trip a lookup, and check ids against an index
    python -m tools.build_passage_store --out /tmp/passages_sample.sqlite --sample 2000 \
        --smoke --check-index minilm-l6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from telltail.corpus import iter_corpus
from telltail.passage_store import (
    count, create_store, get_passages, insert_batch, sample_ids, store_path,
)

BATCH = 50_000


def build(out_path: Path, sample: Optional[int] = None) -> int:
    """Stream the shared corpus loader into the SQLite store. Returns rows written."""
    conn = create_store(out_path)
    written = 0
    batch = []
    t0 = time.time()
    try:
        for pid, text in iter_corpus(streaming=True):
            batch.append((pid, text))
            if len(batch) >= BATCH:
                insert_batch(conn, batch); conn.commit()
                written += len(batch); batch = []
                print(f"  wrote {written:,}  ({written / max(time.time() - t0, 1e-9):,.0f}/s)")
            if sample is not None and written + len(batch) >= sample:
                break
        if batch:
            insert_batch(conn, batch); conn.commit(); written += len(batch)
    finally:
        conn.close()
    dt = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    print(f"[done] {written:,} passages -> {out_path}")
    print(f"[stats] size={size_gb:.2f} GB  build_time={dt:.1f}s  ({written / max(dt, 1e-9):,.0f} rows/s)")
    return written


def smoke_test(out_path: Path, check_index: Optional[str]) -> None:
    """Round-trip a few ids, then (optionally) confirm store ids == an index's docids."""
    n = count(out_path)
    print(f"[smoke] store has {n:,} rows")
    assert n > 0, "empty store"

    some = sample_ids(3, out_path)
    got = get_passages(some, db_path=out_path)
    assert set(got) == set(some), f"lookup mismatch: asked {some}, got {list(got)}"
    for pid in some:
        print(f"  {pid}: {got[pid][:90]!r}")
    assert get_passages(["__not_a_real_id__"], db_path=out_path) == {}, "missing id must be empty"
    print("[smoke] OK — lookups round-trip, missing ids handled.")

    if check_index:
        _check_against_index(out_path, check_index)


def _check_against_index(out_path: Path, alias: str) -> None:
    """Verify the first-N store ids all appear in `alias`'s FAISS docids — i.e. the store
    and the index share one id space (same corpus loader, no drift)."""
    import numpy as np
    from telltail.models import load_registry
    from telltail.paths import index_dir

    reg = load_registry()
    if alias not in reg:
        print(f"[check-index] unknown alias {alias}; skipping"); return
    hf = reg[alias]["hf_id"]
    dpath = index_dir() / f"{hf.replace('/', '_')}_docids.npy"
    if not dpath.exists():
        print(f"[check-index] no docids for {alias} at {dpath.name}; skipping"); return

    docid_set = set(map(str, np.load(str(dpath), allow_pickle=True)))
    store_sample = sample_ids(500, out_path)
    missing = [pid for pid in store_sample if pid not in docid_set]
    assert not missing, (f"{len(missing)} store ids NOT in {alias} docids (id drift!): "
                         f"{missing[:5]}")
    print(f"[check-index] OK — all {len(store_sample)} sampled store ids are in {alias}'s "
          f"docids ({len(docid_set):,} total). Store and index share one id space.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the passage store (passage_id -> text) as SQLite")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .sqlite path (default: $TELLTAIL_DATA_DIR/passages.sqlite)")
    ap.add_argument("--sample", type=int, default=None,
                    help="build only the first N corpus passages (smoke/testing)")
    ap.add_argument("--smoke", action="store_true", help="run lookup round-trip after building")
    ap.add_argument("--check-index", default=None,
                    help="alias of an index to verify store ids against (e.g. minilm-l6)")
    args = ap.parse_args()

    out = args.out or store_path()
    build(out, sample=args.sample)
    if args.smoke or args.check_index:
        smoke_test(out, args.check_index)


if __name__ == "__main__":
    main()
