"""`fetch` — build the global docid->passage-text cache (ports mapDocs.py).

Collects the union of doc-ids from the retrieval signatures (`*_top{k}_docids.npy`
under retrieval/), streams the BeIR/msmarco corpus once, and writes
passages/global_cache.jsonl with lines {"docid","text","title"}. Incremental:
only missing doc-ids are fetched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


def collect_needed_docids(npy_paths: List[Path], max_k: int = 50) -> Set[str]:
    needed: Set[str] = set()
    for p in npy_paths:
        arr = np.load(p, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] < max_k:
            raise ValueError(f"{p}: expected [Q, >= {max_k}], got {arr.shape}")
        for x in arr[:, :max_k].reshape(-1):
            if x is None:
                continue
            s = str(x)
            if s and s.lower() != "none":
                needed.add(s)
    return needed


def read_cache(path: Path) -> Dict[str, Dict[str, str]]:
    mp: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return mp
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                mp[str(obj["docid"])] = {k: v for k, v in obj.items() if k != "docid"}
    return mp


def stream_msmarco(needed: Set[str], include_title: bool = True,
                   verbose_every: int = 200_000) -> Dict[str, Dict[str, str]]:
    from datasets import load_dataset
    needed = set(map(str, needed))
    found: Dict[str, Dict[str, str]] = {}
    print(f"[fetch] streaming BeIR/msmarco for {len(needed)} passages...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
    seen = 0
    for row in ds:
        seen += 1
        if verbose_every and seen % verbose_every == 0:
            print(f"  scanned {seen:,} | found {len(found):,}/{len(needed):,}")
        _id = row.get("_id")
        if _id is None:
            continue
        _id = str(_id)
        if _id in needed and _id not in found:
            payload = {"text": row.get("text", "") or ""}
            if include_title:
                payload["title"] = row.get("title", "") or ""
            found[_id] = payload
            if len(found) == len(needed):
                break
    print(f"[fetch] fetched {len(found)}/{len(needed)}")
    return found


def write_cache(out_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for docid, payload in cache.items():
            f.write(json.dumps({"docid": str(docid), **payload}, ensure_ascii=False) + "\n")
    print(f"[fetch] wrote {out_path} ({len(cache)} records)")


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--queries", type=Path, required=True,
                    help="Query CSV (its stem names the query set).")
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--no-title", action="store_true")


def main(args) -> None:
    from .paths import query_set_name, retrieval_dir, passages_dir
    qs = query_set_name(args.queries)
    rdir = retrieval_dir(qs)
    npy_paths = sorted(rdir.glob(f"**/*_top{args.max_k}_docids.npy"))
    if not npy_paths:
        raise SystemExit(f"No *_top{args.max_k}_docids.npy under {rdir} (run `retrieve` first).")
    needed = collect_needed_docids(npy_paths, max_k=args.max_k)
    print(f"[fetch] needed doc-ids union = {len(needed)}")
    out = passages_dir(qs) / "global_cache.jsonl"
    existing = read_cache(out)
    missing = set(map(str, needed)) - set(existing)
    print(f"[fetch] cache has {len(existing)}; missing {len(missing)}")
    if missing:
        existing.update(stream_msmarco(missing, include_title=not args.no_title))
        write_cache(out, existing)
    else:
        print("[fetch] cache already complete.")
