"""`score` — build target-minicorpus score caches (ports build_score_cache.py).

For each target V, C_V = union of V's top-`corpus_top_k` retrieved docs over all
queries. For each candidate M, scores[M,V] = M(queries) @ M(C_V passages).T with
shape [Q, |C_V|]. Written under scores/ as target=<V>/candidate=<M>/scores.npy
(+ per-target corpus_docids.npy / victim_full_top50_docids.npy / query_ids.npy,
whose filenames match the frozen caches so `evaluate`/`sweep` read either).

Unlike the original, this imports embedding directly from telltail.models (no
importlib of generate_rankings_byK.py); the OpenAI key comes only from the env.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from ._common import normalize_docid


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def union_docids_from_topk(top50: np.ndarray, k: int) -> List[str]:
    k = min(k, top50.shape[1])
    seen: Set[str] = set()
    out: List[str] = []
    for qi in range(top50.shape[0]):
        for j in range(k):
            did = normalize_docid(top50[qi, j])
            if did is None or did in seen:
                continue
            seen.add(did)
            out.append(did)
    return out


def _read_global_cache(path: Path, text_mode: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "") or "")
            if text_mode == "title_text":
                title = str(obj.get("title", "") or "")
                if title:
                    text = f"{title}\n{text}" if text else title
            out[str(obj["docid"])] = text
    return out


def _find_top50(retrieval_dir: Path, alias: str, top_k: int = 50) -> Optional[Path]:
    hits = sorted(retrieval_dir.glob(f"**/{alias}_top{top_k}_docids.npy"))
    return hits[0] if hits else None


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--corpus-top-k", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--text-mode", default="text", choices=["text", "title_text"])
    ap.add_argument("--targets", default="", help="Optional comma-separated target aliases.")
    ap.add_argument("--candidates", default="",
                    help="Optional comma-separated candidate aliases (default: all registry models).")
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="Cap encoder max_seq_length (matches the frozen score caches).")
    ap.add_argument("--skip-existing", action="store_true")


def main(args) -> None:
    import pandas as pd
    from telltail.models import load_registry, embed, load_encoder, is_openai
    from .paths import query_set_name, retrieval_dir, passages_dir, scores_dir

    if not (1 <= args.corpus_top_k <= 50):
        raise ValueError("--corpus-top-k must be in [1,50]")
    qs = query_set_name(args.queries)
    rdir, out_root = retrieval_dir(qs), scores_dir(qs)
    out_root.mkdir(parents=True, exist_ok=True)
    reg = load_registry()

    qdf = pd.read_csv(args.queries)
    queries = qdf["query_text"].astype(str).tolist()
    query_ids = qdf["query_id"].tolist() if "query_id" in qdf.columns else list(range(len(queries)))
    Q = len(queries)

    cache = _read_global_cache(passages_dir(qs) / "global_cache.jsonl", args.text_mode)
    print(f"[score] Q={Q} passages_cache={len(cache)}")

    all_aliases = list(reg.keys())
    targets = ([t.strip() for t in args.targets.split(",") if t.strip()]
               if args.targets.strip() else all_aliases)
    candidates = ([c.strip() for c in args.candidates.split(",") if c.strip()]
                  if args.candidates.strip() else all_aliases)
    # only targets that actually have a retrieval signature
    targets = [t for t in targets if _find_top50(rdir, t) is not None]
    if not targets:
        raise SystemExit(f"No target signatures under {rdir} (run `retrieve` first).")

    # Preload target top-50 + write per-target static files.
    target_top50: Dict[str, np.ndarray] = {}
    for t in targets:
        arr = np.load(_find_top50(rdir, t), allow_pickle=True)
        Q_eff = min(Q, arr.shape[0])
        target_top50[t] = arr[:Q_eff, :50]
        tdir = out_root / f"target={t}"
        tdir.mkdir(parents=True, exist_ok=True)
        corpus = union_docids_from_topk(target_top50[t], args.corpus_top_k)
        np.save(tdir / "corpus_docids.npy", np.array(corpus, dtype=object))
        np.save(tdir / "victim_full_top50_docids.npy", target_top50[t].astype(object, copy=False))
        np.save(tdir / "query_ids.npy", np.array(query_ids[:Q_eff], dtype=object))

    for m in candidates:
        hf_id = reg[m]["hf_id"]
        print(f"[score] candidate {m} ({hf_id})")
        encoder = None if is_openai(m) else load_encoder(m, args.device)
        if encoder is not None:  # match build_score_cache.py: set 512 unconditionally
            try:
                encoder.max_seq_length = args.max_seq_length
            except Exception:
                pass
        q_emb_cache: Dict[int, np.ndarray] = {}

        def q_emb_for(Q_eff: int) -> np.ndarray:
            if Q_eff not in q_emb_cache:
                q_emb_cache[Q_eff] = embed(m, queries[:Q_eff], kind="query", normalize=True,
                                           batch_size=args.batch_size, device=args.device,
                                           encoder=encoder)
            return q_emb_cache[Q_eff]

        for t in targets:
            tdir = out_root / f"target={t}"
            cdir = tdir / f"candidate={m}"
            spath = cdir / "scores.npy"
            if args.skip_existing and spath.exists():
                continue
            corpus = np.load(tdir / "corpus_docids.npy", allow_pickle=True).astype(object).tolist()
            missing = [d for d in corpus if str(d) not in cache]
            if missing:
                raise RuntimeError(f"target={t}: {len(missing)} passages missing text, e.g. {missing[:3]}")
            Q_eff = target_top50[t].shape[0]
            p_texts = [cache[str(d)] for d in corpus]
            p_emb = embed(m, p_texts, kind="passage", normalize=True,
                          batch_size=args.batch_size, device=args.device, encoder=encoder)
            scores = np.ascontiguousarray(
                q_emb_for(Q_eff).astype(np.float32) @ p_emb.astype(np.float32).T).astype(np.float32)
            cdir.mkdir(parents=True, exist_ok=True)
            np.save(spath, scores)
            del p_emb, scores
            gc.collect()
        del encoder
        gc.collect()

    (out_root / "manifest_score_cache.json").write_text(json.dumps({
        "created_at": _now(), "kind": "target_minicorpus_score_cache",
        "query_set": qs, "corpus_top_k": args.corpus_top_k, "text_mode": args.text_mode,
        "targets": targets, "candidates": candidates,
    }, indent=2), encoding="utf-8")
    print(f"[score] done: {len(targets)} targets x {len(candidates)} candidates -> {out_root}")
