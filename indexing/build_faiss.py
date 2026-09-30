#!/usr/bin/env python3
"""Build the per-model FAISS indices over the MS MARCO passage corpus.

This is the code behind the (large, unshipped) indices that `retrieve` searches.
For each model it builds an IVF + 8-bit scalar-quantizer index over all ~8.8M
MS MARCO passages:

    IndexIVFScalarQuantizer(IndexFlatIP(dim), dim, nlist=8192,
                            QT_8bit, METRIC_INNER_PRODUCT)

Passages are embedded through the shared registry (`telltail.models`, driven by
configs/models.yaml — same passage prefixes and OpenAI/ST backend split the rest
of the pipeline uses), L2-normalized (so inner product = cosine), the quantizer
is trained on a sample, then all passages are added in chunks. Writes
`<sanitized_hf_id>.index`, `_docids.npy`, and `_spec.json` — the exact filenames
`retrieve` expects — into `$TELLTAIL_INDEX_DIR` (or `--out`).

Output is NOT committed (each index is multi-GB). Run per model or shard across
SLURM jobs with `--models`.

Examples:
    python indexing/build_faiss.py --models minilm-l6,e5-base
    python indexing/build_faiss.py --all --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import List

import numpy as np


def sanitize(hf_id: str) -> str:
    return hf_id.replace("/", "_")


def build_one(alias: str, hf_id: str, out_dir: Path, corpus_texts: List[str],
              docids: np.ndarray, *, batch_size: int, chunk_size: int,
              train_samples: int, nlist: int, device, max_seq_length: int,
              skip_existing: bool) -> None:
    import faiss
    from telltail.models import embed, load_encoder, is_openai

    ip = out_dir / f"{sanitize(hf_id)}.index"
    dp = out_dir / f"{sanitize(hf_id)}_docids.npy"
    if skip_existing and ip.exists() and dp.exists():
        print(f"[skip] {alias}: already built")
        return
    print(f"[build] {alias} ({hf_id})")

    encoder = None
    if not is_openai(alias):
        encoder = load_encoder(alias, device)
        try:
            encoder.max_seq_length = min(int(getattr(encoder, "max_seq_length", 512) or 512),
                                         max_seq_length)
        except Exception:
            pass

    def embed_norm(texts: List[str]) -> np.ndarray:
        # Registry applies the passage prefix; normalize here via faiss (matches
        # the original builder: encode un-normalized, then L2-normalize).
        vecs = embed(alias, texts, kind="passage", normalize=False,
                     batch_size=batch_size, device=device, encoder=encoder)
        vecs = np.ascontiguousarray(vecs.astype(np.float32, copy=False))
        faiss.normalize_L2(vecs)
        return vecs

    dim = int(embed_norm(["test"]).shape[1])
    print(f"   dim={dim}")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT)

    nsamp = min(train_samples, len(corpus_texts))
    print(f"   training on {nsamp:,} passages")
    index.train(embed_norm(corpus_texts[:nsamp]))
    gc.collect()

    print(f"   adding {len(corpus_texts):,} passages")
    for s in range(0, len(corpus_texts), chunk_size):
        index.add(embed_norm(corpus_texts[s:s + chunk_size]))
        gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(ip))
    np.save(str(dp), docids)
    (out_dir / f"{sanitize(hf_id)}_spec.json").write_text(json.dumps({
        "model": hf_id, "alias": alias, "faiss_metric": "IP", "normalized": True,
        "index_type": "IVFScalarQuantizer", "nlist": nlist,
        "scalar_quantizer": "QT_8bit",
    }, indent=2))
    print(f"   saved {ip.name}")
    del index, encoder
    gc.collect()


def main() -> None:
    from telltail.models import load_registry
    from telltail.paths import index_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="Comma-separated aliases.")
    ap.add_argument("--all", action="store_true", help="Build every model in the registry.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: $TELLTAIL_INDEX_DIR).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--train-samples", type=int, default=100_000)
    ap.add_argument("--nlist", type=int, default=8192)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    if args.all:
        aliases = list(reg.keys())
    elif args.models.strip():
        aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    else:
        raise SystemExit("pass --models <aliases> or --all")
    out_dir = (args.out or index_dir()).expanduser().resolve()

    from datasets import load_dataset
    print("[corpus] loading BeIR/msmarco (full corpus)...")
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    corpus_texts = [ex["text"] for ex in corpus]
    docids = np.array([str(ex["_id"]) for ex in corpus], dtype=object)
    print(f"[corpus] {len(corpus_texts):,} passages")

    for alias in aliases:
        if alias not in reg:
            print(f"[warn] unknown alias {alias}, skipping"); continue
        build_one(alias, reg[alias]["hf_id"], out_dir, corpus_texts, docids,
                  batch_size=args.batch_size, chunk_size=args.chunk_size,
                  train_samples=args.train_samples, nlist=args.nlist,
                  device=args.device, max_seq_length=args.max_seq_length,
                  skip_existing=not args.force)
    print(f"[done] indices under {out_dir}")


if __name__ == "__main__":
    main()
