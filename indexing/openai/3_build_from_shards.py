#!/usr/bin/env python3
"""Step 3/3 — assemble the FAISS index from the downloaded embedding shards.

Reads the `<name>.npy` (+ `<name>_docids.npy`) shards written by step 2 and builds
the same index type as the local models — IVF + 8-bit scalar quantizer,
nlist=8192, inner product — writing `<out_prefix>.index`, `_docids.npy`,
`_spec.json` (defaults land in `$TELLTAIL_INDEX_DIR` as the openai model, which
`retrieve` looks up).

    python indexing/openai/3_build_from_shards.py --shard-dir <out_dir>/emb_shards \
        --out-dir $TELLTAIL_INDEX_DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-prefix", default="openai_text-embedding-3-small",
                    help="Basename; must match the registry sanitized hf_id for retrieve.")
    ap.add_argument("--nlist", type=int, default=8192)
    ap.add_argument("--train-samples", type=int, default=100_000)
    ap.add_argument("--add-chunk", type=int, default=200_000)
    args = ap.parse_args()

    shard_files = sorted(p for p in args.shard_dir.glob("*.npy")
                         if not p.name.endswith("_docids.npy"))
    if not shard_files:
        raise SystemExit(f"No shard .npy files in {args.shard_dir}")
    dim = int(np.load(shard_files[0], mmap_mode="r").shape[1])
    print(f"{len(shard_files)} shards, dim={dim}", flush=True)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, args.nlist, faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT)

    # train on the first train_samples vectors
    train, n = [], 0
    for sf in shard_files:
        x = np.load(sf).astype(np.float32)
        train.append(x)
        n += len(x)
        if n >= args.train_samples:
            break
    train = np.vstack(train)[: args.train_samples]
    print(f"training on {len(train):,} vectors", flush=True)
    index.train(train)
    del train

    # add all shards (in add_chunk batches), collecting docids in shard order
    all_docids = []
    for sf in shard_files:
        x = np.load(sf).astype(np.float32)
        d = np.load(sf.with_name(sf.stem + "_docids.npy"), allow_pickle=True)
        for s in range(0, len(x), args.add_chunk):
            index.add(np.ascontiguousarray(x[s:s + args.add_chunk]))
        all_docids.extend([str(v) for v in d])
    print(f"added {index.ntotal:,} vectors", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_dir / f"{args.out_prefix}.index"))
    np.save(args.out_dir / f"{args.out_prefix}_docids.npy",
            np.array(all_docids, dtype=object))
    (args.out_dir / f"{args.out_prefix}_spec.json").write_text(json.dumps({
        "model": "openai/text-embedding-3-small", "faiss_metric": "IP",
        "normalized": True, "index_type": "IVFScalarQuantizer",
        "nlist": args.nlist, "scalar_quantizer": "QT_8bit",
    }, indent=2))
    print(f"wrote {args.out_dir / (args.out_prefix + '.index')}", flush=True)


if __name__ == "__main__":
    main()
