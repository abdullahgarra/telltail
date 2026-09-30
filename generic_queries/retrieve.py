"""`retrieve` — per-model top-k retrieval signatures over MS MARCO (ports
generateMatrices.py).

For each candidate model, load its FAISS index (from $TELLTAIL_INDEX_DIR),
embed the queries with that model (via telltail.models), search top-k, and save
`<sanitized_model>/<alias>_top{k}_docids.npy` under retrieval/. Query prefixes
and the OpenAI/ST backend split come from configs/models.yaml — no hardcoded
model registry, no API-key CLI flag (OPENAI_API_KEY from the environment).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _sanitize(hf_id: str) -> str:
    return hf_id.replace("/", "_")


def find_faiss_files(hf_id: str, index_dirs: List[Path]
                     ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    san = _sanitize(hf_id)
    for base in index_dirs:
        idx, docs, spec = base / f"{san}.index", base / f"{san}_docids.npy", base / f"{san}_spec.json"
        if idx.exists() and docs.exists():
            return idx, docs, (spec if spec.exists() else None)
    return None, None, None


def _load_spec(spec_path: Optional[Path]) -> dict:
    if spec_path is None:
        return {}
    try:
        return json.loads(spec_path.read_text())
    except Exception:
        return {}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--candidates-only", action="store_true",
                    help="Only the 19 candidate models (default: all models in the registry).")
    ap.add_argument("--models", default="",
                    help="Optional comma-separated alias subset.")
    ap.add_argument("--nprobe", type=int, default=8192)
    ap.add_argument("--force", action="store_true")


def main(args) -> None:
    import faiss
    import pandas as pd
    from telltail.models import load_registry, candidates, embed, load_encoder, is_openai
    from telltail.paths import index_dir
    from .paths import query_set_name, retrieval_dir

    qs = query_set_name(args.queries)
    out_dir = retrieval_dir(qs)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dirs = [index_dir()]

    reg = load_registry()
    if args.models.strip():
        aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    elif args.candidates_only:
        aliases = sorted(candidates())
    else:
        aliases = list(reg.keys())

    qdf = pd.read_csv(args.queries)
    if not {"query_id", "query_text"}.issubset(qdf.columns):
        raise ValueError("queries CSV must have columns: query_id, query_text")
    query_ids = qdf["query_id"].tolist()
    query_texts = qdf["query_text"].astype(str).tolist()

    manifest: Dict[str, dict] = {"queries_csv": str(Path(args.queries).resolve()),
                                 "num_queries": len(query_texts), "top_k": args.top_k,
                                 "index_dirs": [str(p) for p in index_dirs],
                                 "found": {}, "missing": {}, "outputs": {}}

    for alias in aliases:
        hf_id = reg[alias]["hf_id"]
        idx_path, docids_path, spec_path = find_faiss_files(hf_id, index_dirs)
        if idx_path is None:
            manifest["missing"][alias] = hf_id
            print(f"[retrieve] MISSING index for {alias} ({hf_id})")
            continue
        manifest["found"][alias] = {"model_id": hf_id, "index_path": str(idx_path),
                                    "docids_path": str(docids_path),
                                    "spec_path": str(spec_path) if spec_path else ""}
        spec = _load_spec(spec_path)
        normalize = bool(spec.get("normalize", True))

        model_out = out_dir / _sanitize(hf_id)
        npy_path = model_out / f"{alias}_top{args.top_k}_docids.npy"
        if npy_path.exists() and not args.force:
            print(f"[retrieve] SKIP {alias} (exists)")
            continue

        print(f"[retrieve] {alias} ({hf_id}) idx={idx_path.name} normalize={normalize}")
        index = faiss.read_index(str(idx_path))
        if hasattr(index, "nprobe"):
            index.nprobe = args.nprobe
        docids = np.load(str(docids_path), allow_pickle=True)

        encoder = None if is_openai(alias) else load_encoder(alias, args.device)
        q_emb = embed(alias, query_texts, kind="query", normalize=normalize,
                      batch_size=args.batch_size, device=args.device, encoder=encoder)
        q_emb = np.ascontiguousarray(q_emb.astype(np.float32, copy=False))
        _, I = index.search(q_emb, args.top_k)

        retrieved = np.empty_like(I, dtype=docids.dtype)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                idx = I[i, j]
                retrieved[i, j] = docids[idx] if idx >= 0 else None

        model_out.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), retrieved)
        cols = ["query_id"] + [f"rank_{r}" for r in range(1, args.top_k + 1)]
        rows = [[qid] + [retrieved[qi, r] for r in range(args.top_k)] for qi, qid in enumerate(query_ids)]
        pd.DataFrame(rows, columns=cols).to_csv(
            model_out / f"{alias}_top{args.top_k}_docids.csv", index=False)
        manifest["outputs"][alias] = {"model_id": hf_id, "normalize": normalize,
                                      "npy_docids": str(npy_path)}
        print(f"[retrieve]   saved {npy_path}")

    (out_dir / "manifest_offline_fingerprints.json").write_text(json.dumps(manifest, indent=2))
    print(f"[retrieve] found={len(manifest['found'])} missing={len(manifest['missing'])}")
