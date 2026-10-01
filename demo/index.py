"""Build a small per-victim demo index from demo/data/corpus.jsonl.

Same embedding recipe as indexing/build_faiss.py (models.yaml prefix via
kind='index_passage', encoder.max_seq_length = index_max_seq_length, L2-normalize), but an
**exhaustive flat SQ8** index (IndexScalarQuantizer, no IVF) over the ~5.2k demo corpus so
the demo is CPU-runnable and search is exact. Cached under $TELLTAIL_OUT_DIR/demo/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_corpus(path: Path) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            o = json.loads(line)
            ids.append(str(o["id"])); texts.append(str(o["text"]))
    return ids, texts


def _cache_dir() -> Path:
    base = os.environ.get("TELLTAIL_OUT_DIR")
    d = (Path(base) / "demo") if base else (Path(__file__).resolve().parents[1] / "_local/demo/indexes")
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_index(alias: str, ids: List[str], texts: List[str], *, device=None,
                batch_size: int = 128, use_cache: bool = True):
    """Return (faiss_index, docids). Builds a flat SQ8 IP index via the build_faiss recipe."""
    import faiss
    from telltail.models import load_encoder, embed, load_registry

    cdir = _cache_dir()
    ipath, dpath = cdir / f"{alias}.index", cdir / f"{alias}_docids.npy"
    if use_cache and ipath.exists() and dpath.exists():
        return faiss.read_index(str(ipath)), np.load(str(dpath), allow_pickle=True)

    reg = load_registry()
    if "index_max_seq_length" not in reg[alias]:
        raise SystemExit(f"[fatal] {alias}: no index_max_seq_length in models.yaml")
    enc = load_encoder(alias, device)
    enc.max_seq_length = int(reg[alias]["index_max_seq_length"])
    vecs = embed(alias, texts, kind="index_passage", normalize=False,
                 batch_size=batch_size, device=device, encoder=enc)
    vecs = np.ascontiguousarray(vecs.astype(np.float32, copy=False))
    faiss.normalize_L2(vecs)
    dim = int(vecs.shape[1])
    index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit,
                                       faiss.METRIC_INNER_PRODUCT)  # flat, exhaustive
    index.train(vecs)
    index.add(vecs)
    docids = np.array(ids, dtype=object)
    faiss.write_index(index, str(ipath))
    np.save(str(dpath), docids)
    return index, docids
