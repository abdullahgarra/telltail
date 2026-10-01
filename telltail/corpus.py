"""Single source of truth for the passage corpus.

Both the FAISS index builder (`indexing/build_faiss.py`) and the passage-store builder
(`tools/build_passage_store.py`) iterate the corpus through the functions here, so the
store's passage ids and the indices' docids are guaranteed to come from the same place
and in the same id space — they cannot drift.

Corpus: `BeIR/msmarco` (the `corpus` split), mapping `_id -> text`.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

CORPUS_HF = "BeIR/msmarco"
CORPUS_CONFIG = "corpus"
CORPUS_SPLIT = "corpus"


def iter_corpus(streaming: bool = True) -> Iterator[Tuple[str, str]]:
    """Yield ``(passage_id, passage_text)`` for every passage, in corpus order.

    Streaming by default so callers that only need ids/text (the passage store) don't
    materialize 8.8M rows in memory.
    """
    from datasets import load_dataset
    ds = load_dataset(CORPUS_HF, CORPUS_CONFIG, split=CORPUS_SPLIT, streaming=streaming)
    for ex in ds:
        yield str(ex["_id"]), ex.get("text", "") or ""


def load_corpus_arrays() -> Tuple[List[str], np.ndarray]:
    """Materialize the whole corpus as ``(texts, docids)`` for the FAISS builder, which
    needs random access to add passages in chunks. Same order and id space as
    ``iter_corpus``."""
    texts: List[str] = []
    ids: List[str] = []
    for pid, txt in iter_corpus(streaming=False):
        ids.append(pid)
        texts.append(txt)
    return texts, np.array(ids, dtype=object)
