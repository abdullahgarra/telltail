"""Shared, numerics-critical helpers for the generic-queries pipeline.

These are ported VERBATIM (algorithmically) from the original
``eval_asr_budgeted_minicorpus_from_scores_fast.py`` so that ``evaluate`` and
``sweep`` reproduce the paper CSVs bit-for-bit: same doc-id normalization, same
mergesort tie-breaking, same subset RNG call order, same alias remap.

Terminology: the paper's *victim* is the deployed **target**; each candidate
retriever is a **candidate**. On disk the score cache uses ``target=<alias>/``
+ ``candidate=<alias>/`` for caches this package writes, and we also read the
legacy ``victim=<alias>/`` + ``retriever=<alias>/`` layout produced by the
original scripts (needed for regression against the frozen caches).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Legacy alias that must map to the canonical one.
ALIAS_REMAP = {"gtr-t5": "gtr-t5-base"}

# Directory-prefix pairs we accept, newest first.
_TARGET_PREFIXES = ("target=", "victim=")
_CANDIDATE_PREFIXES = ("candidate=", "retriever=")


def canonical_alias(alias: str) -> str:
    return ALIAS_REMAP.get(alias, alias)


def aliases_to_try(alias: str) -> List[str]:
    can = canonical_alias(alias)
    out = [alias, can]
    if can == "gtr-t5-base":
        out.append("gtr-t5")
    seen: Set[str] = set()
    ans: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            ans.append(x)
    return ans


def normalize_docid(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x)
    if not s or s.lower() == "none":
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def canonicalize_rows(arr: np.ndarray, match_mode: str) -> np.ndarray:
    arr = arr.astype(str, copy=False)
    if match_mode == "ordered":
        return arr
    if match_mode == "set":
        return np.sort(arr, axis=1)
    raise ValueError(f"Unknown match_mode: {match_mode}")


def union_budget_docids(
    target_top50: np.ndarray, query_indices: Sequence[int], corpus_top_k: int
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    k = min(corpus_top_k, target_top50.shape[1])
    for qi in query_indices:
        for j in range(k):
            did = normalize_docid(target_top50[qi, j])
            if did is None or did in seen:
                continue
            seen.add(did)
            out.append(did)
    return out


def topk_from_scores(
    scores_sub: np.ndarray, docids_sub: Sequence[str], top_k: int
) -> np.ndarray:
    if scores_sub.ndim != 2:
        raise ValueError(f"scores_sub must be 2D, got {scores_sub.shape}")
    n_docs = scores_sub.shape[1]
    if n_docs == 0:
        raise ValueError("budgeted corpus has zero docs")
    k_eff = min(top_k, n_docs)
    docids_arr = np.array(list(docids_sub), dtype=object)
    # Stable full sort: deterministic tie-breaking following docids_sub order.
    idx = np.argsort(-scores_sub, axis=1, kind="mergesort")[:, :k_eff]
    return docids_arr[idx]


def rank_from_cols(
    scores: np.ndarray,
    col_indices: np.ndarray,
    kept_docids: Sequence[str],
    query_indices: Sequence[int],
    top_k: int,
    match_mode: str,
) -> np.ndarray:
    rows = np.array(query_indices, dtype=int)
    scores_sub = scores[np.ix_(rows, col_indices)]
    top_docids = topk_from_scores(scores_sub, kept_docids, top_k=top_k)
    return canonicalize_rows(top_docids, match_mode=match_mode)


def generate_query_subsets(
    q_eff: int,
    qb: int,
    n_repeats: int,
    rng: np.random.Generator,
    enumerate_when_leq_repeats: bool,
) -> List[Tuple[int, ...]]:
    if qb > q_eff:
        raise ValueError(f"qb={qb} > q_eff={q_eff}")
    if qb == q_eff:
        return [tuple(range(q_eff))]
    n_possible = math.comb(q_eff, qb)
    if enumerate_when_leq_repeats and n_possible <= n_repeats:
        return list(itertools.combinations(range(q_eff), qb))
    return [
        tuple(sorted(rng.choice(q_eff, size=qb, replace=False).tolist()))
        for _ in range(n_repeats)
    ]


# --- cache directory access (supports both new and legacy layouts) -----------
def list_target_dirs(cache_root: Path) -> List[Path]:
    dirs: List[Path] = []
    for pref in _TARGET_PREFIXES:
        dirs.extend(p for p in cache_root.glob(f"{pref}*") if p.is_dir())
    return sorted(dirs, key=lambda p: p.name.split("=", 1)[1])


def target_alias_of(target_dir: Path) -> str:
    return target_dir.name.split("=", 1)[1]


def find_candidate_dir(target_dir: Path, alias: str) -> Optional[Path]:
    for a in aliases_to_try(alias):
        for pref in _CANDIDATE_PREFIXES:
            p = target_dir / f"{pref}{a}"
            if p.is_dir() and (p / "scores.npy").is_file():
                return p
    return None


def load_score_array(target_dir: Path, alias: str) -> Optional[np.ndarray]:
    cdir = find_candidate_dir(target_dir, alias)
    if cdir is None:
        return None
    return np.load(cdir / "scores.npy", allow_pickle=False)


# The fixed 19-model candidate set S (paper §V-A). Sourced from the registry at
# call sites; kept here as the canonical default ordering used by the evaluator.
def default_candidates() -> List[str]:
    from telltail.models import candidates as _reg_candidates
    return sorted(_reg_candidates())
