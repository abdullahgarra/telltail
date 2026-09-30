"""`sweep` — open-set OSCR curves from the SAME score cache as `evaluate`.

Reimplements the logic of setup1_openAUC.py / setup2_openAUC.py, but instead of
reading the restricted-rankings dirs it ranks the proxy corpus per candidate
directly from the victim-minicorpus score cache — exactly as `evaluate` does
(same mergesort, same doc-id order), at full budget (|Q|=20) with
corpus_policy=same_as_top_k. This makes the OSCR curves consistent with the
budgeted ASR numbers.

Per candidate c and target t, at top-k:
  TM1 score = mean over queries of ordered exact match of the top-k lists.
  TM2 score = mean over queries of Jaccard of the top-k sets.
Prediction = argmax over candidates; a tie at the max counts as NOT correct
(as in the original scripts). Thresholds swept: inf, every distinct score, -inf.
Output columns match the golden oscr_curve CSVs (TM1 adds score_type +
known_rejection_rate; TM2 does not).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import (
    canonical_alias, list_target_dirs, load_score_array, normalize_docid,
    rank_from_cols, target_alias_of, union_budget_docids,
)

INTERFACE_META = {
    "tm1": dict(setup="setup1", score_type="ordered_exact_match_fraction"),
    "tm2": dict(setup="baseline"),
}


def _score_pair(gt_ord: np.ndarray, cand_ord: np.ndarray, interface: str) -> float:
    """Mean over queries of (TM1) ordered exact match, or (TM2) Jaccard."""
    Q = gt_ord.shape[0]
    total = 0.0
    for i in range(Q):
        if interface == "tm1":
            total += 1.0 if bool(np.array_equal(cand_ord[i], gt_ord[i])) else 0.0
        else:
            a = set(gt_ord[i].tolist())
            b = set(cand_ord[i].tolist())
            total += len(a & b) / max(len(a | b), 1)
    return float(total / max(Q, 1))


def build_score_matrix(
    cache_root: Path, interface: str, k: int, candidates: Sequence[str],
    targets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """S[candidate, target] at full budget for a single top-k."""
    cache_root = Path(cache_root)
    target_dirs = list_target_dirs(cache_root)
    if targets is not None:
        want = {canonical_alias(t) for t in targets}
        target_dirs = [p for p in target_dirs if canonical_alias(target_alias_of(p)) in want]
    rows: List[dict] = []
    for tdir in target_dirs:
        t_alias = canonical_alias(target_alias_of(tdir))
        corpus = [normalize_docid(x) for x in
                  np.load(tdir / "corpus_docids.npy", allow_pickle=True).tolist()]
        corpus = [x for x in corpus if x is not None]
        docid_to_col = {d: i for i, d in enumerate(corpus)}
        target_top50 = np.load(tdir / "victim_full_top50_docids.npy", allow_pickle=True)
        q_eff = int(target_top50.shape[0])
        query_indices = tuple(range(q_eff))
        # full-budget corpus (same_as_top_k => corpus_top_k = k)
        budget_docids = union_budget_docids(target_top50, query_indices, k)
        col_indices, kept = [], []
        for did in budget_docids:
            c = docid_to_col.get(did)
            if c is not None:
                col_indices.append(c)
                kept.append(did)
        col_indices = np.array(col_indices, dtype=int)

        self_scores = load_score_array(tdir, t_alias)
        if self_scores is None or len(kept) == 0:
            continue
        gt_ord = rank_from_cols(self_scores, col_indices, kept, query_indices, k, "ordered")
        for c_alias in candidates:
            cs = self_scores if c_alias == t_alias else load_score_array(tdir, c_alias)
            if cs is None:
                rows.append(dict(candidate=c_alias, target=t_alias, score=np.nan))
                continue
            cand_ord = rank_from_cols(cs, col_indices, kept, query_indices, k, "ordered")
            rows.append(dict(candidate=c_alias, target=t_alias,
                             score=_score_pair(gt_ord, cand_ord, interface)))
    return pd.DataFrame(rows).pivot_table(index="candidate", columns="target",
                                          values="score", aggfunc="mean")


def prediction_table(S: pd.DataFrame) -> pd.DataFrame:
    """Per target: knownness_score = max; tie at top => not correct."""
    cand_set = set(S.index)
    rows: List[dict] = []
    for target in S.columns:
        col = S[target].dropna()
        is_known = target in cand_set
        if col.empty:
            rows.append(dict(target=target, is_known=is_known,
                             knownness_score=np.nan, is_correct_closed=False))
            continue
        mx = float(col.max())
        tied = sorted(col[col == mx].index.tolist())
        if len(tied) > 1:
            is_correct_closed = False
        else:
            is_correct_closed = bool(is_known and tied[0] == target)
        rows.append(dict(target=target, is_known=is_known,
                         knownness_score=mx, is_correct_closed=is_correct_closed))
    return pd.DataFrame(rows)


def oscr_curve(pred: pd.DataFrame, interface: str) -> pd.DataFrame:
    df = pred.dropna(subset=["knownness_score"])
    known = df["is_known"].to_numpy(dtype=bool)
    correct = df["is_correct_closed"].to_numpy(dtype=bool)
    scores = df["knownness_score"].to_numpy(dtype=float)
    n_known = int(known.sum())
    n_unknown = int((~known).sum())
    taus = np.unique(scores)
    taus = np.concatenate(([np.inf], taus[::-1], [-np.inf]))
    rows: List[dict] = []
    for tau in taus:
        accept = scores >= tau
        ccr = float(((accept & known) & correct).sum() / n_known) if n_known else np.nan
        kar = float((accept & known).sum() / n_known) if n_known else np.nan
        n_fa = int((accept & ~known).sum())
        far = float(n_fa / n_unknown) if n_unknown else np.nan
        row = dict(threshold=float(tau), ccr=ccr, far=far,
                   false_accepts=n_fa, known_accept_rate=kar)
        if interface == "tm1":
            row["known_rejection_rate"] = float(1.0 - kar) if not np.isnan(kar) else np.nan
        rows.append(row)
    sort_keys = ["far", "threshold"] if interface == "tm1" else ["far"]
    return pd.DataFrame(rows).sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


def run_sweep(
    cache_root: Path, interface: str, top_ks: Sequence[int],
    candidates: Sequence[str], probe_type: str,
    targets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    meta = INTERFACE_META[interface]
    method = (f"telltail_{probe_type}_ordered" if interface == "tm1"
              else f"ro_{probe_type}")
    frames: List[pd.DataFrame] = []
    for k in top_ks:
        S = build_score_matrix(cache_root, interface, k, candidates, targets)
        pred = prediction_table(S)
        cur = oscr_curve(pred, interface)
        cur.insert(0, "k", k)
        if interface == "tm1":
            cur.insert(0, "score_type", meta["score_type"])
        cur.insert(0, "probe_type", probe_type)
        cur.insert(0, "setup", meta["setup"])
        cur.insert(0, "method", method)
        frames.append(cur)
    return pd.concat(frames, ignore_index=True)


# --- metrics used by the paper plots (for regression) ------------------------
def auoscr(curve_k: pd.DataFrame) -> float:
    df = curve_k.dropna(subset=["far", "ccr"]).sort_values("far")
    if len(df) < 2 or df["far"].nunique() < 2:
        return float("nan")
    g = df.groupby("far", as_index=False)["ccr"].max().sort_values("far")
    far, ccr = g["far"].to_numpy(), g["ccr"].to_numpy()
    if far[0] > 0:
        far, ccr = np.concatenate(([0.0], far)), np.concatenate(([ccr[0]], ccr))
    if far[-1] < 1:
        far, ccr = np.concatenate((far, [1.0])), np.concatenate((ccr, [ccr[-1]]))
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ccr, far))


def ccr_at_fa(curve_k: pd.DataFrame, budget: int) -> float:
    feasible = curve_k[curve_k["false_accepts"] <= budget]
    return float(feasible["ccr"].max()) if len(feasible) else float("nan")


def paper_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    """Per-k CCR@FA<=0, CCR@FA<=2, AUOSCR — the quantities the paper plots."""
    out = []
    for k, g in curve.groupby("k", sort=True):
        out.append(dict(k=int(k), ccr_fa0=ccr_at_fa(g, 0),
                        ccr_fa2=ccr_at_fa(g, 2), auoscr=auoscr(g)))
    return pd.DataFrame(out)


# --- CLI ---------------------------------------------------------------------
def _parse_int_list(raw: str) -> List[int]:
    return [int(p) for p in raw.split(",") if p.strip()]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--interface", required=True, choices=["tm1", "tm2"])
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--score-cache", type=Path, default=None)
    ap.add_argument("--top-ks", default="1,2,3,5,10,20,50")
    ap.add_argument("--probe-type", default=None,
                    help="topic/random label for method/probe_type columns; inferred from query set if omitted.")
    ap.add_argument("--out-dir", type=Path, default=None)


def _infer_probe(qs: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    low = qs.lower()
    if "topic" in low:
        return "topic"
    if "random" in low:
        return "random"
    return qs


def main(args) -> None:
    from telltail.models import candidates as reg_candidates
    from .paths import generic_dir, query_set_name
    qs = query_set_name(args.queries) if args.queries else "adhoc"
    cache = args.score_cache or (generic_dir(qs) / "scores")
    out = args.out_dir or (generic_dir(qs) / "results")
    out.mkdir(parents=True, exist_ok=True)
    probe = _infer_probe(qs, args.probe_type)
    curve = run_sweep(cache_root=Path(cache), interface=args.interface,
                      top_ks=_parse_int_list(args.top_ks),
                      candidates=sorted(reg_candidates()), probe_type=probe)
    prefix = f"{args.interface}_{qs}_oscr_curve"
    curve.to_csv(out / f"{prefix}.csv", index=False)
    print(f"[sweep] {prefix}: wrote {len(curve)} rows to {out}")
