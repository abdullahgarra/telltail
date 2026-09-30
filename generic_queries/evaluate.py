"""`evaluate` — ASR-by-query-budget over a victim-minicorpus score cache.

Ported from ``eval_asr_budgeted_minicorpus_from_scores_fast.py`` with the
**relaxed** setup1 validation (the live version used for the paper): setup1 may
use ``corpus_policy=same_as_top_k`` and swept ``top_ks`` (not only [50]). All
numerics — subset RNG order, mergesort ranking, doc-id normalization, unique
match prediction — are preserved so outputs match the golden CSVs exactly.

Target-level parallelism (``n_jobs``) splits the 53 targets across processes and
sums the integer prediction counts per run — bit-identical to the serial result.

Interfaces (replace the old --setup):
  tm1 -> ordered exact-match (setup1)
  tm2 -> unordered set match  (setup2)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import (
    canonical_alias, generate_query_subsets, list_target_dirs, load_score_array,
    normalize_docid, rank_from_cols, target_alias_of, union_budget_docids,
)

INTERFACE_TO_SETUP = {"tm1": ("setup1", "ordered"), "tm2": ("setup2", "set")}


def predict_from_matches(matches: List[str]) -> str:
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return "UNK"
    return "ABSTAIN"


@dataclass(frozen=True)
class RunKey:
    setup: str
    top_k: int
    corpus_top_k: int
    match_mode: str
    query_budget: int
    repeat_id: int
    query_indices: str


def _new_counts() -> Dict[str, int]:
    return {"n_correct": 0, "n_total": 0, "n_known": 0, "n_unknown": 0,
            "n_known_correct": 0, "n_unknown_correct": 0, "n_pred_known": 0,
            "n_pred_unk": 0, "n_pred_abstain": 0}


def _update_counts(acc, is_known, prediction, correct) -> None:
    acc["n_total"] += 1
    if correct:
        acc["n_correct"] += 1
    if is_known:
        acc["n_known"] += 1
        if correct:
            acc["n_known_correct"] += 1
    else:
        acc["n_unknown"] += 1
        if correct:
            acc["n_unknown_correct"] += 1
    if prediction == "UNK":
        acc["n_pred_unk"] += 1
    elif prediction == "ABSTAIN":
        acc["n_pred_abstain"] += 1
    else:
        acc["n_pred_known"] += 1


def _counts_to_metrics(key: RunKey, c) -> Dict[str, object]:
    n_total, n_known, n_unknown = c["n_total"], c["n_known"], c["n_unknown"]
    return {
        "setup": key.setup, "top_k": key.top_k, "corpus_top_k": key.corpus_top_k,
        "match_mode": key.match_mode, "query_budget": key.query_budget,
        "repeat_id": key.repeat_id, "query_indices": key.query_indices,
        "n_sampled_queries": key.query_budget,
        "asr": c["n_correct"] / n_total if n_total else np.nan,
        "known_acc": c["n_known_correct"] / n_known if n_known else np.nan,
        "unknown_acc": c["n_unknown_correct"] / n_unknown if n_unknown else np.nan,
        "abstain_rate": c["n_pred_abstain"] / n_total if n_total else np.nan,
        "pred_known_rate": c["n_pred_known"] / n_total if n_total else np.nan,
        "pred_unk_rate": c["n_pred_unk"] / n_total if n_total else np.nan,
        "pred_abstain_rate": c["n_pred_abstain"] / n_total if n_total else np.nan,
        **c,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["asr", "known_acc", "unknown_acc", "abstain_rate",
                   "pred_known_rate", "pred_unk_rate", "pred_abstain_rate"]
    group_cols = ["setup", "top_k", "corpus_top_k", "match_mode", "query_budget"]
    rows: List[dict] = []
    for keys, g in raw.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            mean = vals.mean() if len(vals) else np.nan
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            sem = std / math.sqrt(len(vals)) if len(vals) else np.nan
            ci95 = 1.96 * sem if not np.isnan(sem) else np.nan
            row[f"{col}_mean"], row[f"{col}_std"] = mean, std
            row[f"{col}_sem"], row[f"{col}_ci95"] = sem, ci95
        for count_col in ["n_total", "n_known", "n_unknown"]:
            row[count_col] = int(g[count_col].iloc[0]) if count_col in g.columns and len(g) else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def _generate_subsets(q_eff_first, budgets, n_repeats, seed, enumerate_when_leq_repeats):
    """Global subset table (RNG order matches the original exactly)."""
    rng = np.random.default_rng(seed)
    subsets_by_qb: Dict[int, List[Tuple[int, ...]]] = {}
    for qb in sorted(set(budgets)):
        if qb > q_eff_first:
            continue
        subsets_by_qb[qb] = generate_query_subsets(
            q_eff=q_eff_first, qb=qb, n_repeats=n_repeats, rng=rng,
            enumerate_when_leq_repeats=enumerate_when_leq_repeats)
    return subsets_by_qb


def accumulate_counts(cache_root, interface, top_ks, candidates, corpus_policy,
                      subsets_by_qb, targets) -> Dict[RunKey, Dict[str, int]]:
    """Victim-major loop over ``targets``; return {RunKey: counts}. Module-level
    and picklable so it can run in a worker process."""
    setup, match_mode = INTERFACE_TO_SETUP[interface]
    cache_root = Path(cache_root)
    candidates = list(candidates)
    known_set = set(candidates)
    want = {canonical_alias(t) for t in targets}
    target_dirs = [p for p in list_target_dirs(cache_root)
                   if canonical_alias(target_alias_of(p)) in want]

    run_counts: Dict[RunKey, Dict[str, int]] = {}
    for target_dir in target_dirs:
        eval_alias = canonical_alias(target_alias_of(target_dir))
        full_docids_arr = np.load(target_dir / "corpus_docids.npy", allow_pickle=True)
        full_docids = [normalize_docid(x) for x in full_docids_arr.tolist()]
        full_docids = [x for x in full_docids if x is not None]
        docid_to_col = {d: i for i, d in enumerate(full_docids)}

        target_top50 = np.load(target_dir / "victim_full_top50_docids.npy", allow_pickle=True)
        q_eff = int(target_top50.shape[0])

        self_scores = load_score_array(target_dir, eval_alias)
        if self_scores is None:
            continue
        candidate_scores: Dict[str, Optional[np.ndarray]] = {}
        for cand_alias in candidates:
            candidate_scores[cand_alias] = (self_scores if cand_alias == eval_alias
                                            else load_score_array(target_dir, cand_alias))

        budget_col_cache: Dict[Tuple[Tuple[int, ...], int], Tuple[np.ndarray, List[str]]] = {}

        def get_budget_cols(query_indices, corpus_top_k):
            key = (query_indices, corpus_top_k)
            if key in budget_col_cache:
                return budget_col_cache[key]
            budget_docids = union_budget_docids(target_top50, query_indices, corpus_top_k)
            col_indices, kept_docids, missing = [], [], []
            for did in budget_docids:
                col = docid_to_col.get(did)
                if col is None:
                    missing.append(did)
                else:
                    col_indices.append(col)
                    kept_docids.append(did)
            if missing:
                raise RuntimeError(f"target={eval_alias}: budget docids missing: {missing[:5]}")
            if not kept_docids:
                raise RuntimeError(f"target={eval_alias}: empty budget corpus")
            ans = (np.array(col_indices, dtype=int), kept_docids)
            budget_col_cache[key] = ans
            return ans

        for qb, subsets in subsets_by_qb.items():
            if qb > q_eff:
                continue
            for repeat_id, query_indices in enumerate(subsets):
                if max(query_indices) >= q_eff:
                    continue
                query_indices_str = "|".join(map(str, query_indices))
                for top_k in top_ks:
                    corpus_top_k = 50 if corpus_policy == "fixed50" else top_k
                    col_indices, kept_docids = get_budget_cols(query_indices, corpus_top_k)
                    gt = rank_from_cols(self_scores, col_indices, kept_docids,
                                        query_indices, top_k, match_mode)
                    matches: List[str] = []
                    for cand_alias in candidates:
                        cs = candidate_scores.get(cand_alias)
                        if cs is None:
                            continue
                        cand = rank_from_cols(cs, col_indices, kept_docids,
                                              query_indices, top_k, match_mode)
                        if cand.shape == gt.shape and bool(np.array_equal(cand, gt)):
                            matches.append(cand_alias)
                    pred = predict_from_matches(sorted(matches))
                    is_known = eval_alias in known_set
                    truth = eval_alias if is_known else "UNK"
                    key = RunKey(setup, int(top_k), int(corpus_top_k), match_mode,
                                 int(qb), int(repeat_id), query_indices_str)
                    acc = run_counts.setdefault(key, _new_counts())
                    _update_counts(acc, is_known, pred, pred == truth)
    return run_counts


def _merge_counts(dicts) -> Dict[RunKey, Dict[str, int]]:
    merged: Dict[RunKey, Dict[str, int]] = {}
    for d in dicts:
        for k, c in d.items():
            if k not in merged:
                merged[k] = dict(c)
            else:
                m = merged[k]
                for f, v in c.items():
                    m[f] += v
    return merged


def run_evaluate(
    cache_root: Path,
    interface: str,
    top_ks: Sequence[int],
    budgets: Sequence[int],
    n_repeats: int,
    candidates: Sequence[str],
    corpus_policy: str = "same_as_top_k",
    seed: int = 1337,
    enumerate_when_leq_repeats: bool = True,
    targets: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, raw_df). Pure compute; no files written.

    n_jobs>1 splits targets across processes; integer counts are summed per run,
    so the result is identical to the serial computation.
    """
    if max(top_ks) > 50:
        raise ValueError("evaluator assumes top_k <= 50 (victim_full_top50 cache)")
    cache_root = Path(cache_root)
    target_dirs = list_target_dirs(cache_root)
    if targets is not None:
        want = {canonical_alias(t) for t in targets}
        target_dirs = [p for p in target_dirs if canonical_alias(target_alias_of(p)) in want]
    if not target_dirs:
        raise SystemExit(f"No target dirs found under {cache_root}")
    all_targets = [target_alias_of(p) for p in target_dirs]

    first_top50 = np.load(target_dirs[0] / "victim_full_top50_docids.npy", allow_pickle=True)
    q_eff_first = int(first_top50.shape[0])
    subsets_by_qb = _generate_subsets(q_eff_first, budgets, n_repeats, seed,
                                      enumerate_when_leq_repeats)

    if n_jobs <= 1 or len(all_targets) <= 1:
        counts = accumulate_counts(cache_root, interface, top_ks, candidates,
                                   corpus_policy, subsets_by_qb, all_targets)
    else:
        import concurrent.futures as cf
        n = min(n_jobs, len(all_targets))
        chunks = [all_targets[i::n] for i in range(n)]  # round-robin balance
        with cf.ProcessPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(accumulate_counts, cache_root, interface, top_ks,
                              candidates, corpus_policy, subsets_by_qb, ch)
                    for ch in chunks if ch]
            counts = _merge_counts([f.result() for f in futs])

    if not counts:
        raise SystemExit("Nothing evaluated.")
    raw = pd.DataFrame([_counts_to_metrics(k, v) for k, v in counts.items()])
    raw = raw.sort_values(["setup", "query_budget", "repeat_id", "top_k"])
    return summarize(raw), raw


# --- CLI ---------------------------------------------------------------------
def _parse_int_list(raw: str) -> List[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--interface", required=True, choices=["tm1", "tm2"])
    ap.add_argument("--queries", type=Path, help="Query CSV (its stem names the query set).")
    ap.add_argument("--score-cache", type=Path, default=None,
                    help="Score cache root. Default: $TELLTAIL_OUT_DIR/generic/<query_set>/scores")
    ap.add_argument("--top-ks", default="1,2,3,5,10,20,50")
    ap.add_argument("--budgets", default="1,5,10,12,15,18,20")
    ap.add_argument("--n-repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="same_as_top_k",
                    choices=["same_as_top_k", "fixed50"])
    ap.add_argument("--no-enumerate", action="store_true",
                    help="Disable enumerate_when_leq_repeats (default: enabled).")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes over targets.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write results. Default: generic/<query_set>/results")


def _resolve_paths(args):
    from .paths import generic_dir, query_set_name
    qs = query_set_name(args.queries) if args.queries else "adhoc"
    cache = args.score_cache or (generic_dir(qs) / "scores")
    out = args.out_dir or (generic_dir(qs) / "results")
    return qs, Path(cache), Path(out)


def main(args) -> None:
    from telltail.models import candidates as reg_candidates
    qs, cache, out = _resolve_paths(args)
    out.mkdir(parents=True, exist_ok=True)
    summary, raw = run_evaluate(
        cache_root=cache, interface=args.interface,
        top_ks=_parse_int_list(args.top_ks), budgets=_parse_int_list(args.budgets),
        n_repeats=args.n_repeats, candidates=sorted(reg_candidates()),
        corpus_policy=args.corpus, seed=args.seed,
        enumerate_when_leq_repeats=not args.no_enumerate, n_jobs=args.n_jobs)
    setup, match_mode = INTERFACE_TO_SETUP[args.interface]
    prefix = f"{args.interface}_{qs}_{match_mode}_{args.corpus}"
    raw.to_csv(out / f"{prefix}_raw_runs.csv", index=False)
    summary.to_csv(out / f"{prefix}_summary.csv", index=False)
    (out / f"{prefix}_config.json").write_text(json.dumps({
        "interface": args.interface, "setup": setup, "match_mode": match_mode,
        "query_set": qs, "score_cache": str(cache), "top_ks": args.top_ks,
        "budgets": args.budgets, "n_repeats": args.n_repeats, "seed": args.seed,
        "corpus_policy": args.corpus, "enumerate_when_leq_repeats": not args.no_enumerate,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[evaluate] {prefix}: wrote summary ({len(summary)} rows) to {out}")
