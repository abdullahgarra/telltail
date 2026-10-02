"""TellTail-OPT Stage 3 (TM1/TM2): open-set scorer (OSCR + ASR-by-budget).

Reads the long eval CSV (query_id, attack_model, passage_id, rank, error, eval_model). The
only TM difference is the hit rule: TM2 hit = rank<=k; TM1 hit = rank<=min(3,k) (ordered —
only the top-3 exposed positions count). Reproduces the paper goldens
(results/paper/tm{1,2}_telltail_opt_*.csv) exactly.

    python -m opt.score --tm 1 --long <long.csv> --out <dir>
    python -m opt.score --tm 2 --long <long.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ALIAS_REMAP = {"gtr-t5": "gtr-t5-base"}
KS = [1, 2, 3, 4, 5, 10, 20, 50]
BUDGETS = [1, 5, 10, 12, 15, 18, 20]
SEED, TAU = 1337, 0.5
NREP_BY_TM = {"tm1": 200, "tm2": 500}
METRICS = ["asr", "known_acc", "unknown_acc", "abstain_rate",
           "pred_known_rate", "pred_unk_rate", "pred_abstain_rate"]
OSCR_METHOD = "optimization_fingerprinting"
QB_METHOD = "telltail-opt"


def hit_threshold(k: int, tm: str) -> int:
    return min(3, k) if tm == "tm1" else k


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    for col in ("attack_model", "eval_model"):
        df[col] = df[col].replace(ALIAS_REMAP).astype(str)
    return df


def S_for(df: pd.DataFrame, thr: int) -> pd.DataFrame:
    t = df[["attack_model", "eval_model", "rank"]].copy()
    t["hit"] = (t["rank"] <= thr).astype(float)
    return t.pivot_table(index="attack_model", columns="eval_model", values="hit", aggfunc="mean")


# ---------------------------------------------------------------- OSCR ------- #
def oscr_curve(S: pd.DataFrame) -> pd.DataFrame:
    known = set(S.index.astype(str))
    recs = []
    for m in S.columns.astype(str):
        col = S[m].dropna()
        mx = float(col.max()) if len(col) else np.nan
        tied = col[col == mx]
        icc = bool(m in known and len(tied) == 1 and tied.index[0] == m)
        recs.append((m in known, icc, mx))
    known_mask = np.array([r[0] for r in recs])
    correct = np.array([r[1] for r in recs])
    scores = np.array([r[2] for r in recs], dtype=float)
    ok = ~np.isnan(scores)
    known_mask, correct, scores = known_mask[ok], correct[ok], scores[ok]
    nk, nu = int(known_mask.sum()), int((~known_mask).sum())
    taus = np.unique(scores)
    taus = np.concatenate(([np.inf], taus[::-1], [-np.inf]))
    rows = []
    for tau in taus:
        accept = scores > tau
        rows.append(dict(
            threshold=float(tau),
            ccr=float(((accept & known_mask) & correct).sum() / nk) if nk else np.nan,
            far=float((accept & ~known_mask).sum() / nu) if nu else np.nan,
            false_accepts=int((accept & ~known_mask).sum()),
            known_accept_rate=float((accept & known_mask).sum() / nk) if nk else np.nan,
        ))
    return pd.DataFrame(rows).sort_values("far", kind="mergesort").reset_index(drop=True)


def score_oscr(df: pd.DataFrame, tm: str) -> pd.DataFrame:
    setup = "setup1" if tm == "tm1" else "setup2"
    out = []
    for k in KS:
        cur = oscr_curve(S_for(df, hit_threshold(k, tm)))
        cur.insert(0, "k", k); cur.insert(0, "setup", setup); cur.insert(0, "method", OSCR_METHOD)
        out.append(cur)
    return pd.concat(out, ignore_index=True)[
        ["method", "setup", "k", "threshold", "ccr", "far", "false_accepts", "known_accept_rate"]]


# ---------------------------------------------------- ASR-by-budget ---------- #
def gen_subsets(q_eff, qb, nrep, rng):
    if qb >= q_eff:
        return [tuple(range(q_eff))]
    if math.comb(q_eff, qb) <= nrep:
        return list(itertools.combinations(range(q_eff), qb))
    return [tuple(sorted(rng.choice(q_eff, size=qb, replace=False).tolist())) for _ in range(nrep)]


def run_counts(S: pd.DataFrame) -> dict:
    """Per-run open-set metrics from one subset's hit matrix S (attack x eval)."""
    known = set(S.index.astype(str))
    nt = nk = nu = nc = nkc = nuc = pk = pu = pa = 0
    for m in S.columns.astype(str):
        col = S[m].dropna()
        above = sorted(col[col > TAU].index.astype(str))
        pred = above[0] if len(above) == 1 else ("UNK" if len(above) == 0 else "ABSTAIN")
        isk = m in known
        truth = m if isk else "UNK"
        nt += 1; nk += int(isk); nu += int(not isk)
        if pred == truth:
            nc += 1; nkc += int(isk); nuc += int(not isk)
        if pred == "UNK":
            pu += 1
        elif pred == "ABSTAIN":
            pa += 1
        else:
            pk += 1
    return dict(asr=nc / nt, known_acc=nkc / nk, unknown_acc=nuc / nu,
                abstain_rate=pa / nt, pred_known_rate=pk / nt,
                pred_unk_rate=pu / nt, pred_abstain_rate=pa / nt)


def _agg(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    sem = std / math.sqrt(len(a)) if len(a) else np.nan
    return dict(mean=mean, std=std, sem=sem, ci95=1.96 * sem)


def score_qb_k(df: pd.DataFrame, tm: str):
    qids = sorted(df["query_id"].unique())
    q_eff = len(qids)
    qid_arr = np.array(qids)
    nrep = NREP_BY_TM[tm]
    rng = np.random.default_rng(SEED)
    subsets_by_qb = {qb: gen_subsets(q_eff, qb, nrep, rng) for qb in sorted(BUDGETS)}
    n_total = int(df["eval_model"].nunique())
    cand = set(df["attack_model"].unique())
    n_known = len([m for m in df["eval_model"].unique() if m in cand])
    n_unknown = n_total - n_known
    if n_known == 0 or n_unknown == 0:
        raise SystemExit(
            f"opt.score is open-set: it needs >=2 eval models including at least one "
            f"unknown (non-candidate). Got {n_known} known / {n_unknown} unknown. "
            f"Pass a long CSV covering multiple models — a single-model file can't be scored.")

    # Aggregate per (hit-threshold, budget) ONCE; k's that share a threshold reuse it.
    agg_by_thr_qb = {}
    for thr in sorted({hit_threshold(k, tm) for k in KS}):
        hit = df.copy(); hit["hit"] = (hit["rank"] <= thr).astype(float)
        for qb in sorted(BUDGETS):
            subs = subsets_by_qb[qb]
            per = {mt: [] for mt in METRICS}
            for sub in subs:
                qsel = set(qid_arr[list(sub)])
                S = (hit[hit.query_id.isin(qsel)]
                     .pivot_table(index="attack_model", columns="eval_model", values="hit", aggfunc="mean"))
                rc = run_counts(S)
                for mt in METRICS:
                    per[mt].append(rc[mt])
            agg_by_thr_qb[(thr, qb)] = (len(subs), {mt: _agg(per[mt]) for mt in METRICS})

    rows = []
    for k in KS:
        thr = hit_threshold(k, tm)
        for qb in sorted(BUDGETS):
            n_runs, aggs = agg_by_thr_qb[(thr, qb)]
            row = {"method": QB_METHOD,
                   "setup": "setup1" if tm == "tm1" else "setup2",
                   "threshold": TAU}
            if tm == "tm1":
                row.update(threshold_rule="gt", signal=f"appeared@{thr}", appeared_at_k=thr,
                           top_k=k, corpus_top_k=50, match_mode="threshold_appeared_rate",
                           query_budget=qb, k=k, n_runs=n_runs)
            else:
                row.update(top_k=k, corpus_top_k=k, match_mode="threshold_hit_rate",
                           query_budget=qb, k=k, n_runs=n_runs)
            for mt in METRICS:
                a = aggs[mt]
                row[f"{mt}_mean"] = a["mean"]; row[f"{mt}_std"] = a["std"]
                row[f"{mt}_sem"] = a["sem"]; row[f"{mt}_ci95"] = a["ci95"]
            row.update(n_total=n_total, n_known=n_known, n_unknown=n_unknown)
            rows.append(row)
    out = pd.DataFrame(rows)

    # Assert: for TM1, k in {4,5,10,20,50} equal appeared@3 (same hit threshold).
    if tm == "tm1":
        cmp = [c for c in out.columns if c.endswith(("_mean", "_std", "_sem", "_ci95"))]
        base = out[out.top_k == 3].set_index("query_budget")[cmp]
        for k in (4, 5, 10, 20, 50):
            sub = out[out.top_k == k].set_index("query_budget")[cmp]
            assert np.allclose(sub.values, base.values, equal_nan=True), \
                f"TM1 k={k} rows differ from appeared@3"
    return out


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT scorer (TM1/TM2 OSCR + ASR-by-budget)")
    ap.add_argument("--tm", choices=["1", "2"], required=True, help="threat model (1=ordered, 2=unordered)")
    ap.add_argument("--long", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tm = f"tm{args.tm}"
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_long(args.long)
    oscr = score_oscr(df, tm)
    qb = score_qb_k(df, tm)
    oscr.to_csv(args.out / f"{tm}_telltail_opt_oscr_curve.csv", index=False)
    qb.to_csv(args.out / f"{tm}_telltail_opt_qb_k.csv", index=False)
    print(f"[score {tm}] OSCR {len(oscr)} rows, qb_k {len(qb)} rows -> {args.out}")


if __name__ == "__main__":
    main()
