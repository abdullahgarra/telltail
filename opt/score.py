"""TellTail-OPT — Stage 3 scorer (TM2: unordered top-k / OSCR + ASR-by-budget).

Reads the long evaluation CSV (query_id, attack_model, passage_id, rank, error,
eval_model) and produces the TM2 open-set metrics:

  * OSCR curves (per k): S[a,m] = mean_q 1[rank(a's trigger on m) <= k]; knownness =
    max_a S[.,m]; a victim m is "correctly identified" iff its unique argmax candidate
    equals m. Threshold-sweep over the knownness score gives CCR vs FAR.
  * ASR-by-budget: over query subsets of each budget, predict per victim (unique
    S>0.5 -> that model, none -> UNK, multiple -> ABSTAIN); ASR = mean correctness.

This reproduces the research `openAUC.py` / `eval_telltail_setup2_asr_by_query_budget.py`
exactly (verified: OSCR maxΔccr<=1.1e-16, Δfalse_accepts=0; ASR Δ=0 on enumerated
budgets, |Δ|<1e-12 on RNG-sampled). The `gtr-t5 -> gtr-t5-base` remap is applied to BOTH
attack_model and eval_model (else the gtr-t5-base diagonal is lost).

Usage:
    python -m opt.score --long <long.csv> --out <dir> [--method ro_topic]
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
NREP, SEED, TAU = 500, 1337, 0.5


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    for col in ("attack_model", "eval_model"):
        df[col] = df[col].replace(ALIAS_REMAP).astype(str)
    return df


def S_for(df: pd.DataFrame, k: int) -> pd.DataFrame:
    t = df[["attack_model", "eval_model", "rank"]].copy()
    t["hit"] = (t["rank"] <= k).astype(float)
    return t.pivot_table(index="attack_model", columns="eval_model", values="hit", aggfunc="mean")


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


def gen_subsets(q_eff, qb, rng):
    if qb >= q_eff:
        return [tuple(range(q_eff))]
    if math.comb(q_eff, qb) <= NREP:
        return list(itertools.combinations(range(q_eff), qb))
    return [tuple(sorted(rng.choice(q_eff, size=qb, replace=False).tolist())) for _ in range(NREP)]


def asr_for_S(S: pd.DataFrame) -> float:
    known = set(S.index.astype(str))
    nc = tot = 0
    for m in S.columns.astype(str):
        col = S[m].dropna()
        above = sorted(col[col > TAU].index.astype(str))
        pred = above[0] if len(above) == 1 else ("UNK" if len(above) == 0 else "ABSTAIN")
        truth = m if m in known else "UNK"
        nc += int(pred == truth); tot += 1
    return nc / tot if tot else np.nan


def score_oscr(df: pd.DataFrame, method: str, setup: str) -> pd.DataFrame:
    out = []
    for k in KS:
        cur = oscr_curve(S_for(df, k))
        cur.insert(0, "k", k); cur.insert(0, "setup", setup); cur.insert(0, "method", method)
        cur["accept_rule"] = "gt"
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def score_asr_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    qids = sorted(df["query_id"].unique())
    q_eff = len(qids)
    qid_arr = np.array(qids)
    rng = np.random.default_rng(SEED)
    subsets_by_qb = {qb: gen_subsets(q_eff, qb, rng) for qb in sorted(BUDGETS)}
    rows = []
    for k in KS:
        hit = df.copy(); hit["hit"] = (hit["rank"] <= k).astype(float)
        for qb in sorted(BUDGETS):
            subs = subsets_by_qb[qb]
            vals = []
            for sub in subs:
                qsel = set(qid_arr[list(sub)])
                S = (hit[hit.query_id.isin(qsel)]
                     .pivot_table(index="attack_model", columns="eval_model", values="hit", aggfunc="mean"))
                vals.append(asr_for_S(S))
            rows.append(dict(top_k=k, query_budget=qb, n_runs=len(subs),
                             asr_mean=float(np.mean(vals)), asr_std=float(np.std(vals)),
                             threshold=TAU))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT TM2 scorer (OSCR + ASR-by-budget)")
    ap.add_argument("--long", type=Path, required=True, help="evaluation long CSV")
    ap.add_argument("--out", type=Path, required=True, help="output dir")
    ap.add_argument("--method", default="ro", help="method name for the OSCR CSV")
    ap.add_argument("--setup", default="baseline", help="setup name for the OSCR CSV")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_long(args.long)
    oscr = score_oscr(df, args.method, args.setup)
    asr = score_asr_by_budget(df)
    oscr.to_csv(args.out / "setup2_oscr_curves.csv", index=False)
    asr.to_csv(args.out / "setup2_asr_by_budget_summary.csv", index=False)
    print(f"[score] wrote OSCR ({len(oscr)} rows) + ASR-by-budget ({len(asr)} rows) -> {args.out}")


if __name__ == "__main__":
    main()
