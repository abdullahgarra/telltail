#!/usr/bin/env python3
"""Canonical OSCR metrics + plotting: CCR@FA<=0, CCR@FA<=2, and AUOSCR.

**AUOSCR is the monotone-envelope + exact step integration** (a left Riemann sum
of the non-decreasing CCR-vs-FAR envelope), *not* trapezoidal quadrature. On a
diagonal jump (a threshold where an unknown and a correct known are accepted
together), trapezoid would add a spurious triangle; the step integral does not.
These are the definitions used for the paper's OSCR figures — import them; do not
reimplement.

    python plots/plot_ccr_0_oscr.py --tm 2 --probe topic
    python plots/plot_ccr_0_oscr.py --tm 1 --csv path/to/oscr_curve.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent / "results" / "paper"


def auoscr(curve_k: pd.DataFrame) -> float:
    """Area under the OSCR curve for one k: monotone envelope + step integration.

    Sort by FAR, take the running-max (monotone) CCR envelope, pad to FAR in
    [0,1], then integrate as a step function held at the left endpoint of each
    FAR interval: ``sum(ccr_env[:-1] * diff(far))``.
    """
    df = curve_k.dropna(subset=["far", "ccr"]).sort_values(
        "far", kind="mergesort").reset_index(drop=True)
    if len(df) < 2 or df["far"].nunique() < 2:
        return float("nan")
    far = df["far"].to_numpy(dtype=float)
    ccr = np.maximum.accumulate(df["ccr"].to_numpy(dtype=float))  # monotone envelope
    if far[0] > 0:
        far = np.concatenate(([0.0], far))
        ccr = np.concatenate(([0.0], ccr))   # CCR=0 at FAR=0 (no accepts)
    if far[-1] < 1:
        far = np.concatenate((far, [1.0]))
        ccr = np.concatenate((ccr, [ccr[-1]]))
    return float(np.sum(ccr[:-1] * np.diff(far)))   # left-step Riemann sum


def ccr_at_fa(curve_k: pd.DataFrame, budget: int) -> float:
    """Max CCR achievable with at most ``budget`` false accepts."""
    feasible = curve_k[curve_k["false_accepts"] <= budget]
    return float(feasible["ccr"].max()) if len(feasible) else float("nan")


def paper_metrics(curve: pd.DataFrame) -> pd.DataFrame:
    """Per-k CCR@FA<=0, CCR@FA<=2, AUOSCR — the quantities the paper plots."""
    rows = []
    for k, g in curve.groupby("k", sort=True):
        rows.append(dict(k=int(k), ccr_fa0=ccr_at_fa(g, 0),
                         ccr_fa2=ccr_at_fa(g, 2), auoscr=auoscr(g)))
    return pd.DataFrame(rows)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, choices=[1, 2], required=True)
    ap.add_argument("--probe", choices=["topic", "random"], default="topic")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    csv = args.csv or PAPER / f"tm{args.tm}_telltail_{args.probe}_oscr_curve.csv"
    df = pd.read_csv(csv)
    met = paper_metrics(df)
    print(f"{csv.name}:")
    print(met.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for k, g in df.groupby("k", sort=True):
        g = g.sort_values("far")
        ax.plot(g["far"], g["ccr"], marker=".", label=f"k={int(k)}")
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("Correct classification rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2, fontsize=8)
    ax.set_title(f"TM{args.tm} — {args.probe} OSCR (AUOSCR = envelope+step)")
    fig.tight_layout()
    out = args.out or (PAPER.parent / f"oscr_tm{args.tm}_{args.probe}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
