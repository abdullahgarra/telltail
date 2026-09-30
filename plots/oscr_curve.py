#!/usr/bin/env python3
"""Plot OSCR curves (CCR vs FAR) per top-k from an oscr_curve CSV, and print the
paper metrics CCR@FA<=0, CCR@FA<=2, AUOSCR (paper Fig. 4 / App.).

    python plots/oscr_curve.py --tm 1 --probe topic
    python plots/oscr_curve.py --tm 2 --csv path/to/oscr_curve.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent / "results" / "paper"


def _auoscr(g: pd.DataFrame) -> float:
    df = g.dropna(subset=["far", "ccr"]).sort_values("far")
    if len(df) < 2 or df["far"].nunique() < 2:
        return float("nan")
    gg = df.groupby("far", as_index=False)["ccr"].max().sort_values("far")
    far, ccr = gg["far"].to_numpy(), gg["ccr"].to_numpy()
    if far[0] > 0:
        far, ccr = np.r_[0.0, far], np.r_[ccr[0], ccr]
    if far[-1] < 1:
        far, ccr = np.r_[far, 1.0], np.r_[ccr, ccr[-1]]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(ccr, far))


def _ccr_at_fa(g: pd.DataFrame, b: int) -> float:
    f = g[g["false_accepts"] <= b]
    return float(f["ccr"].max()) if len(f) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, choices=[1, 2], required=True)
    ap.add_argument("--probe", choices=["topic", "random"], default="topic")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    csv = args.csv or PAPER / f"tm{args.tm}_telltail_{args.probe}_oscr_curve.csv"
    df = pd.read_csv(csv)
    print(f"{csv.name}:  k   CCR@FA<=0   CCR@FA<=2   AUOSCR")
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for k, g in df.groupby("k", sort=True):
        g = g.sort_values("far")
        ax.plot(g["far"], g["ccr"], marker=".", label=f"k={int(k)}")
        print(f"           {int(k):>3} {_ccr_at_fa(g,0):>10.4f} {_ccr_at_fa(g,2):>10.4f} {_auoscr(g):>8.4f}")
    ax.set_xlabel("False acceptance rate")
    ax.set_ylabel("Correct classification rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2, fontsize=8)
    ax.set_title(f"TM{args.tm} — {args.probe} OSCR")
    fig.tight_layout()
    out = args.out or (PAPER.parent / f"oscr_tm{args.tm}_{args.probe}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
