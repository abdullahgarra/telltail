#!/usr/bin/env python3
"""Plot ASR vs query budget, one line per top-k (paper Fig. 3 / Fig. 5, generic).

Merges the two per-interface ASR plotters into one. Reads a `*_qb_k.csv`
summary (default: the golden CSVs in results/paper/).

    python plots/asr_vs_budget.py --tm 1                 # topic + random
    python plots/asr_vs_budget.py --tm 2 --probe topic
    python plots/asr_vs_budget.py --tm 1 --csv path/to/summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PAPER = Path(__file__).resolve().parent.parent / "results" / "paper"


def plot_one(ax, df: pd.DataFrame, title: str) -> None:
    for top_k, g in df.groupby("top_k", sort=True):
        g = g.sort_values("query_budget")
        x = g["query_budget"].to_numpy()
        y = g["asr_mean"].to_numpy()
        ci = g["asr_ci95"].fillna(0).to_numpy() if "asr_ci95" in g else 0
        ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=f"top-{int(top_k)}")
        ax.fill_between(x, y - ci, y + ci, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Query budget |Q|")
    ax.set_ylabel("Attack success rate")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, ncol=2, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, choices=[1, 2], required=True)
    ap.add_argument("--probe", choices=["topic", "random", "both"], default="both")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Explicit summary CSV (overrides --probe / paper defaults).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    probes = ["topic", "random"] if args.probe == "both" and args.csv is None else \
             ([args.probe] if args.csv is None else ["(csv)"])
    fig, axes = plt.subplots(1, len(probes), figsize=(6.5 * len(probes), 5.0), squeeze=False)
    for ax, probe in zip(axes[0], probes):
        csv = args.csv if args.csv else PAPER / f"tm{args.tm}_telltail_{probe}_qb_k.csv"
        df = pd.read_csv(csv)
        plot_one(ax, df, f"TM{args.tm} — {probe}")
    fig.tight_layout()
    out = args.out or (PAPER.parent / f"asr_vs_budget_tm{args.tm}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
