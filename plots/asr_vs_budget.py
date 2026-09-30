#!/usr/bin/env python3
"""ASR vs query budget — one panel per query-generation strategy
(TellTail-Random | TellTail-Topic [| TellTail-OPT]), one line per retrieval
depth k. Paper styling.

By default reads the published CSVs for a threat model:
    python plots/asr_vs_budget.py --tm 2
    python plots/asr_vs_budget.py --tm 1 --ks 1 2 3 5 10 20 50
OPT is included automatically if results/paper/tm{tm}_telltail_opt_qb_k.csv exists;
otherwise the figure has the two generic-query panels.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PAPER = Path(__file__).resolve().parent.parent / "results" / "paper"

RCPARAMS = {
    "font.family": "serif", "font.size": 15, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
    "lines.linewidth": 2.5, "lines.markersize": 8, "axes.linewidth": 2.0,
    "xtick.major.width": 2.0, "ytick.major.width": 2.0,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
PANELS = [("TellTail-Random", "random"), ("TellTail-Topic", "topic"),
          ("TellTail-OPT", "opt")]


def load(path, label):
    df = pd.read_csv(path)
    for c in ("top_k", "query_budget", "asr_mean"):
        assert c in df.columns, f"{label}: missing '{c}'"
    return df


def red_palette(ks):
    light, dark = np.array([0.90, 0.50, 0.50]), np.array([0.45, 0.0, 0.0])
    return {k: tuple(light + (i / max(len(ks) - 1, 1)) * (dark - light))
            for i, k in enumerate(sorted(ks))}


def marker_map(ks):
    markers = ["s", "o", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    return {k: markers[i % len(markers)] for i, k in enumerate(sorted(ks))}


def resolve_panels(tm, random_csv, topic_csv, opt_csv):
    explicit = {"random": random_csv, "topic": topic_csv, "opt": opt_csv}
    out = []
    for label, key in PANELS:
        p = explicit[key] or (PAPER / f"tm{tm}_telltail_{key}_qb_k.csv")
        if p and Path(p).exists():
            out.append((label, load(p, key), key))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, choices=[1, 2], help="Resolve CSVs from results/paper/.")
    ap.add_argument("--random"); ap.add_argument("--topic"); ap.add_argument("--opt")
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output stem (no ext). Default: results/asr_vs_budget_tm{tm}")
    args = ap.parse_args()

    panels = resolve_panels(args.tm, args.random, args.topic, args.opt)
    if not panels:
        raise SystemExit("no input CSVs found (pass --tm or --random/--topic).")
    all_ks = sorted(set.intersection(*[set(df["top_k"].unique()) for _, df, _ in panels]))
    ks = sorted(args.ks) if args.ks else all_ks
    palette, markers = red_palette(ks), marker_map(ks)

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.12)
    for i, (ax, (label, df, key)) in enumerate(zip(axes, panels)):
        d = df[df["top_k"].isin(ks)]
        for k in sorted(ks):
            sub = d[d["top_k"] == k].sort_values("query_budget")
            ax.plot(sub["query_budget"], sub["asr_mean"], color=palette[k],
                    marker=markers[k], label=f"$k={k}$", markerfacecolor=palette[k],
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel(r"Queries per Candidate $|Q_R|$" if key == "opt"
                      else r"Query Budget $|Q|$")
        ax.set_xlim(left=0); ax.set_ylim(0, 1.05); ax.set_xticks([1, 5, 10, 15, 20])
        ax.xaxis.set_tick_params(rotation=45)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(label, fontsize=15, pad=6)
    axes[0].set_ylabel("Attack Success Rate")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="0.8",
                   borderpad=0.6, handlelength=1.8, fontsize=13)
    fig.tight_layout()
    out = args.out or str((PAPER.parent / f"asr_vs_budget_tm{args.tm}"))
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    print(f"Saved {out}.pdf and {out}.png  (panels: {[k for _,_,k in panels]}, ks={ks})")


if __name__ == "__main__":
    main()
