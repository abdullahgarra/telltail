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


# --- figure (paper styling) -------------------------------------------------
import string  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

RCPARAMS = {
    "font.family": "serif", "font.size": 15, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13,
    "lines.linewidth": 2.5, "lines.markersize": 8, "axes.linewidth": 2.0,
    "xtick.major.width": 2.0, "ytick.major.width": 2.0,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
METHODS = [("TellTail-Random", "random"), ("TellTail-Topic", "topic"),
           ("TellTail-OPT", "opt")]


def _green_palette(names):
    light, dark = np.array([0.76, 0.93, 0.76]), np.array([0.0, 0.35, 0.0])
    return {n: tuple(light + (i / max(len(names) - 1, 1)) * (dark - light))
            for i, n in enumerate(names)}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", type=int, choices=[1, 2], help="Resolve CSVs from results/paper/.")
    ap.add_argument("--random"); ap.add_argument("--topic"); ap.add_argument("--opt")
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output stem (no ext). Default: results/oscr_tm{tm}")
    args = ap.parse_args()

    explicit = {"random": args.random, "topic": args.topic, "opt": args.opt}
    lines = []
    for label, key in METHODS:
        p = explicit[key] or (PAPER / f"tm{args.tm}_telltail_{key}_oscr_curve.csv")
        if p and Path(p).exists():
            lines.append((label, paper_metrics(pd.read_csv(p))))   # per-k: ccr_fa0/ccr_fa2/auoscr
    if not lines:
        raise SystemExit("no OSCR CSVs found (pass --tm or --random/--topic).")
    if args.ks:
        lines = [(n, m[m["k"].isin(args.ks)]) for n, m in lines]

    names = [n for n, _ in lines]
    palette = _green_palette(names)
    markers = {n: ["s", "^", "X", "o", "D"][i % 5] for i, n in enumerate(names)}
    cols = [("ccr_fa0", r"(a) CCR @ fa $\leq$ 0"),
            ("ccr_fa2", r"(b) CCR @ fa $\leq$ 2"),
            ("auoscr", "(c) AUOSCR")]

    plt.rcParams.update(RCPARAMS)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.7 * len(cols), 3.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(wspace=0.12)
    ks_axis = sorted(lines[0][1]["k"].tolist())
    for ax, (col, title) in zip(axes, cols):
        for name, m in lines:
            m = m.sort_values("k")
            ax.plot(m["k"], m[col], color=palette[name], marker=markers[name],
                    label=name.replace("TellTail-", ""), markerfacecolor=palette[name],
                    markeredgecolor="white", markeredgewidth=0.4,
                    markersize=12 if name == "TellTail-Random" else 8)
        ax.set_xlabel(r"Observation Cutoff $k$")
        ax.set_xscale("log"); ax.set_xticks(ks_axis)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_tick_params(rotation=45)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(title, fontsize=15, pad=6)
    axes[0].set_ylabel("Rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 1.10),
               frameon=False, handlelength=1.8, columnspacing=2.0, fontsize=13)
    fig.tight_layout()
    out = args.out or str((PAPER.parent / f"oscr_tm{args.tm}"))
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=300)
    print(f"Saved {out}.pdf and {out}.png  (methods: {names})")


if __name__ == "__main__":
    main()
