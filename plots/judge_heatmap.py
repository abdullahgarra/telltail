#!/usr/bin/env python3
"""TM3 (response-only) LLM-judge heatmap: candidate x evaluated-model.

Reads judge verdicts and plots, per (attack_model, eval_model), the rate at which the
judge labelled the generated response as "the retrieved context is exclusively on the
hidden topic" — i.e. the response-only fingerprint signal. Cell = true_count / observed
query groups (10 groups -> all-top-3-on-topic rate). The self-attack diagonal is boxed.

Input (``--judgments`` dir): the slim ``judgements.jsonl`` shipped under
``results/opt/tm3_judgments/`` (fields: attack_model, eval_model, query_group, judgement),
or any ``*.jsonl`` / ``all_judged.jsonl`` with those fields.

    python plots/judge_heatmap.py                       # defaults below
    python plots/judge_heatmap.py --judgments <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_JUDGMENTS = REPO / "results" / "opt" / "tm3_judgments"
# Default under outputs/ (gitignored) so a Section-C run doesn't overwrite the committed
# paper figure in results/paper/ or dirty the working tree. Pass --out to override.
DEFAULT_OUT = REPO / "outputs" / "plots"


def _find_files(d: Path):
    if (d / "all_judged.jsonl").exists():
        return [d / "all_judged.jsonl"]
    if (d / "per_eval_model").exists():
        f = sorted((d / "per_eval_model").glob("*.judged.jsonl"))
        if f:
            return f
    f = sorted(d.glob("*.jsonl"))
    if f:
        return f
    raise FileNotFoundError(f"no judgment JSONL files in {d}")


def load_judgements(d: Path) -> pd.DataFrame:
    rows = []
    for path in _find_files(d):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            atk, ev = str(r.get("attack_model", "")), str(r.get("eval_model", ""))
            if not ev and path.name.endswith(".judged.jsonl"):
                ev = path.name.replace(".judged.jsonl", "")
            try:
                qg = int(r.get("query_group"))
            except Exception:
                continue
            j = str(r.get("judgement", "")).strip().lower()
            if not atk or not ev or j not in {"true", "false"}:
                continue
            rows.append(dict(attack_model=atk, eval_model=ev, query_group=qg,
                             success=int(j == "true")))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no valid judged rows")
    return df.drop_duplicates(["attack_model", "eval_model", "query_group"], keep="first")


def _sort_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def build_matrix(df: pd.DataFrame):
    g = (df.groupby(["attack_model", "eval_model"], as_index=False)
           .agg(true_count=("success", "sum"), observed=("query_group", "nunique")))
    g["rate"] = g.true_count / g.observed
    mat = g.pivot(index="attack_model", columns="eval_model", values="rate")
    shared = sorted(set(mat.index) & set(mat.columns), key=_sort_key)
    rows = shared + sorted([x for x in mat.index if x not in shared], key=_sort_key)
    cols = shared + sorted([x for x in mat.columns if x not in shared], key=_sort_key)
    return g, mat.reindex(index=rows, columns=cols)


def plot(mat: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 20})
    sns.set_style("darkgrid")
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(max(13, n_cols * 0.9), max(15, n_rows * 0.75)))
    sns.heatmap(mat, ax=ax, cmap="Blues", vmin=0, vmax=1, linewidths=0.5,
                linecolor="white", annot=True, fmt=".1f", annot_kws={"size": 11}, cbar=False)
    for i, atk in enumerate(mat.index):           # box the self-attack diagonal
        if atk in mat.columns:
            j = list(mat.columns).index(atk)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="cyan", lw=2))
    ax.set_xlabel("Evaluated model", fontweight="bold", fontsize=26)
    ax.set_ylabel("Candidate model", fontweight="bold", fontsize=26)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight="bold")
    plt.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 250})):
        fig.savefig(out / f"heatmap_llm_judge_hp_rate{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="TM3 LLM-judge heatmap")
    ap.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    df = load_judgements(args.judgments)
    g, mat = build_matrix(df)
    args.out.mkdir(parents=True, exist_ok=True)
    mat.to_csv(args.out / "matrix_llm_judge_hp_rate.csv")
    plot(mat, args.out)
    self_rate = np.nanmean([mat.loc[m, m] for m in mat.index if m in mat.columns])
    print(f"[judge-heatmap] {len(df)} rows | pairs {len(g)} | mean self-rate {self_rate:.3f}")
    print(f"[judge-heatmap] -> {args.out}/heatmap_llm_judge_hp_rate.(png|pdf) + matrix csv")


if __name__ == "__main__":
    main()
