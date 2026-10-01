"""`python -m demo fingerprint` — fingerprint the demo victim retrievers.

Builds one small index per victim from the shipped demo corpus (demo/data/corpus.jsonl),
evaluates the paper's ready optimized queries (demo/data/ready_queries.csv) against each,
and predicts which candidate each victim is. CPU-runnable; indices cache after first build.

Score: S[candidate, victim] = mean over that candidate's queries of 1[target rank <= k]
(TM2: rank<=k; TM1 via --tm 1: rank<=min(3,k)). Prediction: unique S>0.5 -> that candidate,
none -> UNK, several -> ABSTAIN. Output: a prediction table (victim x k) + a top-3-rate
heatmap (candidates x victims, predicted cell outlined).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .index import load_corpus, build_index

DATA = Path(__file__).resolve().parent / "data"
DEMO_VICTIMS = ["minilm-l6", "minilm-l12", "e5-small", "multilingual-e5-small"]
KS = [1, 3, 5, 10]
K_HEAT = 3
TAU = 0.5


def _hit_thr(k: int, tm: str) -> int:
    return min(3, k) if tm == "tm1" else k


def _predict(col: pd.Series, candidates: set) -> str:
    above = sorted([a for a in col.index if a in candidates and col[a] > TAU])
    return above[0] if len(above) == 1 else ("UNK" if not above else "ABSTAIN")


def _out_dir() -> Path:
    base = os.environ.get("TELLTAIL_OUT_DIR")
    d = (Path(base) / "demo") if base else (Path(__file__).resolve().parents[1] / "_local/demo")
    d.mkdir(parents=True, exist_ok=True)
    return d


def evaluate_victims(victims, device=None):
    """Return the long demo results (query_id, attack_model, passage_id, rank, eval_model)."""
    from sentence_transformers import SentenceTransformer
    from opt.evaluate import build_eval_query, find_rank_oss
    from generic_queries._common import normalize_docid
    from telltail.models import load_registry

    ids, texts = load_corpus(DATA / "corpus.jsonl")
    ready = list(csv.DictReader(open(DATA / "ready_queries.csv")))
    reg = load_registry()
    rows = []
    for v in victims:
        print(f"[demo] building/loading index for victim '{v}' ({len(ids)} passages)...", flush=True)
        index, docids = build_index(v, ids, texts, device=device, use_cache=True)
        model = SentenceTransformer(reg[v]["hf_id"], trust_remote_code=True, device=device)
        prefix = reg[v].get("query_prefix", "")
        is_oai = reg[v].get("backend") == "openai"
        for r in ready:
            eq = build_eval_query(prefix, r["query_text"], r["trigger_suffix"], is_oai)
            rank = find_rank_oss(model, eq, normalize_docid(r["passage_id"]), index, docids)
            rows.append(dict(query_id=r["query_id"], attack_model=r["attack_model"],
                             passage_id=r["passage_id"], rank=rank, eval_model=v))
        del model
    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df


def heatmap(S: np.ndarray, candidates, victims, preds, path: Path, k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(victims), 0.4 * len(candidates) + 1),
                           constrained_layout=True)
    ax.imshow(S, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(victims))); ax.set_xticklabels(victims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(candidates))); ax.set_yticklabels(candidates, fontsize=7)
    ax.set_xlabel("Victim model"); ax.set_ylabel("Candidate model")
    ax.set_title(f"TellTail-OPT demo — top-{k} rate")
    for i in range(len(candidates)):
        for j in range(len(victims)):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if S[i, j] > 0.5 else "black")
    for j, v in enumerate(victims):
        p = preds.get(v)
        if p in candidates:
            ax.add_patch(plt.Rectangle((j - .5, candidates.index(p) - .5), 1, 1,
                                       fill=False, edgecolor="red", lw=2))
    fig.savefig(path, dpi=150)
    return path


def run(victims=None, tm="tm2", device=None):
    from opt.score import S_for
    victims = victims or DEMO_VICTIMS
    df = evaluate_victims(victims, device=device)
    candidates = sorted(df.attack_model.unique())
    cset = set(candidates)

    print("\n=== prediction (victim x k) ===")
    print("  " + f"{'victim':<24}" + "".join(f"{'k='+str(k):<14}" for k in KS))
    heat_preds = {}
    for v in victims:
        d = df[df.eval_model == v]
        line = f"  {v:<24}"
        for k in KS:
            s = S_for(d, _hit_thr(k, tm))[v].fillna(0.0)
            line += f"{_predict(s, cset):<14}"
            if k == K_HEAT:
                heat_preds[v] = _predict(s, cset)
        print(line)

    # heatmap at top-K_HEAT
    S = np.zeros((len(candidates), len(victims)))
    for j, v in enumerate(victims):
        s = S_for(df[df.eval_model == v], _hit_thr(K_HEAT, tm))[v]
        for i, a in enumerate(candidates):
            S[i, j] = float(s.get(a, 0.0))
    png = heatmap(S, candidates, victims, heat_preds, _out_dir() / "demo_fingerprint_heatmap.png", _hit_thr(K_HEAT, tm))
    print(f"\n[demo] heatmap -> {png}")


def add_arguments(ap: argparse.ArgumentParser):
    ap.add_argument("--tm", choices=["1", "2"], default="2", help="threat model (default 2=unordered)")
    ap.add_argument("--victims", default="", help="comma-separated victim aliases (default: the demo set)")
    ap.add_argument("--device", default=None, help="torch device (default: auto; CPU works)")


def main(args):
    victims = [v.strip() for v in args.victims.split(",") if v.strip()] or None
    run(victims=victims, tm=f"tm{args.tm}", device=args.device)
