"""`python -m demo topic` — the topic-level (response-only) attack demo.

The response-only analogue of `demo fingerprint`. Each candidate ships a set of optimized
*topic* triggers (crafted toward the centroid of a topic — here Harry Potter — via
`opt.optimize --mode topic`). We build one small index per victim from the demo corpus,
run every candidate's triggers, and measure whether the victim's **top-3 retrieved
passages are all on-topic (Harry Potter)** — the signal a response-only attacker sees
(the RAG answer would be about HP). A victim is fingerprinted as the candidate whose
topic triggers reliably drive its top-3 on-topic.

Score: S[candidate, victim] = mean over that candidate's query groups of 1[all top-3
passages are HP]. Prediction: unique S>0.5 -> that candidate, none -> UNK, several ->
ABSTAIN. Output: a prediction table + a candidate x victim heatmap.

CPU-runnable; same shipped index path as `demo fingerprint` (demo/index.build_index over
demo/data/corpus.jsonl). HP is detected by keyword (approximation, as in the paper).
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
TOPK = 3
TAU = 0.5

# Harry-Potter keyword list (approximate on-topic detector; same spirit as the paper).
# Stored lowercased so matching is case-insensitive.
HP_KEYWORDS = [
    "potter", "harry potter", "the potter boy", "the boy who lived", "hermione",
    "hermione granger", "ron weasley", "weasley", "dumbledore", "albus dumbledore",
    "voldemort", "lord voldemort", "snape", "severus snape", "draco", "draco malfoy",
    "malfoy", "james potter", "hogwarts", "hogsmeade", "diagon alley", "ministry of magic",
    "azkaban", "griffindor", "gryffindor", "slytherin", "ravenclaw", "hufflepuff",
    "death eaters", "deathly hallow", "order of the phoenix", "horcrux", "wand",
    "sorting hat", "invisibility cloak", "marauder's map", "marauders map",
    "philosopher's stone", "sorcerer's stone", "patronus", "basilisk", "dementor", "dobby",
    "mcgonagall", "expelliarmus", "avada kedavra", "expecto patronum", "unforgivable curses",
    "occlumency", "legilimency", "dark arts", "potions class", "lightning scar",
    "wizarding world", "pure-blood", "pureblood", "muggle-born", "muggleborn", "half-blood",
    "halfblood", "house points", "quidditch", "golden snitch", "hagrid", "rubeus hagrid",
]


def _is_hp(text: str) -> bool:
    t = str(text).lower()
    return any(k in t for k in HP_KEYWORDS)


def _out_dir() -> Path:
    base = os.environ.get("TELLTAIL_OUT_DIR")
    d = (Path(base) / "demo") if base else (Path(__file__).resolve().parents[1] / "_local/demo")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_judge():
    """(generate_fn, judge_fn) for the --judge path. Needs OPENAI + DEEPINFRA keys.
    generate: top-3 passages -> gpt-4o-mini RAG answer (same as opt.llm).
    judge:    answer -> DeepSeek verdict (same as opt.judge)."""
    import os
    from openai import OpenAI
    from opt import config as C
    from opt.llm import build_prompt
    from opt.judge import call_judge, DEFAULT_MODEL, DEFAULT_BASE_URL
    from telltail.paths import data_dir as _  # ensures .env loaded  # noqa

    oai = OpenAI()
    dk = os.getenv("DEEPINFRA_API_KEY")
    if not dk:
        raise SystemExit("--judge needs DEEPINFRA_API_KEY (judge) and OPENAI_API_KEY (generate).")
    judge_client = OpenAI(api_key=dk, base_url=DEFAULT_BASE_URL)

    def generate(user_query, texts):
        r = oai.chat.completions.create(model=C.LLM_MODEL, temperature=C.LLM_TEMPERATURE,
                                        messages=[{"role": "user", "content": build_prompt(user_query, texts)}])
        return r.choices[0].message.content or ""

    def judge(answer):
        return call_judge(judge_client, answer, DEFAULT_MODEL)["judgement"] == "true"
    return generate, judge


def evaluate_victims(victims, device=None, judge=False):
    """Return long results: (attack_model, query_group, eval_model, all3_hp).

    Default: all3_hp = all top-3 passages are on-topic by keyword (CPU, no key).
    --judge: all3_hp = the LLM judge labels the generated RAG answer as on-topic
    (needs OPENAI_API_KEY + DEEPINFRA_API_KEY; one generate + one judge call per cell)."""
    import faiss
    from sentence_transformers import SentenceTransformer
    from opt.retrieve import build_topic_eval_query
    from telltail.models import load_registry

    ids, texts = load_corpus(DATA / "corpus.jsonl")
    id2text = dict(zip(map(str, ids), texts))
    ready = list(csv.DictReader(open(DATA / "topic_ready_queries.csv")))
    reg = load_registry()
    gen_fn = judge_fn = None
    if judge:
        gen_fn, judge_fn = _make_judge()

    rows = []
    for v in victims:
        print(f"[demo] building/loading index for victim '{v}' ({len(ids)} passages)"
              f"{' + generate/judge' if judge else ''}...", flush=True)
        index, docids = build_index(v, ids, texts, device=device, use_cache=True)
        model = SentenceTransformer(reg[v]["hf_id"], trust_remote_code=True, device=device)
        prefix = reg[v].get("query_prefix", "")
        is_oai = reg[v].get("backend") == "openai"
        for r in ready:
            eq = build_topic_eval_query(prefix, r["query_text"], r["trigger_suffix"], is_oai)
            q = model.encode([eq], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
            faiss.normalize_L2(q)
            _, I = index.search(q, TOPK)
            top = [str(docids[int(i)]) for i in I[0] if int(i) >= 0]
            if judge:
                top_texts = [id2text.get(p, "") for p in top]
                uq = f"{r['query_text']} {r['trigger_suffix']}".strip()
                all3_hp = len(top) == TOPK and judge_fn(gen_fn(uq, top_texts))
            else:
                all3_hp = len(top) == TOPK and all(_is_hp(id2text.get(p, "")) for p in top)
            rows.append(dict(attack_model=r["attack_model"], query_group=r["query_group"],
                             eval_model=v, all3_hp=bool(all3_hp)))
        del model
    return pd.DataFrame(rows)


def _predict(col: pd.Series, candidates) -> str:
    above = sorted([a for a in col.index if a in candidates and col[a] > TAU])
    return above[0] if len(above) == 1 else ("UNK" if not above else "ABSTAIN")


def heatmap(S, candidates, victims, preds, path, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(victims), 0.4 * len(candidates) + 1),
                           constrained_layout=True)
    ax.imshow(S, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(victims))); ax.set_xticklabels(victims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(candidates))); ax.set_yticklabels(candidates, fontsize=7)
    ax.set_xlabel("Victim model"); ax.set_ylabel("Candidate model")
    ax.set_title(f"TellTail topic demo — all-top{k}-HP rate")
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


def run(victims=None, device=None, judge=False):
    victims = victims or DEMO_VICTIMS
    df = evaluate_victims(victims, device=device, judge=judge)
    candidates = sorted(df.attack_model.unique())
    cset = set(candidates)

    # S[candidate, victim] = mean over candidate's groups of 1[all top-3 HP]
    S = np.zeros((len(candidates), len(victims)))
    for j, v in enumerate(victims):
        d = df[df.eval_model == v]
        rate = d.groupby("attack_model")["all3_hp"].mean()
        for i, a in enumerate(candidates):
            S[i, j] = float(rate.get(a, 0.0))
    mat = pd.DataFrame(S, index=candidates, columns=victims)

    print("\n=== prediction (victim -> candidate, all-top3-HP rate > 0.5) ===")
    preds = {}
    for j, v in enumerate(victims):
        preds[v] = _predict(mat[v], cset)
        print(f"  {v:<24} -> {preds[v]}")

    tag = "judge" if judge else "keyword"
    png = heatmap(S, candidates, victims, preds, _out_dir() / f"demo_topic_heatmap_{tag}.png", TOPK)
    print(f"\n[demo] heatmap -> {png}")


def add_arguments(ap: argparse.ArgumentParser):
    ap.add_argument("--victims", default="", help="comma-separated victim aliases (default: the demo set)")
    ap.add_argument("--device", default=None, help="torch device (default: auto; CPU works)")
    ap.add_argument("--judge", action="store_true",
                    help="score cells by the LLM judge on generated RAG answers instead of keyword "
                         "matching (needs OPENAI_API_KEY + DEEPINFRA_API_KEY; makes API calls)")


def main(args):
    victims = [v.strip() for v in args.victims.split(",") if v.strip()] or None
    run(victims=victims, device=args.device, judge=getattr(args, "judge", False))
