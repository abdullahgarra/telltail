"""`python -m demo topic` — topic-level (response-only) attack demo.

Runs each candidate's topic triggers against every victim and fingerprints a victim as the
candidate whose triggers drive its top-3 retrieved passages on-topic (Harry Potter).
Prints a prediction table (unique rate>0.5 -> that candidate, else UNK/ABSTAIN) and writes
a candidate x victim heatmap.

Fingerprint signal: keyword match (CPU, no key) by default; `--judge` instead scores each
cell with the LLM judge on the generated RAG answer (needs OPENAI_API_KEY + DEEPINFRA_API_KEY).
For the self/diagonal target it always saves the top-3 retrieved passages per query, and
prints 3 sample passages (default) / a RAG-response table (`--llm`, needs OPENAI_API_KEY) /
responses + judge verdict (`--judge`).
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
TARGET = "minilm-l6"        # self/diagonal model for the detailed passages/response view
TOPK = 3
TAU = 0.5

# Harry-Potter keyword list for the keyword path (lowercased; case-insensitive match).
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
    d = (Path(base) / "demo") if base else (Path(__file__).resolve().parents[1] / "outputs/demo")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_generate():
    """RAG generate_fn: top-3 passages -> gpt-4o-mini answer (same as opt.llm). Needs OPENAI."""
    import os
    from openai import OpenAI
    from opt import config as C
    from opt.llm import build_prompt
    from telltail.paths import data_dir as _  # ensures .env loaded  # noqa

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--llm/--judge needs OPENAI_API_KEY (RAG generation).")
    oai = OpenAI()

    def generate(user_query, texts):
        r = oai.chat.completions.create(model=C.LLM_MODEL, temperature=C.LLM_TEMPERATURE,
                                        messages=[{"role": "user", "content": build_prompt(user_query, texts)}])
        return r.choices[0].message.content or ""
    return generate


def _make_judge_fn():
    """judge_fn: RAG answer -> DeepSeek on-topic verdict (same as opt.judge). Needs DEEPINFRA."""
    import os
    from openai import OpenAI
    from opt.judge import call_judge, DEFAULT_MODEL, DEFAULT_BASE_URL

    dk = os.getenv("DEEPINFRA_API_KEY")
    if not dk:
        raise SystemExit("--judge needs DEEPINFRA_API_KEY (LLM judge).")
    judge_client = OpenAI(api_key=dk, base_url=DEFAULT_BASE_URL)

    def judge(answer):
        return call_judge(judge_client, answer, DEFAULT_MODEL)["judgement"] == "true"
    return judge


def evaluate_victims(victims, device=None, judge=False):
    """Return long results: (attack_model, query_group, eval_model, all3_hp).

    Default: all3_hp = all top-3 passages are on-topic by keyword (CPU, no key).
    --judge: all3_hp = the LLM judge labels the generated RAG answer as on-topic
    (needs OPENAI_API_KEY + DEEPINFRA_API_KEY; one generate + one judge call per cell)."""
    import faiss
    from sentence_transformers import SentenceTransformer
    from opt.retrieve import build_topic_eval_query
    from telltail.models import load_registry, _default_device

    device = _default_device(device)   # resolve auto -> cpu/cuda once (falls back off unusable GPUs)
    ids, texts = load_corpus(DATA / "corpus.jsonl")
    id2text = dict(zip(map(str, ids), texts))
    ready = list(csv.DictReader(open(DATA / "topic_ready_queries.csv")))
    reg = load_registry()
    gen_fn = judge_fn = None
    if judge:
        gen_fn, judge_fn = _make_generate(), _make_judge_fn()

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


def detail_for_target(target, device=None, mode="keyword"):
    """Self/diagonal view: retrieve `target`'s own triggers' top-3 passages, save them, and
    print 3 sample passages (keyword) / a RAG-response table (llm) / responses + verdict (judge)."""
    import faiss
    from sentence_transformers import SentenceTransformer
    from opt.retrieve import build_topic_eval_query
    from telltail.models import load_registry, _default_device

    device = _default_device(device)
    ids, texts = load_corpus(DATA / "corpus.jsonl")
    id2text = dict(zip(map(str, ids), texts))
    ready = [r for r in csv.DictReader(open(DATA / "topic_ready_queries.csv"))
             if r["attack_model"] == target]
    if not ready:
        print(f"\n[demo] no topic triggers for target '{target}'; skipping detail view.")
        return

    reg = load_registry()
    index, docids = build_index(target, ids, texts, device=device, use_cache=True)
    model = SentenceTransformer(reg[target]["hf_id"], trust_remote_code=True, device=device)
    prefix = reg[target].get("query_prefix", "")
    is_oai = reg[target].get("backend") == "openai"
    gen_fn = _make_generate() if mode in ("llm", "judge") else None
    judge_fn = _make_judge_fn() if mode == "judge" else None

    recs = []
    for r in ready:
        eq = build_topic_eval_query(prefix, r["query_text"], r["trigger_suffix"], is_oai)
        q = model.encode([eq], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        _, I = index.search(q, TOPK)
        pids = [str(docids[int(i)]) for i in I[0] if int(i) >= 0]
        rec = dict(query_group=r["query_group"], passage_ids=pids,
                   passage_texts=[id2text.get(p, "") for p in pids])
        if gen_fn is not None:
            uq = f"{r['query_text']} {r['trigger_suffix']}".strip()
            rec["response"] = gen_fn(uq, rec["passage_texts"])
            if judge_fn is not None:
                rec["judgement"] = "on-topic" if judge_fn(rec["response"]) else "off-topic"
        recs.append(rec)
    del model

    out = _out_dir()
    pcsv = out / f"topic_target_passages_{target}.csv"
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_group", "rank", "passage_id", "passage_text"])
        for rec in recs:
            for rank, (pid, txt) in enumerate(zip(rec["passage_ids"], rec["passage_texts"]), 1):
                w.writerow([rec["query_group"], rank, pid, txt])
    print(f"\n[demo] saved target '{target}' top-{TOPK} passages ({len(recs)} queries) -> {pcsv}")

    if mode == "keyword":
        print(f"\n=== target '{target}': top-{TOPK} passages for 3 sample queries ===")
        for i, rec in enumerate(recs[:3], 1):
            print(f"  query_{i}:")
            for rank, (pid, txt) in enumerate(zip(rec["passage_ids"], rec["passage_texts"]), 1):
                print(f"    {rank}. [{pid}] {' '.join(str(txt).split())[:110]}")
    else:
        print(f"\n=== target '{target}': RAG responses ({len(recs)} queries) ===")
        for i, rec in enumerate(recs, 1):
            verdict = f"   [judgement: {rec['judgement']}]" if mode == "judge" else ""
            print(f"\n--- query_{i}{verdict} ---")
            print(" ".join(str(rec.get("response", "")).split()))
        rcsv = out / f"topic_target_responses_{target}.csv"
        cols = ["query_group", "response"] + (["judgement"] if mode == "judge" else [])
        with open(rcsv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for rec in recs:
                w.writerow([rec["query_group"], rec.get("response", "")]
                           + ([rec["judgement"]] if mode == "judge" else []))
        print(f"[demo] saved responses -> {rcsv}")


def run(victims=None, device=None, judge=False, llm=False):
    mode = "judge" if judge else ("llm" if llm else "keyword")
    victims = victims or DEMO_VICTIMS
    df = evaluate_victims(victims, device=device, judge=(mode == "judge"))
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

    tag = "judge" if mode == "judge" else "keyword"       # fingerprint signal (--llm uses keyword)
    png = heatmap(S, candidates, victims, preds, _out_dir() / f"demo_topic_heatmap_{tag}.png", TOPK)
    print(f"\n[demo] heatmap -> {png}")

    # self/diagonal detailed view for the target model
    target = TARGET if TARGET in cset else (candidates[0] if candidates else None)
    if target is not None:
        detail_for_target(target, device=device, mode=mode)


def add_arguments(ap: argparse.ArgumentParser):
    ap.add_argument("--victims", default="", help="comma-separated victim aliases (default: the demo set)")
    ap.add_argument("--device", default=None, help="torch device (default: auto; CPU works)")
    ap.add_argument("--llm", action="store_true",
                    help="also print/save the target's RAG responses (gpt-4o-mini) per query — "
                         "judge them yourself (needs OPENAI_API_KEY; makes API calls)")
    ap.add_argument("--judge", action="store_true",
                    help="score cells by the LLM judge on generated RAG answers instead of keyword "
                         "matching, and add a judgement column to the target responses "
                         "(needs OPENAI_API_KEY + DEEPINFRA_API_KEY; makes API calls)")


def main(args):
    victims = [v.strip() for v in args.victims.split(",") if v.strip()] or None
    run(victims=victims, device=args.device,
        judge=getattr(args, "judge", False), llm=getattr(args, "llm", False))
