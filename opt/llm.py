"""TellTail-OPT — Stage 3 for the topic-level / response-only attack: RAG LLM responses.

Reads the retrieved passages from stage 2 (`opt.retrieve`), takes the top-K per
(attack_model, query_group, eval_model), drops them into the Open WebUI RAG template, and
calls the generator LLM (gpt-4o-mini, temperature 0.8). Responses are saved as one JSONL
per eval model — this is the attacker's observable in the response-only threat model.

No model loading, no FAISS, no GPU: only the OpenAI API. The key is read from the repo
`.env` (`OPENAI_API_KEY`), loaded by `telltail.paths`.

What the research code did and we deliberately dropped: the commented-out INSPECTION_PROMPT
("tell me the topic / 'All context is Harry Potter related'") and all keyword matching —
the user runs their own judge over these responses, separately.

Usage:
    python -m opt.llm --retrieved <retrieved.csv> --attacks <phase1_attacks.csv> \
        --out <responses_dir> [--topk 3]
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import time
from pathlib import Path
from typing import Dict, List

from . import config as C

print = functools.partial(print, flush=True)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def build_prompt(user_query: str, texts: List[str]) -> str:
    """Open WebUI RAG template: passages as <source id="i"> blocks, query appended."""
    ctx = "\n\n".join(f'<source id="{i}">\n{t}\n</source>' for i, t in enumerate(texts, 1))
    base = C.RAG_TEMPLATE.replace("{{CONTEXT}}", ctx)
    return f"{base}\n\n### User Query:\n{user_query}"


def _load_phase1_lookups(attacks_csv: Path):
    """query_text + trigger_suffix keyed by 'attack_model::query_group' (from stage 1)."""
    import csv
    qtext: Dict[str, str] = {}
    trig: Dict[str, str] = {}
    if not attacks_csv or not Path(attacks_csv).exists():
        return qtext, trig
    for r in csv.DictReader(open(attacks_csv, encoding="utf-8")):
        key = f"{r.get('attack_model','')}::{r.get('query_group','')}"
        if r.get("query_text"):
            qtext[key] = r["query_text"]
        if r.get("trigger_suffix") not in (None, "", "None"):
            trig[key] = str(r["trigger_suffix"]).strip()
    return qtext, trig


def run(retrieved_csv: Path, out_dir: Path, attacks_csv: Path = None,
        topk: int = C.LLM_TOPK_DEFAULT) -> None:
    import pandas as pd
    from openai import OpenAI

    from telltail.paths import out_dir as _  # ensures .env is loaded (OPENAI_API_KEY)  # noqa

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    df = pd.read_csv(retrieved_csv)
    qtext_lookup, trig_lookup = _load_phase1_lookups(attacks_csv)

    group_cols = ["attack_model", "query_group", "eval_model"]
    groups = df.groupby(group_cols)
    print(f"[info] {len(groups)} (attack, group, eval) triples | topk={topk} "
          f"| model={C.LLM_MODEL} temp={C.LLM_TEMPERATURE}")

    processed = 0
    for (attack_model, query_group, eval_model), g in groups:
        top = g.sort_values("rank").head(topk)
        passages = [str(t) for t in top["passage_text"].tolist()]
        doc_ids = [str(i) for i in top["passage_id"].tolist()]
        if not passages:
            continue

        key = f"{attack_model}::{query_group}"
        query_text = qtext_lookup.get(key, "")
        trigger_suffix = trig_lookup.get(key, "")
        user_query = f"{query_text} {trigger_suffix}".strip()

        t0 = time.time()
        out_file = out_dir / f"{_sanitize(eval_model)}.jsonl"
        try:
            prompt = build_prompt(user_query, passages)
            responses = []
            for _call in range(C.LLM_NUM_CALLS):
                resp = client.chat.completions.create(
                    model=C.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=C.LLM_TEMPERATURE,
                )
                responses.append(resp.choices[0].message.content or "")
            record = {
                "attack_model": str(attack_model), "query_group": int(query_group),
                "eval_model": str(eval_model), "query_text": query_text,
                "trigger_suffix": trigger_suffix, "user_query": user_query,
                "topk": topk, "num_llm_calls": C.LLM_NUM_CALLS,
                "temperature": C.LLM_TEMPERATURE, "llm_model": C.LLM_MODEL,
                "doc_ids": doc_ids,
                "retrieved_passages": [{"rank": i + 1, "doc_id": d, "text": t}
                                       for i, (d, t) in enumerate(zip(doc_ids, passages))],
                "llm_responses": responses,
                "latency_seconds": round(time.time() - t0, 4),
            }
        except Exception as e:
            record = {
                "attack_model": str(attack_model), "query_group": int(query_group),
                "eval_model": str(eval_model), "query_text": query_text,
                "trigger_suffix": trigger_suffix, "user_query": user_query,
                "topk": topk, "temperature": C.LLM_TEMPERATURE, "llm_model": C.LLM_MODEL,
                "error": str(e),
            }
            print(f"[error] atk={attack_model} g={query_group} eval={eval_model}: {e}")

        with open(out_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        processed += 1
        if processed % 50 == 0:
            print(f"[progress] {processed}/{len(groups)}")

    print(f"\n[done] {processed} responses -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT Stage-3 RAG LLM (response-only)")
    ap.add_argument("--retrieved", type=Path, required=True, help="retrieved-passages CSV from opt.retrieve")
    ap.add_argument("--attacks", type=Path, default=None,
                    help="phase1_attacks.csv (for query_text/trigger lookup)")
    ap.add_argument("--out", type=Path, required=True, help="output dir for <eval_model>.jsonl responses")
    ap.add_argument("--topk", type=int, default=C.LLM_TOPK_DEFAULT,
                    help=f"passages fed to the LLM (default {C.LLM_TOPK_DEFAULT})")
    args = ap.parse_args()
    run(args.retrieved, args.out, attacks_csv=args.attacks, topk=args.topk)


if __name__ == "__main__":
    main()
