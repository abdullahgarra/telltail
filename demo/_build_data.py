#!/usr/bin/env python3
"""DEV tool (reviewers never run this): build the two shipped demo data files.

  demo/data/corpus.jsonl      : 5,000 random MS MARCO passages (fixed seed) + all 246
                                distinct target passages from the 19x20 golden pairs (dedup)
  demo/data/ready_queries.csv : 380 golden rows (attack_model, query_id, query_text,
                                trigger_suffix, passage_id), gtr-t5 -> gtr-t5-base remap
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "_local/telltail_opt_src/tm1_tm2/2_evaluate/random_queries_attack_results/chunk_*/exp3_no_lexical/phase1_attacks.csv"
TOP50 = ROOT / "_local/telltail_opt_src/tm1_tm2/1_optimize/inputs/how_built_sloppy/nonMSMARCOqueries/top_50_per_model/*/*_top50_with_text.csv"
EXCLUDE_TOP50 = ("jina",)   # jina-v3 is eval-only, not a candidate; openai has no OPT top-50
OUT = ROOT / "demo/data"
REMAP = {"gtr-t5": "gtr-t5-base"}
CORPUS_TOTAL, SEED = 5000, 1337   # 3,548 unique top-50 (18 candidates, incl 246 targets) + random


def norm_id(x):
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- ready_queries.csv + target texts from golden phase1 ---
    ready, targets = [], {}
    for f in sorted(glob.glob(str(PHASE1))):
        for r in csv.DictReader(open(f)):
            am = REMAP.get(r["attack_model"], r["attack_model"])
            pid = norm_id(r["passage_id"])
            ready.append(dict(attack_model=am, query_id=r["query_id"],
                              query_text=r["query_text"], trigger_suffix=r["trigger_suffix"],
                              passage_id=pid))
            targets[pid] = r["passage_text"]
    ready.sort(key=lambda r: (r["attack_model"], int(r["query_id"])))
    with open(OUT / "ready_queries.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["attack_model", "query_id", "query_text",
                                           "trigger_suffix", "passage_id"])
        w.writeheader(); w.writerows(ready)
    print(f"[ready_queries] {len(ready)} rows, {len(set(r['attack_model'] for r in ready))} models, "
          f"{len(set(r['query_id'] for r in ready))} queries")
    print(f"[targets] {len(targets)} distinct target passages")

    # --- unique top-50 passages across the 18 candidate models (incl. the 246 targets) ---
    distractors = dict(targets)
    files = [f for f in sorted(glob.glob(str(TOP50)))
             if not any(x in f for x in EXCLUDE_TOP50)]
    for f in files:
        for r in csv.DictReader(open(f)):
            pid = norm_id(r["passage_id"])
            if pid not in distractors:
                distractors[pid] = r.get("passage_text", "")
    print(f"[top-50] {len(files)} candidate files -> {len(distractors)} unique passages "
          f"(incl. {len(targets)} targets)")

    # --- pad with random MS MARCO to CORPUS_TOTAL, excluding distractor ids ---
    from datasets import load_dataset
    print("[corpus] loading BeIR/msmarco...", flush=True)
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    n = len(corpus)
    rng = np.random.default_rng(SEED)
    n_random = max(0, CORPUS_TOTAL - len(distractors))
    picked = {}
    pool = rng.choice(n, size=n_random * 2 + 100, replace=False)
    for idx in pool:
        if len(picked) >= n_random:
            break
        row = corpus[int(idx)]
        pid = norm_id(row["_id"])
        if pid in distractors or pid in picked:
            continue
        picked[pid] = row["text"]
    assert len(picked) == n_random, f"got {len(picked)} random, wanted {n_random}"

    # --- corpus.jsonl = top-50-unique (∪ targets) + random ---
    rows = []
    seen = set()
    for pid, text in list(distractors.items()) + list(picked.items()):
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({"id": pid, "text": text})
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[corpus.jsonl] {len(rows)} passages = {len(distractors)} top-50-unique "
          f"(incl. {len(targets)} targets) + {len(picked)} random")
    for p in ("corpus.jsonl", "ready_queries.csv"):
        mb = (OUT / p).stat().st_size / 1e6
        print(f"   {p}: {mb:.2f} MB")


if __name__ == "__main__":
    main()
