"""TellTail-OPT Stage 2 (topic / response-only): retrieve the top-k passages per trigger.

For each trigger (from `opt.optimize --mode topic`) x eval model, searches the model's full
index and saves the top-k passages (id + text from the SQLite passage store). Fed to the
RAG LLM in stage 3. Default k=3; pass `--k 100` for the paper's set.

    python -m opt.retrieve --attacks <phase1_attacks.csv> --out <retrieved.csv> \
        --eval-models minilm-l6,e5-large [--k 3] [--passage-store passages.sqlite]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config as C


def build_topic_eval_query(eval_prefix: str, query_text: str, trigger_suffix: str,
                           is_openai: bool) -> str:
    # Prefix is glued directly (no extra space) — matches the research retrieve, and
    # differs from opt.evaluate's TM1/TM2 rule. OpenAI eval models get no prefix.
    if is_openai:
        return f"{query_text} {trigger_suffix}"
    return f"{eval_prefix}{query_text} {trigger_suffix}"


def _sanitize(hf: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", hf)


OUT_COLUMNS = ["attack_model", "query_group", "eval_model", "rank", "passage_id", "passage_text"]


def run(attacks_csv: Path, out_csv: Path, eval_models: List[str], k: int = C.RETRIEVE_K_DEFAULT,
        attack_models=None, index_dir_override: Optional[Path] = None,
        passage_store_path: Optional[Path] = None, allow_openai: bool = False) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    from telltail.models import load_registry
    from telltail.paths import index_dir
    from telltail.passage_store import get_passages

    reg = load_registry()
    idir = Path(index_dir_override) if index_dir_override else index_dir()

    rows = [r for r in csv.DictReader(open(attacks_csv, encoding="utf-8"))
            if r.get("trigger_suffix") not in (None, "", "None")]
    if attack_models:
        keep = set(attack_models)
        rows = [r for r in rows if r["attack_model"] in keep]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=OUT_COLUMNS).writeheader()

    for eval_key in eval_models:
        if eval_key not in reg:
            print(f"[skip] {eval_key}: not in registry"); continue
        is_oai = reg[eval_key].get("backend") == "openai"
        if is_oai and not allow_openai:
            print(f"[skip] {eval_key}: OpenAI eval (needs API; pass --allow-openai)"); continue
        if is_oai:
            print(f"[skip] {eval_key}: OpenAI retrieval path not ported yet — skipping."); continue

        hf = reg[eval_key]["hf_id"]
        eval_prefix = reg[eval_key].get("query_prefix", "")
        ipath = idir / f"{_sanitize(hf)}.index"
        dpath = idir / f"{_sanitize(hf)}_docids.npy"
        if not (ipath.exists() and dpath.exists()):
            print(f"[skip] {eval_key}: no frozen index at {ipath.name}"); continue

        print(f"\n=== eval model {eval_key} ({hf}) | k={k} ===")
        index = faiss.read_index(str(ipath))
        if hasattr(index, "nprobe"):
            index.nprobe = C.NPROBE
        else:
            try:
                faiss.extract_index_ivf(index).nprobe = C.NPROBE
            except Exception:
                pass
        docids = np.load(str(dpath), allow_pickle=True)
        model = SentenceTransformer(hf, trust_remote_code=True)

        out = []
        n_pids = n_found = 0
        for i, r in enumerate(rows):
            eval_query = build_topic_eval_query(eval_prefix, r["query_text"], r["trigger_suffix"], is_oai)
            q = model.encode([eval_query], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
            faiss.normalize_L2(q)
            _, indices = index.search(q, k)
            pids = [str(docids[int(fid)]) for fid in indices[0] if int(fid) >= 0]
            texts = get_passages(pids, db_path=passage_store_path)
            n_pids += len(pids); n_found += sum(1 for p in pids if texts.get(p))
            for rank, pid in enumerate(pids, 1):
                out.append(dict(attack_model=r["attack_model"], query_group=r["query_group"],
                                eval_model=eval_key, rank=rank, passage_id=pid,
                                passage_text=texts.get(pid, "")))
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rows)}] triggers retrieved")
        if n_pids and n_found / n_pids < 0.5:
            print(f"  [WARN] passage store had text for only {n_found}/{n_pids} retrieved ids "
                  f"({passage_store_path}). A sampled/incomplete store yields empty context and "
                  f"meaningless LLM output downstream — build the FULL store (no --sample).")
        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=OUT_COLUMNS).writerows(out)
        print(f"[saved] {eval_key}: {len(out)} rows")
        del model, index, docids
        import gc; gc.collect()

    print(f"\n[done] {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT Stage-2 retrieve (topic/response-only)")
    ap.add_argument("--attacks", type=Path, required=True, help="phase1_attacks.csv (topic mode)")
    ap.add_argument("--out", type=Path, required=True, help="output retrieved-passages CSV")
    ap.add_argument("--eval-models", required=True, help="comma-separated eval aliases")
    ap.add_argument("--k", type=int, default=C.RETRIEVE_K_DEFAULT,
                    help=f"passages to retrieve per trigger (default {C.RETRIEVE_K_DEFAULT}; "
                         f"use 100 for the paper's set)")
    ap.add_argument("--attack-models", default="", help="optional subset of attack aliases")
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--passage-store", type=Path, default=None,
                    help="SQLite passage store (default: $TELLTAIL_DATA_DIR/passages.sqlite)")
    ap.add_argument("--allow-openai", action="store_true", help="(reserved) allow OpenAI eval models")
    args = ap.parse_args()
    run(args.attacks, args.out,
        [m.strip() for m in args.eval_models.split(",") if m.strip()],
        k=args.k,
        attack_models=[m.strip() for m in args.attack_models.split(",") if m.strip()] or None,
        index_dir_override=args.index_dir, passage_store_path=args.passage_store,
        allow_openai=args.allow_openai)


if __name__ == "__main__":
    main()
