"""TellTail-OPT — Stage 2 evaluator (TM1/TM2: rank of the target passage).

Ports the research `2_evaluate/evaluate_single_chunk.py`. For each eval model and each
attacked row, builds the eval query (the eval model's own prefix + the attacked
query+suffix), embeds it, searches the model's FULL frozen index (k=ntotal, nprobe=8192),
and records the rank of the target passage id.

Eval-string rule (byte-faithful to the research eval):
  * OpenAI eval model:  f"{query_text} {trigger_suffix}"                 (no prefix)
  * OSS eval model:     f"{prefix} {query_text} {trigger_suffix}"        if prefix and
                        not prefix.endswith(' '); else f"{prefix}{query_text} {trigger_suffix}"
Prefixes come from `configs/models.yaml` (== the research 2_evaluate config).

Usage:
    python -m opt.evaluate --attacks <phase1_attacks.csv> --out <long.csv> \
        --eval-models minilm-l6,e5-large [--attack-models ...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np

from generic_queries._common import normalize_docid


def build_eval_query(eval_prefix: str, query_text: str, trigger_suffix: str,
                     is_openai: bool) -> str:
    """Byte-faithful to evaluate_single_chunk.py:321-328."""
    if is_openai:
        return f"{query_text} {trigger_suffix}"
    if eval_prefix and not eval_prefix.endswith(" "):
        return f"{eval_prefix} {query_text} {trigger_suffix}"
    return f"{eval_prefix}{query_text} {trigger_suffix}"


def _sanitize(hf: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", hf)


def find_rank_oss(model, eval_query: str, target_doc_id: str, index, docids) -> Optional[int]:
    """Full-index exact rank of `target_doc_id` for the encoded `eval_query`."""
    import faiss
    q = model.encode([eval_query], convert_to_numpy=True)[0].reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    pos = np.where(docids == target_doc_id)[0]
    if len(pos) == 0:
        return None
    pos = int(pos[0])
    _, indices = index.search(q, index.ntotal)
    hit = np.where(indices[0] == pos)[0]
    return int(hit[0] + 1) if len(hit) else None


LONG_COLUMNS = ["query_id", "attack_model", "passage_id", "rank", "error", "eval_model"]


def run(attacks_csv: Path, out_csv: Path, eval_models: List[str],
        attack_models=None, index_dir_override: Optional[Path] = None) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    from telltail.models import load_registry
    from telltail.paths import index_dir

    reg = load_registry()
    idir = Path(index_dir_override) if index_dir_override else index_dir()

    rows = [r for r in csv.DictReader(open(attacks_csv, encoding="utf-8"))
            if r.get("trigger_suffix") not in (None, "", "None")]
    if attack_models:
        keep = set(attack_models)
        rows = [r for r in rows if r["attack_model"] in keep]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=LONG_COLUMNS).writeheader()

    for eval_key in eval_models:
        if eval_key not in reg:
            print(f"[skip] {eval_key}: not in registry"); continue
        is_oai = reg[eval_key].get("backend") == "openai"
        if is_oai:
            print(f"[skip] {eval_key}: OpenAI eval (no API calls in validation)"); continue
        hf = reg[eval_key]["hf_id"]
        eval_prefix = reg[eval_key].get("query_prefix", "")
        ipath = idir / f"{_sanitize(hf)}.index"
        dpath = idir / f"{_sanitize(hf)}_docids.npy"
        if not (ipath.exists() and dpath.exists()):
            print(f"[skip] {eval_key}: no frozen index at {ipath.name}"); continue

        print(f"\n=== eval model {eval_key} ({hf}) ===")
        index = faiss.read_index(str(ipath))
        if hasattr(index, "nprobe"):
            index.nprobe = 8192
        else:
            try:
                faiss.extract_index_ivf(index).nprobe = 8192
            except Exception:
                pass
        docids = np.load(str(dpath), allow_pickle=True)
        model = SentenceTransformer(hf, trust_remote_code=True)

        out = []
        for i, r in enumerate(rows):
            eval_query = build_eval_query(eval_prefix, r["query_text"], r["trigger_suffix"], is_oai)
            target = normalize_docid(r["passage_id"])
            try:
                rank = find_rank_oss(model, eval_query, target, index, docids)
                out.append(dict(query_id=r["query_id"], attack_model=r["attack_model"],
                                passage_id=r["passage_id"], rank=rank, error=None,
                                eval_model=eval_key))
                print(f"  [{i+1}/{len(rows)}] atk={r['attack_model']} q{r['query_id']} rank={rank}")
            except Exception as e:
                out.append(dict(query_id=r["query_id"], attack_model=r["attack_model"],
                                passage_id=r["passage_id"], rank=None, error=str(e),
                                eval_model=eval_key))
                print(f"  [{i+1}/{len(rows)}] ERROR: {e}")
        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=LONG_COLUMNS).writerows(out)
        del model, index, docids
        import gc; gc.collect()
    print(f"\n[done] {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT Stage-2 evaluator (TM1/TM2)")
    ap.add_argument("--attacks", type=Path, required=True, help="phase1_attacks.csv")
    ap.add_argument("--out", type=Path, required=True, help="output long CSV")
    ap.add_argument("--eval-models", required=True, help="comma-separated eval aliases")
    ap.add_argument("--attack-models", default="", help="optional subset of attack aliases")
    ap.add_argument("--index-dir", type=Path, default=None)
    args = ap.parse_args()
    run(args.attacks, args.out,
        [m.strip() for m in args.eval_models.split(",") if m.strip()],
        attack_models=[m.strip() for m in args.attack_models.split(",") if m.strip()] or None,
        index_dir_override=args.index_dir)


if __name__ == "__main__":
    main()
