#!/usr/bin/env python3
"""Step 1/3 — write OpenAI Batch-API request files for the MS MARCO corpus.

Streams BeIR/msmarco and emits JSONL request shards (one `/v1/embeddings` request
per passage, `custom_id = docid`), rotating files under the Batch API limits
(<=50k inputs and <=~190 MB each). No API key needed here — this only writes files.

    python indexing/openai/1_make_batches.py --out-dir <batch_dir>
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write JSONL batch-request files")
    ap.add_argument("--prefix", default="msmarco_openai_embed")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--dimensions", type=int, default=None,
                    help="Optional reduced embedding dim")
    ap.add_argument("--max-inputs-per-file", type=int, default=50_000)
    ap.add_argument("--max-file-mb", type=int, default=190)
    ap.add_argument("--truncate-chars", type=int, default=0,
                    help="0 disables; else truncate passage text to N chars")
    args = ap.parse_args()

    from datasets import load_dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = args.max_file_mb * 1024 * 1024

    print("loading BeIR/msmarco corpus...", flush=True)
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")

    file_idx, cur_count, cur_bytes = 0, 0, 0
    jf = mf = None

    def open_new(fi):
        nonlocal jf, mf, cur_count, cur_bytes
        if jf:
            jf.close()
        if mf:
            mf.close()
        jf = open(args.out_dir / f"{args.prefix}_{fi:04d}.jsonl", "w", encoding="utf-8")
        mf = open(args.out_dir / f"{args.prefix}_{fi:04d}_docids.jsonl", "w", encoding="utf-8")
        cur_count = cur_bytes = 0

    open_new(file_idx)
    for ex in corpus:
        docid = str(ex["_id"])
        text = ex.get("text", "") or ""
        if args.truncate_chars and len(text) > args.truncate_chars:
            text = text[: args.truncate_chars]
        body = {"model": args.model, "input": text}
        if args.dimensions is not None:
            body["dimensions"] = int(args.dimensions)
        line = json.dumps({"custom_id": docid, "method": "POST",
                           "url": "/v1/embeddings", "body": body},
                          ensure_ascii=False) + "\n"
        lb = len(line.encode("utf-8"))
        if (cur_count + 1 > args.max_inputs_per_file) or (cur_bytes + lb > max_bytes):
            file_idx += 1
            open_new(file_idx)
        jf.write(line)
        mf.write(json.dumps({"custom_id": docid, "docid": docid}) + "\n")
        cur_count += 1
        cur_bytes += lb
    if jf:
        jf.close()
    if mf:
        mf.close()
    print(f"done: {file_idx + 1} request file(s) in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
