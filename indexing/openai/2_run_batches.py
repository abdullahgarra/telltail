#!/usr/bin/env python3
"""Step 2/3 — submit the request files to the OpenAI Batch API and download shards.

Submits each JSONL request file from step 1 as a Batch job (keeping up to
--max-in-flight active), polls to completion, downloads the outputs, and parses
them into embedding shards `<name>.npy` + `<name>_docids.npy` (L2-normalized so
inner product = cosine) under `<out_dir>/emb_shards/`. Resumable via a state file.

The API key is read ONLY from the environment (OPENAI_API_KEY) — never hardcoded.

    OPENAI_API_KEY=... python indexing/openai/2_run_batches.py \
        --batch-dir <batch_dir> --out-dir <out_dir>
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def download_file(client, file_id, out_path: Path, max_retries=8, base_sleep=2.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            out_path.write_bytes(client.files.content(file_id).read())
            return
        except Exception as e:
            msg = str(e).lower()
            transient = any(t in msg for t in ("504", "502", "503", "timeout",
                                               "gateway time-out", "rate limit", "429"))
            if not transient or attempt == max_retries:
                raise
            s = base_sleep * (2 ** (attempt - 1)) * (0.7 + 0.6 * random.random())
            print(f"[{now()}] download retry {attempt}/{max_retries} in {s:.1f}s", flush=True)
            time.sleep(s)


def parse_output_to_shard(output_jsonl: Path, shard_npy: Path, shard_docids: Path):
    import faiss
    docids, embs, bad = [], [], 0
    for line in open(output_jsonl, "r", encoding="utf-8"):
        obj = json.loads(line)
        resp = obj.get("response")
        if obj.get("error") is not None or resp is None or resp.get("status_code") != 200:
            bad += 1
            continue
        data = resp.get("body", {}).get("data", [])
        emb = data[0].get("embedding") if data else None
        if emb is None:
            bad += 1
            continue
        docids.append(str(obj.get("custom_id")))
        embs.append(np.asarray(emb, dtype=np.float32))
    if not embs:
        raise RuntimeError(f"No embeddings parsed from {output_jsonl} (bad={bad})")
    X = np.vstack(embs).astype(np.float32, copy=False)
    faiss.normalize_L2(X)
    shard_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_npy, X)
    np.save(shard_docids, np.array(docids, dtype=object))
    return {"n_ok": len(docids), "n_bad": bad, "dim": int(X.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-in-flight", type=int, default=5)
    ap.add_argument("--poll-s", type=int, default=300)
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--state-file", default="batch_state.json")
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set in the environment.")
    from openai import OpenAI
    client = OpenAI(api_key=key)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / args.state_file
    state = load_json(state_path, {"jobs": {}})
    shard_dir = args.out_dir / "emb_shards"

    req_files = sorted(p for p in args.batch_dir.glob("*.jsonl")
                       if not p.name.endswith("_docids.jsonl"))
    if not req_files:
        raise SystemExit(f"No request files in {args.batch_dir}")

    def clean(k):
        return k[:-5] if k.endswith(".jsonl") else k

    def has_shard(k):
        ck = clean(k)
        a, b = shard_dir / f"{ck}.npy", shard_dir / f"{ck}_docids.npy"
        return a.exists() and b.exists() and a.stat().st_size > 1024

    def is_done(k):
        return state["jobs"].get(k, {}).get("status") == "done" or has_shard(k)

    pending = [p for p in req_files if not is_done(p.name)]
    print(f"{len(req_files)} request files, {len(pending)} pending", flush=True)

    def submit(p: Path) -> str:
        fobj = client.files.create(file=open(p, "rb"), purpose="batch")
        return client.batches.create(input_file_id=fobj.id, endpoint="/v1/embeddings",
                                     completion_window=args.completion_window).id

    active_states = ("submitted", "validating", "in_progress", "finalizing", "downloading")
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        active = [(k, j) for k, j in state["jobs"].items() if j.get("status") in active_states]
        while len(active) < args.max_in_flight and pending:
            p = pending.pop(0)
            try:
                bid = submit(p)
            except Exception as e:
                if any(w in str(e).lower() for w in ("enqueued", "queue", "limit")):
                    pending.insert(0, p)
                    print(f"[{now()}] queue limit; pausing submissions", flush=True)
                    break
                raise
            state["jobs"][p.name] = {"request_file": str(p), "batch_id": bid,
                                     "status": "submitted", "submitted_at": time.time()}
            save_json(state_path, state)
            active.append((p.name, state["jobs"][p.name]))
            print(f"submitted {p.name} -> {bid}", flush=True)

        n_terminal = sum(1 for j in state["jobs"].values() if j.get("remote_status") in terminal)
        if not pending and n_terminal == len(state["jobs"]) and state["jobs"]:
            print("all batches terminal.", flush=True)
            break

        for k, job in list(active):
            b = client.batches.retrieve(job["batch_id"])
            job["remote_status"] = b.status
            if b.status in ("validating", "in_progress", "finalizing"):
                job["status"] = b.status
            elif b.status == "completed":
                job["status"] = "downloading"; save_json(state_path, state)
                ck = clean(k)
                out_jsonl = args.out_dir / "batch_outputs" / f"{ck}.out.jsonl"
                download_file(client, b.output_file_id, out_jsonl)
                stats = parse_output_to_shard(out_jsonl, shard_dir / f"{ck}.npy",
                                              shard_dir / f"{ck}_docids.npy")
                try:
                    out_jsonl.unlink()
                except Exception:
                    pass
                job.update(status="done", stats=stats)
                save_json(state_path, state)
                print(f"done {k}: n_ok={stats['n_ok']:,} bad={stats['n_bad']} dim={stats['dim']}", flush=True)
            elif b.status in ("failed", "expired", "cancelled"):
                job["status"] = "failed"; save_json(state_path, state)
                print(f"FAILED {k}: {b.status}", flush=True)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
