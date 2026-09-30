#!/usr/bin/env python3
"""Package the tier-1 release bundle: retrievals + passage cache + score caches.

For each query set it collects, from the frozen research artifacts, exactly what
`evaluate`/`sweep` need to run without any GPU or corpus:
  - retrieval/<sanitized_hf>/<alias>_top50_docids.npy   (target top-50 signatures)
  - passages/global_cache.jsonl                          (passage-text cache)
  - scores/target=<t>/{corpus_docids,victim_full_top50_docids,query_ids}.npy
    scores/target=<t>/candidate=<c>/scores.npy           (19 candidates + self)

Written under <out>/generic/<query_set>/ in the release layout (target=/candidate=),
with a manifest.json that contains NO absolute paths. Reports total size.

    python tools/package_release.py --research-root /path/to/emb_fingerprinting
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from telltail.models import candidates as reg_candidates            # noqa: E402
from generic_queries._common import canonical_alias, aliases_to_try  # noqa: E402

# frozen sources, relative to --research-root
QUERY_SETS = {
    "msmarco_topic": {
        "sig": "baseline/topic_based_baseline/single_topic_baseline_offline_signatures",
        "passages": "baseline/topic_based_baseline/single_topic_model_minicorpus/global_cache.jsonl",
        "scores": "baseline/topic_based_baseline/score_cache_topic",
    },
    "msmarco_random": {
        "sig": "baseline/fixed_exploratory/baseline_offline_signatures",
        "passages": "baseline/fixed_exploratory/model_minicorpus/global_cache.jsonl",
        "scores": "baseline/fixed_exploratory/score_cache_random",
    },
}
STATIC = ["corpus_docids.npy", "victim_full_top50_docids.npy", "query_ids.npy"]


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def find_retriever(victim_dir: Path, alias: str):
    for a in aliases_to_try(alias):
        d = victim_dir / f"retriever={a}"
        if (d / "scores.npy").is_file():
            return d
    return None


def package_one(qs: str, cfg: dict, research: Path, out_root: Path):
    cands = sorted(reg_candidates())
    out = out_root / "generic" / qs
    # 1) retrieval signatures (top-50 docids per model)
    sig = research / cfg["sig"]
    n_sig = 0
    for npy in sig.glob("**/*_top50_docids.npy"):
        rel = npy.relative_to(sig)
        dst = out / "retrieval" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npy, dst)
        n_sig += 1
    # 2) passage-text cache
    (out / "passages").mkdir(parents=True, exist_ok=True)
    shutil.copy2(research / cfg["passages"], out / "passages" / "global_cache.jsonl")
    # 3) score caches: victim=/retriever= -> target=/candidate= (candidates + self)
    scores_src = research / cfg["scores"]
    targets, n_cells, missing = [], 0, []
    for vdir in sorted(scores_src.glob("victim=*")):
        t = canonical_alias(vdir.name.split("victim=", 1)[1])
        targets.append(t)
        tdir = out / "scores" / f"target={t}"
        (tdir).mkdir(parents=True, exist_ok=True)
        for s in STATIC:
            if (vdir / s).is_file():
                shutil.copy2(vdir / s, tdir / s)
        wanted = sorted(set(cands) | {t})
        for c in wanted:
            rdir = find_retriever(vdir, c)
            if rdir is None:
                missing.append((t, c))
                continue
            cdst = tdir / f"candidate={c}"
            cdst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rdir / "scores.npy", cdst / "scores.npy")
            n_cells += 1
    # clean manifest (no absolute paths)
    (out / "scores" / "manifest.json").write_text(json.dumps({
        "query_set": qs, "layout": "target=<t>/candidate=<c>/scores.npy",
        "n_targets": len(targets), "n_candidates": len(cands),
        "candidates": cands, "n_score_cells": n_cells,
        "note": "tier-1 bundle for evaluate/sweep; scores over each target's C_V(20) minicorpus",
    }, indent=2), encoding="utf-8")
    return {"query_set": qs, "n_signatures": n_sig, "n_targets": len(targets),
            "n_score_cells": n_cells, "missing": missing,
            "size_bytes": dir_size(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-root", required=True, type=Path,
                    help="Path to the emb_fingerprinting research tree (frozen sources).")
    ap.add_argument("--out", type=Path, default=ROOT / "_local" / "release")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for qs, cfg in QUERY_SETS.items():
        r = package_one(qs, cfg, args.research_root, args.out)
        total += r["size_bytes"]
        mb = r["size_bytes"] / 1e6
        print(f"[{qs}] signatures={r['n_signatures']} targets={r['n_targets']} "
              f"score_cells={r['n_score_cells']} size={mb:.1f} MB"
              + (f"  MISSING={len(r['missing'])}" if r["missing"] else ""))
    print(f"\n[bundle] {args.out}  total = {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
