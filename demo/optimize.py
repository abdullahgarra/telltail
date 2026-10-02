"""`python -m demo optimize [--query-id N]` — craft a NEW optimized query for minilm-l6
and test it on the demo victims (GPU; needs the tropt optimizer library).

Takes one of minilm-l6's benign (query, target) pairs, runs the released white-box
optimizer (opt.optimize) to craft a fresh trigger suffix, then retrieves the target on
each demo victim index with both the NEW query and the paper's ready query. Expected: the
new query ranks the target ~1 on minilm-l6 and far away on the other victims.
"""
from __future__ import annotations

import csv

from .fingerprint import DATA, DEMO_VICTIMS
from .index import load_corpus, build_index

ATTACK = "minilm-l6"
DEFAULT_QID = 13


def main(args):
    import torch
    from sentence_transformers import SentenceTransformer
    from tropt.models.huggingface.encoder import EncoderHFModel
    from opt.optimize import (PassageTarget, forbidden_tokens_semantic, optimize_one,
                              _extract_encoder_and_tokenizer)
    from opt.evaluate import build_eval_query, find_rank_oss
    from opt import config as OPTC
    from telltail.models import load_registry
    from generic_queries._common import normalize_docid

    reg = load_registry()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    qid = args.query_id if args.query_id is not None else DEFAULT_QID

    ready = list(csv.DictReader(open(DATA / "ready_queries.csv")))
    try:
        row = next(r for r in ready if r["attack_model"] == ATTACK and int(r["query_id"]) == qid)
    except StopIteration:
        raise SystemExit(f"no {ATTACK} query with id {qid} (valid: 0..19)")
    query_text = row["query_text"]
    passage_id = normalize_docid(row["passage_id"])
    old_trigger = row["trigger_suffix"]

    ids, texts = load_corpus(DATA / "corpus.jsonl")
    id2text = dict(zip(ids, texts))
    passage_text = id2text[passage_id]

    print(f"[demo] optimizing a new query for victim '{ATTACK}' (query_id={qid}) on {device}...")
    tropt_model = EncoderHFModel(model_name=reg[ATTACK]["hf_id"])
    tokenizer, hf_encoder = _extract_encoder_and_tokenizer(tropt_model)
    hf_encoder.eval(); hf_encoder.to(device)
    forbidden, _ = forbidden_tokens_semantic(
        passage_text, tokenizer, hf_encoder, OPTC.blocking_percent(ATTACK), device)
    prefix = reg[ATTACK].get("query_prefix", "")
    new_trigger = optimize_one(model=tropt_model, query_text=query_text, query_prefix=prefix,
                               target=PassageTarget(passage_text=passage_text, passage_id=passage_id),
                               forbidden_token_ids=forbidden, device=device)
    del tropt_model, hf_encoder

    print(f"\n  benign query  : {query_text}")
    print(f"  target passage: {passage_id}")
    print(f"  NEW trigger   : {new_trigger}")
    print(f"\n  {'victim':<24}{'NEW query rank':>16}{'ready query rank':>18}")
    for v in DEMO_VICTIMS:
        index, docids = build_index(v, ids, texts, device=device, use_cache=True)
        model = SentenceTransformer(reg[v]["hf_id"], trust_remote_code=True, device=device)
        vp = reg[v].get("query_prefix", ""); is_oai = reg[v].get("backend") == "openai"
        nr = find_rank_oss(model, build_eval_query(vp, query_text, new_trigger, is_oai),
                           passage_id, index, docids)
        orr = find_rank_oss(model, build_eval_query(vp, query_text, old_trigger, is_oai),
                            passage_id, index, docids)
        tag = "  <- attacked model" if v == ATTACK else ""
        print(f"  {v:<24}{str(nr):>16}{str(orr):>18}{tag}")
        del model
    print(f"\n  full NEW query: {prefix}{query_text} {new_trigger}")
