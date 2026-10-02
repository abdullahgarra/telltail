"""TellTail-OPT Stage 1: craft per-model trigger suffixes toward a target.

`--mode passage` (TM1/TM2) targets a victim passage; `--mode topic` (TM3) targets a topic
centroid. One optimizer, a pluggable `Target` (see `PassageTarget` / `VectorTarget`).

    python -m opt.optimize --queries opt/inputs/queries.csv --out outputs/phase1
    python -m opt.optimize --mode topic --out outputs/topic_phase1
"""
from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# Target abstraction (the only thing TM1/TM2 vs TM3 changes on the optimize side)
# --------------------------------------------------------------------------- #
@dataclass
class PassageTarget:
    """TM1/TM2 target: a victim passage. Blocking uses this passage's word embeddings."""
    passage_text: str
    passage_id: str = ""

    # Research fidelity: target text is the BARE passage (no passage prefix) — the
    # original read a non-existent `document_prefix` key, so no prefix was applied.
    def target_text(self) -> str:
        return self.passage_text

    def target_vector(self):
        return None

    def blocking_source_text(self) -> Optional[str]:
        return self.passage_text


@dataclass
class VectorTarget:
    """Topic-level target: a centroid vector (mean of a topic's passage embeddings). Used
    by `run_topic`; blocking uses a separate source (the longest passages), not a single
    passage, so `blocking_source_text` is None here."""
    vector: "np.ndarray"
    vector_id: str = ""

    def target_text(self):
        return None

    def target_vector(self):
        return self.vector

    def blocking_source_text(self) -> Optional[str]:
        return None


# --------------------------------------------------------------------------- #
# HF encoder plumbing (ported verbatim from the research utils)
# --------------------------------------------------------------------------- #
def _vocab_embedding_weight(hf_encoder):
    try:
        emb = hf_encoder.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
    except NotImplementedError:
        pass
    if hasattr(hf_encoder, "embeddings") and hasattr(hf_encoder.embeddings, "word_embeddings"):
        return hf_encoder.embeddings.word_embeddings.weight
    if (hasattr(hf_encoder, "bert") and hasattr(hf_encoder.bert, "embeddings")
            and hasattr(hf_encoder.bert.embeddings, "word_embeddings")):
        return hf_encoder.bert.embeddings.word_embeddings.weight
    raise RuntimeError(f"Couldn't find vocab embedding weight for: {type(hf_encoder)}")


def _extract_encoder_and_tokenizer(tropt_model):
    from sentence_transformers import SentenceTransformer
    st = getattr(tropt_model, "model", None)
    if st is None:
        raise RuntimeError("tropt_model has no attribute `.model`")
    if not isinstance(st, SentenceTransformer) and not hasattr(st, "_first_module"):
        raise RuntimeError(f"tropt_model.model is not ST and has no _first_module: {type(st)}")
    tokenizer = getattr(st, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Could not access tokenizer from SentenceTransformer")
    first = st._first_module()
    hf_encoder = getattr(first, "auto_model", None) or getattr(first, "model", None)
    if hf_encoder is None or not hasattr(hf_encoder, "get_input_embeddings"):
        raise RuntimeError(f"Could not extract HF encoder from first module: {type(first)}")
    return tokenizer, hf_encoder


def _word_embeddings_and_tokens(passage_text, tokenizer, hf_encoder, device):
    """Word-level contextual embeddings (mean-pooled per whitespace word), normalized."""
    import torch
    import torch.nn.functional as F
    words = passage_text.split()
    enc = tokenizer(passage_text, return_tensors="pt", add_special_tokens=True,
                    truncation=True, max_length=C.BLOCKING_MAX_SEQ_LENGTH)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = hf_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state.squeeze(0).float()
    hidden = F.normalize(hidden, p=2, dim=1)
    word_embeddings, word_token_ids_list = [], []
    token_idx = 1  # skip [CLS]
    for word in words:
        word_enc = tokenizer.encode(word, add_special_tokens=False)
        n = len(word_enc)
        if token_idx + n > len(hidden):
            break
        word_embeddings.append(hidden[token_idx:token_idx + n].mean(dim=0))
        word_token_ids_list.append(word_enc)
        token_idx += n
    if not word_embeddings:
        raise ValueError("No valid words found in passage")
    return torch.stack(word_embeddings), words[:len(word_embeddings)], word_token_ids_list


def forbidden_tokens_semantic(passage_text, tokenizer, hf_encoder, percent, device,
                              batch_size=8192) -> Tuple[List[int], int]:
    """Global weighted blocking: forbid the top-`percent`% of vocab by max cosine sim to
    the passage's word embeddings (passage tokens prioritized). Ported verbatim."""
    import torch
    import torch.nn.functional as F
    word_embeddings, words, word_token_ids_list = _word_embeddings_and_tokens(
        passage_text, tokenizer, hf_encoder, device)
    vocab_size = tokenizer.vocab_size

    passage_token_ids = set()
    for tl in word_token_ids_list:
        passage_token_ids.update(tl)

    total_budget = int(vocab_size * percent / 100.0)
    if total_budget - len(passage_token_ids) <= 0:
        return list(passage_token_ids), 0

    word_embeddings = F.normalize(word_embeddings.float(), p=2, dim=1)
    vocab_weight = _vocab_embedding_weight(hf_encoder).detach()
    max_scores = np.empty(vocab_size, dtype=np.float32)
    n_batches = (vocab_size + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, vocab_size)
        vb = F.normalize(vocab_weight[s:e].float(), p=2, dim=1)
        sims = word_embeddings @ vb.T          # [W, B]
        max_scores[s:e] = sims.max(dim=0)[0].detach().cpu().numpy().astype(np.float32, copy=False)
        del vb, sims

    forbidden = set(passage_token_ids)
    k = total_budget
    top_idx = np.argpartition(max_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(max_scores[top_idx])[::-1]]
    n_added = 0
    for tid in top_idx:
        if len(forbidden) >= total_budget:
            break
        tid = int(tid)
        if tid in forbidden:
            continue
        forbidden.add(tid); n_added += 1
    if len(forbidden) < total_budget:  # worst-case fallback (rare)
        for tid in np.argsort(max_scores)[::-1]:
            if len(forbidden) >= total_budget:
                break
            tid = int(tid)
            if tid in forbidden:
                continue
            forbidden.add(tid); n_added += 1
    return list(forbidden), n_added


# --------------------------------------------------------------------------- #
# tropt optimizer wrapper (white-box today; black-box/OpenAI is the next increment)
# --------------------------------------------------------------------------- #
def optimize_one(*, model, query_text: str, query_prefix: str, target,
                 forbidden_token_ids: List[int], device: str) -> str:
    """Return the optimized trigger suffix for one (query, target) under `model`."""
    from opt._tropt_api import craft_fingerprint_query  # noqa: local wrapper
    query_template = f"{query_prefix}{query_text} {{{{OPTIMIZED_TRIGGER}}}}"
    initial_trigger = ("! " * C.TRIGGER_LEN).strip()
    return craft_fingerprint_query(
        model=model,
        query_template=query_template,
        initial_trigger=initial_trigger,
        forbidden_tokens=forbidden_token_ids,
        target_text=target.target_text(),
        target_vector=target.target_vector(),
        device=device,
    )


def _clear():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
# Schema identical to the research phase1_attacks.csv. `num_tokens_blocked_lexical` is
# always 0 (the golden used the no-lexical variant); kept for schema compatibility.
PHASE1_COLUMNS = ["query_id", "passage_id", "query_text", "passage_text", "attack_model",
                  "trigger_suffix", "full_triggered_query",
                  "num_tokens_blocked_semantic", "num_tokens_blocked_lexical", "error"]


def run(queries_csv: Path, out_dir: Path, only_models=None) -> None:
    import torch
    from telltail.models import load_registry

    random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = load_registry()

    rows = list(csv.DictReader(open(queries_csv, encoding="utf-8")))
    models = sorted({r["attack_model"] for r in rows})
    if only_models:
        models = [m for m in models if m in set(only_models)]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_attacks.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=PHASE1_COLUMNS).writeheader()

    from tropt.models.huggingface.encoder import EncoderHFModel

    for alias in models:
        if alias not in reg:
            print(f"[skip] {alias}: not in registry"); continue
        if reg[alias].get("backend") == "openai":
            print(f"[openai] {alias}: token-blocking here differs from the config used for "
                  f"the paper's released queries — skipping. The shipped ready queries are "
                  f"the paper's.", file=sys.stderr)
            continue

        hf_id = reg[alias]["hf_id"]
        query_prefix = reg[alias].get("query_prefix", "")
        percent = C.blocking_percent(alias)
        print(f"\n=== {alias} ({hf_id}) | blocking={percent}% ===")
        try:
            tropt_model = EncoderHFModel(model_name=hf_id)
            tokenizer, hf_encoder = _extract_encoder_and_tokenizer(tropt_model)
            hf_encoder.eval(); hf_encoder.to(device)
        except Exception as e:
            print(f"[fail-load] {alias}: {e}"); _clear(); continue

        model_rows = [r for r in rows if r["attack_model"] == alias]
        out_records = []
        for i, r in enumerate(model_rows):
            qid, pid = str(r["query_id"]), str(r["passage_id"])
            qtext, ptext = r["query_text"], r["passage_text"]
            rec = dict(query_id=qid, passage_id=pid, query_text=qtext, passage_text=ptext,
                       attack_model=alias, trigger_suffix=None, full_triggered_query=None,
                       num_tokens_blocked_semantic=0, num_tokens_blocked_lexical=0, error=None)
            try:
                target = PassageTarget(passage_text=ptext, passage_id=pid)
                forbidden, n_sem = forbidden_tokens_semantic(
                    target.blocking_source_text(), tokenizer, hf_encoder, percent, device)
                suffix = optimize_one(model=tropt_model, query_text=qtext,
                                      query_prefix=query_prefix, target=target,
                                      forbidden_token_ids=forbidden, device=device)
                rec.update(trigger_suffix=suffix,
                           full_triggered_query=f"{query_prefix}{qtext} {suffix}",
                           num_tokens_blocked_semantic=n_sem)
                print(f"  [{i+1}/{len(model_rows)}] q{qid} ok")
            except Exception as e:
                import traceback; traceback.print_exc()
                rec["error"] = str(e)
                print(f"  [{i+1}/{len(model_rows)}] q{qid} ERROR: {e}")
            out_records.append(rec)

        with open(out_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=PHASE1_COLUMNS).writerows(out_records)
        print(f"[saved] {alias}: {len(out_records)} rows -> {out_path}")
        del tropt_model, tokenizer, hf_encoder
        _clear()

    print(f"\n[done] {out_path}")


# --------------------------------------------------------------------------- #
# Topic-level attack (response-only evaluation): centroid target + 3-stage blocking
# --------------------------------------------------------------------------- #
# Same optimizer, same semantic blocking, same prefixes as the passage-level attack
# above — only the target (a topic centroid instead of one passage) and a third
# (basic-BPE lexical) blocking stage differ.
def _load_block_words(path):
    """One word per line; blank lines and `#` comments stripped (research-faithful)."""
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def forbidden_tokens_3stage(blocking_text, tokenizer, hf_encoder, percent, block_words, device):
    """Three-stage global weighted blocking (`get_forbidden_tokens_exp3`, basic-BPE):
      (1) passage tokens of `blocking_text`  (2) semantic fill to `percent`% of vocab
      (shared with the passage-level attack)  (3) basic-BPE lexical over `block_words`,
      forbidden ON TOP of the semantic budget. Returns (ids, n_semantic, n_lexical)."""
    forbidden, n_sem = forbidden_tokens_semantic(
        blocking_text, tokenizer, hf_encoder, percent, device)
    forbidden_set = set(forbidden)
    n_lex = 0
    if block_words:
        basic_ids = set()
        for word in block_words:
            for variant in {word, word.lower(), word.upper(), word.capitalize()}:
                basic_ids.update(tokenizer.encode(variant, add_special_tokens=False))
        n_lex = len(basic_ids - forbidden_set)
        forbidden_set.update(basic_ids)
    return list(forbidden_set), n_sem, n_lex


def _centroid_vector(tropt_model, passages):
    """Centroid = mean of the attacked model's embeddings of the (bare) group passages.
    Reproduces `run_centroid_attack`: no passage prefix (the document_prefix bug)."""
    import torch
    with torch.no_grad():
        embs = tropt_model(passages)
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        return embs.mean(dim=0, keepdim=True)


def _load_topic_input(queries_csv: Path):
    """Return (query_id, query_text, [passage_text...]) from the single-query input."""
    rows = list(csv.DictReader(open(queries_csv, encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty input: {queries_csv}")
    qids = {r["query_id"] for r in rows}
    if len(qids) != 1:
        raise ValueError(f"topic input expects ONE query_id; got {sorted(qids)}")
    return rows[0]["query_id"], rows[0]["query_text"], [r["passage_text"] for r in rows]


def _split_groups(passages):
    need = C.NUM_TOPIC_GROUPS * C.PASSAGES_PER_GROUP
    if len(passages) < need:
        raise ValueError(f"need >= {need} passages for {C.NUM_TOPIC_GROUPS} groups of "
                         f"{C.PASSAGES_PER_GROUP}; got {len(passages)}")
    return [passages[i * C.PASSAGES_PER_GROUP:(i + 1) * C.PASSAGES_PER_GROUP]
            for i in range(C.NUM_TOPIC_GROUPS)]


def _blocking_text(passages):
    """The TOP_N longest passages (of ALL passages), longest first, joined."""
    lengths = [len(t) for t in passages]
    top_idx = np.argsort(lengths)[-C.TOP_N_BLOCKING_PASSAGES:][::-1]
    return C.BLOCKING_JOINER.join(passages[i] for i in top_idx)


# Schema identical to the research topic-attack phase1_attacks.csv.
TOPIC_PHASE1_COLUMNS = ["query_id", "query_text", "query_group", "attack_model",
                        "trigger_suffix", "full_triggered_query",
                        "num_tokens_blocked_semantic", "num_tokens_blocked_lexical", "error"]


def run_topic(queries_csv: Path, out_dir: Path, only_models=None, attack_models=None) -> None:
    import random
    import torch
    from telltail.models import load_registry

    random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = load_registry()

    query_id, query_text, passages = _load_topic_input(queries_csv)
    groups = _split_groups(passages)
    blocking_text = _blocking_text(passages)
    block_words = _load_block_words(C.BLOCK_WORDS_PATH)
    print(f"[+] query_id={query_id} | {len(passages)} passages | {len(groups)} groups "
          f"| blocking_text={len(blocking_text)} chars | {len(block_words)} block words")

    models = attack_models or sorted(a for a, e in reg.items() if e.get("candidate"))
    if only_models:
        models = [m for m in models if m in set(only_models)]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_attacks.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=TOPIC_PHASE1_COLUMNS).writeheader()

    from tropt.models.huggingface.encoder import EncoderHFModel

    for alias in models:
        if alias not in reg:
            print(f"[skip] {alias}: not in registry"); continue
        if reg[alias].get("backend") == "openai":
            print(f"[openai] {alias}: token-blocking here differs from the config used for "
                  f"the paper's released queries — skipping. The shipped ready queries are "
                  f"the paper's.", file=sys.stderr)
            continue

        hf_id = reg[alias]["hf_id"]
        query_prefix = reg[alias].get("query_prefix", "")
        percent = C.blocking_percent(alias)
        print(f"\n=== {alias} ({hf_id}) | blocking={percent}% ===")
        try:
            tropt_model = EncoderHFModel(model_name=hf_id)
            tokenizer, hf_encoder = _extract_encoder_and_tokenizer(tropt_model)
            hf_encoder.eval(); hf_encoder.to(device)
        except Exception as e:
            print(f"[fail-load] {alias}: {e}"); _clear(); continue

        # Forbidden tokens computed ONCE per model (from the N longest passages).
        try:
            forbidden, n_sem, n_lex = forbidden_tokens_3stage(
                blocking_text, tokenizer, hf_encoder, percent, block_words, device)
            print(f"  blocking: semantic={n_sem} lexical={n_lex} total={len(forbidden)}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[fail-block] {alias}: {e}")
            del tropt_model, tokenizer, hf_encoder; _clear(); continue

        out_records = []
        for gi, group in enumerate(groups):
            rec = dict(query_id=query_id, query_text=query_text, query_group=gi,
                       attack_model=alias, trigger_suffix=None, full_triggered_query=None,
                       num_tokens_blocked_semantic=n_sem, num_tokens_blocked_lexical=n_lex,
                       error=None)
            try:
                centroid = _centroid_vector(tropt_model, group)
                target = VectorTarget(vector=centroid, vector_id=f"{alias}:g{gi}")
                suffix = optimize_one(model=tropt_model, query_text=query_text,
                                      query_prefix=query_prefix, target=target,
                                      forbidden_token_ids=forbidden, device=device)
                rec.update(trigger_suffix=suffix,
                           full_triggered_query=f"{query_prefix}{query_text} {suffix}")
                print(f"  [g{gi}/{len(groups)-1}] ok  trig[:60]={suffix[:60]!r}")
            except Exception as e:
                import traceback; traceback.print_exc()
                rec["error"] = str(e)
                print(f"  [g{gi}/{len(groups)-1}] ERROR: {e}")
            out_records.append(rec)

        with open(out_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=TOPIC_PHASE1_COLUMNS).writerows(out_records)
        print(f"[saved] {alias}: {len(out_records)} rows -> {out_path}")
        del tropt_model, tokenizer, hf_encoder
        _clear()

    print(f"\n[done] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="TellTail-OPT Stage-1 optimizer")
    ap.add_argument("--mode", choices=["passage", "topic"], default="passage",
                    help="passage = TM1/TM2 (one victim passage); "
                         "topic = topic-level centroid attack (response-only eval)")
    ap.add_argument("--queries", type=Path, default=None,
                    help="input CSV (default: opt/inputs/queries.csv for passage, "
                         "opt/inputs/query_passages.csv for topic)")
    ap.add_argument("--out", type=Path, required=True, help="output dir for phase1_attacks.csv")
    ap.add_argument("--models", default="", help="optional comma-separated subset of attack aliases")
    args = ap.parse_args()
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    if args.mode == "topic":
        queries = args.queries or Path("opt/inputs/query_passages.csv")
        run_topic(queries, args.out, only_models=only)
    else:
        queries = args.queries or Path("opt/inputs/queries.csv")
        run(queries, args.out, only_models=only)


if __name__ == "__main__":
    main()
