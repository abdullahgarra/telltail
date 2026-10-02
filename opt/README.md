# opt/ — TellTail-OPT (model-specific optimized queries)

TellTail-OPT crafts per-model **optimized trigger suffixes** (via the GASLITE discrete-text
optimizer) so that a benign query, once suffixed, retrieves a chosen target (topic or passage) under the victim retriever.

## Pipeline

```
opt/inputs/queries.csv                      # (query, target passage, attack_model) triples
        │
        ▼  python -m opt.optimize --queries opt/inputs/queries.csv --out <dir>
phase1_attacks.csv                          # + trigger_suffix per row
        │
        ▼  python -m opt.evaluate --attacks <dir>/phase1_attacks.csv --out <long.csv> \
        │        --eval-models <aliases>
<long.csv>                                  # rank of each target in each eval model's FULL index
        │
        ▼  python -m opt.score --tm {1,2} --long <long.csv> --out <dir>
{tm1,tm2}_telltail_opt_oscr_curve.csv + _qb_k.csv           # open-set metrics (TM1/TM2)
```

- **`optimize.py`** — one optimizer, a pluggable **target**, selected by `--mode`:
  `--mode passage` (default; TM1/TM2) aims the suffix at one victim passage;
  `--mode topic` aims it at the **centroid** of a group of passages (the topic-level
  attack, below). Semantic "global weighted blocking" forbids the top-k% of each model's
  vocab. Prefixes come from `configs/models.yaml`. Requires our own `tropt` library.
- **`evaluate.py`** — embeds `eval-prefix + benign_query + trigger_suffix` with each eval model and
  records the target's rank in that model's **full** frozen FAISS index (`k=ntotal`,
  `nprobe=8192`).
- **`score.py`** — open-set scorer for both threat models via `--tm`: per-k hit matrix →
  OSCR curves (CCR vs FAR) and ASR-by-budget. The *only* difference between TMs is the hit
  rule — `--tm 2`: `hit = rank <= k`; `--tm 1`: `hit = rank <= min(3,k)`.
- **`config.py`** — attack hyper-parameters and per-model token-blocking budgets
  (gemma 90%, openai-3-small 25%, everything else 50%), plus the topic-attack / RAG knobs.

## Topic-level attack (response-only — threat model 3)

Here the attacker observes only the **LLM's answer**, not the ranking. The trigger is
optimized toward the **centroid of a topic's passages**; at eval time the triggered query
retrieves passages, those are fed to a RAG LLM, and the generated response is saved.

```
opt/inputs/query_passages.csv               # one benign query + 100 topic passages
        │
        ▼  python -m opt.optimize --mode topic --out <dir>
<dir>/phase1_attacks.csv                     # 10 triggers/model (10 passage-groups → centroids)
        │
        ▼  python -m opt.retrieve --attacks <dir>/phase1_attacks.csv --out <dir>/retrieved.csv \
        │        --eval-models <aliases> [--k 3]
<dir>/retrieved.csv                          # top-k passages per (attack, group, eval model)
        │
        ▼  python -m opt.llm --retrieved <dir>/retrieved.csv --attacks <dir>/phase1_attacks.csv \
        │        --out <dir>/responses
<dir>/responses/<eval_model>.jsonl           # gpt-4o-mini responses
        │
        ▼  python -m opt.judge --responses <dir>/responses --out <dir>/judge     # DeepInfra key
<dir>/judge/judgements.jsonl                 # per-response verdict (on-topic? true/false)
        │
        ▼  python plots/judge_heatmap.py --judgments <dir>/judge --out <dir>
heatmap_llm_judge_hp_rate.png                # candidate x eval, judge-verdict rate
```

- **optimize `--mode topic`** adds a third blocking stage (basic-BPE lexical over
  `opt/inputs/hp_block_words.txt`) on top of the shared passage-token + semantic stages.
- **`retrieve.py`** retrieves the top-k passages (default **k=3**, since full top-100
  retrieval is slow — pass **`--k 100`** for the paper's set; the LLM uses the top-3 either
  way) and attaches passage text from the on-disk store (below).
- **`llm.py`** fills the Open WebUI RAG template, calls **gpt-4o-mini, temperature 0.8**,
  and writes one JSONL per eval model.
- **`judge.py`** asks a judge LLM (DeepSeek-V4-Flash via DeepInfra; `DEEPINFRA_API_KEY`)
  whether each response says the retrieved context is *exclusively* on the hidden topic —
  the response-only signal. Writes a slim `judgements.jsonl` (+ full records).
  `plots/judge_heatmap.py` turns the verdicts into the candidate×eval heatmap.

### Passage store (full TM3 reproduction only — not the demo, not TM1/TM2)

`opt.retrieve` needs `passage_id → text`. TM1/TM2 score **ranks**, not text; and the demo reads
text from its small shipped corpus — so only **full-scale TM3 reproduction** needs this store.
Build it once onto your own disk (CPU + disk only, ~3 GB, nothing committed):

```
python -m tools.build_passage_store --out $TELLTAIL_DATA_DIR/passages.sqlite
# smoke test on a sample + verify ids line up with an index:
python -m tools.build_passage_store --out /tmp/passages_sample.sqlite --sample 2000 \
    --smoke --check-index minilm-l6
```

It iterates the corpus through the **same loader** as `indexing/build_faiss.py`
(`telltail.corpus`), so store ids and index docids share one id space. If it's missing,
`opt.retrieve` fails with the exact build command above.