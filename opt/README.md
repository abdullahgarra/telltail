# opt/ — TellTail-OPT (model-specific optimized queries)

TellTail-OPT crafts per-model **optimized trigger suffixes** (via the GASLITE / RASLITE+
discrete-text optimizer) so that a carrier query, once suffixed, retrieves a chosen target
under the victim retriever — a stronger fingerprint than the Random / Topic query variants.

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
        ▼  python -m opt.score --long <long.csv> --out <dir>
setup2_oscr_curves.csv + setup2_asr_by_budget_summary.csv   # TM2 open-set metrics
```

- **`optimize.py`** — one optimizer, a pluggable **target**: `PassageTarget` (TM1/TM2,
  suffix aimed at a victim passage) and `VectorTarget` (TM3, suffix aimed at a centroid
  vector). Semantic "global weighted blocking" forbids the top-k% of each model's vocab.
  Prefixes come from `configs/models.yaml`. Requires the private `tropt` optimizer library.
- **`evaluate.py`** — embeds `eval-prefix + query + trigger` with each eval model and
  records the target's rank in that model's **full** frozen FAISS index (`k=ntotal`,
  `nprobe=8192`). No `tropt` dependency.
- **`score.py`** — TM2 (unordered top-k) open-set scorer: per-k hit matrix → OSCR curves
  (CCR vs FAR) and ASR-by-budget. Reproduces the paper's TM2 numbers exactly.
- **`config.py`** — attack hyper-parameters and per-model token-blocking budgets
  (gemma 90%, openai-3-small 25%, everything else 50%).

## Threat-model status

- **TM2-OPT (unordered top-k): implemented & validated.** `evaluate.py` reproduces the
  golden ranks (hit/miss identical at every k; diagonal exact) and `score.py` reproduces
  the golden OSCR + ASR-by-budget.
- **TM1-OPT (ordered / full-rank): PENDING.** Its scorer (the "appeared-at" budget rule)
  and TM1 golden are not yet ported; TM1 reproduction is not yet verified.
- **OpenAI black-box (RASLITE+) optimize path: PENDING.** `optimize.py` currently flags
  `openai-3-small` as `[pending]`; the white-box path covers the 18 HF models.

## Keys

No keys in the repo. The OpenAI path reads credentials only from the environment / `.env`.
