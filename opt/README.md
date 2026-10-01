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
        ▼  python -m opt.score --tm {1,2} --long <long.csv> --out <dir>
{tm1,tm2}_telltail_opt_oscr_curve.csv + _qb_k.csv           # open-set metrics (TM1/TM2)
```

- **`optimize.py`** — one optimizer, a pluggable **target**: `PassageTarget` (TM1/TM2,
  suffix aimed at a victim passage) and `VectorTarget` (TM3, suffix aimed at a centroid
  vector). Semantic "global weighted blocking" forbids the top-k% of each model's vocab.
  Prefixes come from `configs/models.yaml`. Requires the private `tropt` optimizer library.
- **`evaluate.py`** — embeds `eval-prefix + query + trigger` with each eval model and
  records the target's rank in that model's **full** frozen FAISS index (`k=ntotal`,
  `nprobe=8192`). No `tropt` dependency.
- **`score.py`** — open-set scorer for both threat models via `--tm`: per-k hit matrix →
  OSCR curves (CCR vs FAR) and ASR-by-budget. The *only* difference between TMs is the hit
  rule — `--tm 2`: `hit = rank <= k`; `--tm 1`: `hit = rank <= min(3,k)` (ordered: only the
  top-3 exposed positions count, so k>=3 collapse to appeared@3). Emits the exact golden
  schemas (`tm1`=43-col/`setup1`, `tm2`=40-col/`setup2`) and reproduces both paper goldens
  bit-exactly (every column, Δ=0).
- **`config.py`** — attack hyper-parameters and per-model token-blocking budgets
  (gemma 90%, openai-3-small 25%, everything else 50%).

## Threat-model status

- **TM2-OPT (unordered top-k): implemented & validated.** `evaluate.py` reproduces the
  golden ranks (hit/miss identical at every k; diagonal exact) and `score.py --tm 2`
  reproduces the golden OSCR + ASR-by-budget (every column, Δ=0).
- **TM1-OPT (ordered): implemented & validated.** `score.py --tm 1` (hit = rank ≤ min(3,k),
  the "appeared-at" rule) reproduces the golden OSCR + qb_k bit-exactly.
- **OpenAI black-box (RASLITE+) optimize path: PENDING.** `optimize.py` currently flags
  `openai-3-small` as `[pending]`; the white-box path covers the 18 HF models.

## Keys

No keys in the repo. The OpenAI path reads credentials only from the environment / `.env`.
