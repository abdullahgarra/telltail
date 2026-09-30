<p align="center">
  <img src="assets/telltail_icon.png" alt="TellTail" width="200"/>
</p>

<h1 align="center"><samp><b>&nbsp;T&nbsp;e&nbsp;l&nbsp;l&nbsp;T&nbsp;a&nbsp;i&nbsp;l&nbsp;</b></samp></h1>

<p align="center"><b>Fingerprinting retrievers in black-box systems</b></p>

Reference implementation of the generic-query attacks **TellTail-Random** and
**TellTail-Topic** across two threat models (TM1 ordered, TM2 unordered top-k).

## Setup

Reference environment: **Python 3.12** (results produced with the pinned versions
in `requirements.txt` — torch 2.9.1 / CUDA 12.8, faiss-cpu, numpy 2.x).

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt                     # pinned reference versions
pip install -e .                                    # the telltail package
cp .env.example .env                                # then fill in the paths (below)
```

The generic-query pipeline uses only standard PyPI packages — it does **not**
require the private OPT-attack optimizer library.

`.env` variables (all paths resolved only through these — no hardcoded paths):

| Variable | Meaning |
|---|---|
| `TELLTAIL_DATA_DIR` | corpus / query inputs |
| `TELLTAIL_INDEX_DIR` | precomputed FAISS indices (one per retriever) |
| `TELLTAIL_OUT_DIR` | pipeline outputs (retrieval / passages / scores / results) |
| `OPENAI_API_KEY` | only for the `openai-3-small` model |
| `HF_TOKEN` | only for gated HF models |

## Pipeline

One CLI, five stages: `python -m generic_queries {retrieve,fetch,score,evaluate,sweep}`.
Outputs land under `$TELLTAIL_OUT_DIR/generic/<query_set>/{retrieval,passages,scores,results}/`,
where `<query_set>` is the stem of the `--queries` CSV.

| Stage | Does | Needs |
|---|---|---|
| `retrieve` | per-model top-50 signatures over MS MARCO via each model's FAISS index | indices, GPU |
| `fetch` | union doc-ids → passage text (`global_cache.jsonl`) by streaming BeIR/msmarco | network |
| `score` | per-target minicorpus score cache `scores[candidate,target]` (shape `[Q, |C_V|]`) | GPU |
| `evaluate` | ASR by query budget × top-k → `*_qb_k` summary (TM1 ordered / TM2 set match) | score cache |
| `sweep` | open-set OSCR curves (CCR/FAR, AUOSCR) from the **same** score cache | score cache |

`--interface tm1|tm2` selects the threat model. Terminology: *target* = the
deployed victim retriever; *candidate* = a retriever in the attacker's set.

### Reproducing generic-query results

Three tiers, cheapest first. All paths come from the `.env` variables above — no
absolute paths.

**Tier 0 — figures from the published CSVs (no GPU, seconds).**
Plot straight from the 8 CSVs in `results/paper/`; nothing to recompute.
```bash
python plots/asr_vs_budget.py --tm 1          # ASR vs query budget (Fig. 3/5)
python plots/asr_vs_budget.py --tm 2
python plots/plot_ccr_0_oscr.py --tm 2 --probe topic   # OSCR + CCR@FA / AUOSCR
```

**Tier 1 — numbers from released score caches (CPU, minutes).**
Point `evaluate`/`sweep` at a released score cache (`--score-cache <dir>`) to
regenerate the qb_k and OSCR CSVs without any GPU or corpus:
```bash
python -m generic_queries evaluate --interface tm1 --queries data/queries/msmarco_topic.csv \
    --score-cache <cache> --top-ks 1,2,3,5,10,20,50 --n-repeats 100 --n-jobs 8
python -m generic_queries sweep    --interface tm2 --queries data/queries/msmarco_topic.csv \
    --score-cache <cache> --top-ks 1,2,3,4,5,10,20,50
```

**Tier 2 — full pipeline from scratch (needs FAISS indices + GPU).**
Rebuild retrieval signatures, passage cache, and score caches, then evaluate:
```bash
bash scripts/reproduce_generic.sh    # retrieve -> fetch -> score -> evaluate -> sweep
```
Requires `TELLTAIL_INDEX_DIR` populated with one FAISS index per model and a GPU.

Exact paper configs: budgets `1,5,10,12,15,18,20`, seed `1337`, corpus
`same_as_top_k`, enumerate-when-≤-repeats on; TM1 `top_ks 1,2,3,5,10,20,50`,
`n_repeats 100`; TM2 `top_ks 1,2,3,4,5,10,20,50`, `n_repeats 500`.

## Layout

- `generic_queries/` — the pipeline (retrieve/fetch/score/evaluate/sweep) + `_common.py` (numerics).
- `telltail/` — core library (model registry, env paths, embedding).
- `configs/models.yaml` — the 53-model registry (19 candidates); byte-exact prefixes.
- `data/queries/` — `msmarco_topic.csv`, `msmarco_random.csv`.
- `plots/` — `asr_vs_budget.py` (`--tm {1,2}`), `plot_ccr_0_oscr.py` (OSCR + AUOSCR), `oscr_curve.py`.
- `scripts/reproduce_generic.sh` — end-to-end reproduction.
- `results/paper/` — the 8 published CSVs.
- `tools/` — `build_models_yaml.py`.

## Notes

- Doc-id and index building are not shipped (indices are ~8.8M passages each);
  `retrieve` rebuilds signatures from local FAISS indices via `$TELLTAIL_INDEX_DIR`.
- The OpenAI key is read only from the environment — there is no API-key CLI flag.
