# TellTail

Fingerprinting retrievers in black-box systems — reference implementation of the
generic-query attacks **TellTail-Random** and **TellTail-Topic** across two
threat models (TM1 ordered, TM2 unordered top-k).

## Setup

```bash
pip install -e .
cp .env.example .env   # then fill in the paths (see below)
```

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

### Reproduce the paper CSVs

```bash
bash scripts/reproduce_generic.sh      # both query sets, TM1 + TM2, evaluate + sweep
python plots/asr_vs_budget.py --tm 1   # Fig. 3/5-style ASR vs budget
python plots/oscr_curve.py --tm 2 --probe topic
```

Exact paper configs: budgets `1,5,10,12,15,18,20`, seed `1337`, corpus
`same_as_top_k`, enumerate-when-≤-repeats on; TM1 `top_ks 1,2,3,5,10,20,50`,
`n_repeats 100`; TM2 `top_ks 1,2,3,4,5,10,20,50`, `n_repeats 500`.

The 8 published CSVs are in `results/paper/`. `REGRESSION.md` documents how the
clean code reproduces them from the frozen score caches.

## Layout

- `generic_queries/` — the pipeline (retrieve/fetch/score/evaluate/sweep) + `_common.py` (numerics).
- `telltail/` — core library (model registry, env paths, embedding).
- `configs/models.yaml` — the 53-model registry (19 candidates); byte-exact prefixes.
- `data/queries/` — `msmarco_topic.csv`, `msmarco_random.csv`.
- `plots/` — `asr_vs_budget.py` (`--tm {1,2}`), `oscr_curve.py`.
- `scripts/reproduce_generic.sh` — end-to-end reproduction.
- `results/paper/` — the 8 golden CSVs.
- `tools/` — `build_models_yaml.py`, `run_regression.py`.

## Notes

- Doc-id and index building are not shipped (indices are ~8.8M passages each);
  `retrieve` rebuilds signatures from local FAISS indices via `$TELLTAIL_INDEX_DIR`.
- The OpenAI key is read only from the environment — there is no API-key CLI flag.
