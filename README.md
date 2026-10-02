<p align="center">
  <img src="assets/telltail_icon.png" alt="TellTail" width="200"/>
</p>

<h1 align="center"><samp><b>&nbsp;T&nbsp;e&nbsp;l&nbsp;l&nbsp;T&nbsp;a&nbsp;i&nbsp;l&nbsp;</b></samp></h1>

<p align="center"><b>🪶 Fingerprinting retrievers in black-box systems</b></p>

TellTail identifies the embedding retriever behind a black-box RAG system by probing it
with crafted queries and reading what comes back. The headline attack is **TellTail-OPT**
(model-specific optimized triggers); the quickest way to see it is the demo below. 🪶

<p align="center">
  <img src="assets/telltail-opt.gif" alt="TellTail-OPT overview" width="760"/>
</p>
<p align="center"><sub><b>TellTail-OPT.</b> Optimize a query suffix (with token-blocking) so the
suffixed query steers the victim retriever toward a chosen <b>target passage</b>
(<i>passage-level</i>, TM1/TM2) or a <b>topic centroid</b> built from synthetic passages
(<i>topic-level</i>, TM3) — the retrieved set then fingerprints the retriever.</sub></p>

| Threat model | What the attacker observes | Attack |
|---|---|---|
| **TM1** (ordered) | the ranked top-k passages | passage-level |
| **TM2** (unordered) | the top-k passage set | passage-level |
| **TM3** (response-only) | only the generated RAG answer | topic-level |

## 🪶 Quick demo

The repo ships a small corpus of **~5.6k MS MARCO
passages** (`demo/data/`) — used to run all the optimized queries used in the paper against four "victim"
models — and builds a small index per victim on the fly (cached after the first run and outputs default to `outputs/demo/`).

From a fresh clone (Python 3.12):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m demo fingerprint                 # CPU · no .env · no keys
```

The first run downloads the four victim models (public, no HF token) and builds their small
indices; later runs should be faster.
Running `optimize` needs a GPU. 
For paper-scale reproduction:
[Full reproduction](#full-reproduction).

### 1. Fingerprint victims in TM2

<sub>`python -m demo fingerprint` · CPU · no .env · no keys</sub>

Queries four victim retrievers with the paper's ready optimized queries and names each one.
By default this is **TM2** (unordered top-k) — the target only has to land in the victim's
top-k *set*:

```
victim                  k=1         k=3         k=5         k=10
minilm-l6               minilm-l6   minilm-l6   minilm-l6   minilm-l6
minilm-l12              minilm-l12  minilm-l12  minilm-l12  minilm-l12
e5-small                e5-small    e5-small    e5-small    e5-small
multilingual-e5-small   UNK         UNK         UNK         UNK
```

Three victims are in the candidate set and are identified; `multilingual-e5-small` is an
**unknown** (non-candidate) multilingual sibling of `e5-small` and correctly returns **UNK**.
The command also writes a top-3-rate heatmap to `outputs/demo/`.

For **TM1** (ordered top-k), add `--tm 1` — then only the top-3 ordered positions count.

### 2. Fingerprint victims under TM3

<sub>`python -m demo topic` · CPU</sub>

Fingerprinting under the response-only threat model. We use each candidate's optimized **topic**
triggers - used in the paper - toward Harry Potter. The command
runs them against each of the demo's victims and checks whether the victim's **top-3 retrieved passages are
all on-topic**.

In the paper, the top-3 passages are passed to an LLM (gpt-4o-mini) whose answer is scored by
an LLM judge. For a key-free demo, on-topic is decided by **keyword matching** for simplicity;
the top-3 retrieved passages per query are saved under `outputs/demo/`, and after plotting the
top-3 passages for 3 sample queries of the self target are printed so you can eyeball them.

```
victim                  prediction
minilm-l6               minilm-l6
minilm-l12              minilm-l12
e5-small                e5-small
multilingual-e5-small   UNK
```

Two opt-in layers add the LLM, shown for the self/diagonal **target** (its ~10 trigger queries):
- **`--llm`** prints (and saves) the gpt-4o-mini RAG response per query — *judge it yourself* 🕹️
  (needs `OPENAI_API_KEY`).
- **`--judge`** adds a **judgement** column (the LLM judge's verdict), and switches the
  fingerprint signal from keyword to the judge (needs `OPENAI_API_KEY` + `DEEPINFRA_API_KEY`).

### 3. Optimize a NEW query

<sub>`python -m demo optimize` · GPU</sub>

This is the **passage-level** attack (TM1/TM2): it optimizes a trigger toward a single
**target passage** (not a topic centroid).

**optimizing a trigger needs a CUDA GPU** 🍪. If
you'd like to craft your own suffix, run it on a GPU:

```bash
python -m demo optimize    # CUDA GPU required — torch 2.9.1 / CUDA 12.8, compute capability >= 7.0
```

It crafts a fresh trigger for `minilm-l6` and shows it fingerprinting only that model:

e.g., 

```
victim                  NEW query rank
minilm-l6                            1   <- attacked model
minilm-l12                         143
e5-small                          2528
multilingual-e5-small             2740
```

100 GASLITE steps with 50% semantic token-blocking: the new suffix **ranks the target
passage high on minilm-l6** and far down on the others. Flags: `--query-id N` starts from a
different benign query; `--device` picks the torch device.


## Setup

The demo needs only the install above. The **full pipeline** additionally needs `.env`
(paths to the corpus / indices / outputs). Reference environment: **Python 3.12**, pinned
versions in `requirements.txt` (torch 2.9.1 / CUDA 12.8, faiss-cpu, numpy 2.x).

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt && pip install -e .
cp .env.example .env                                # full pipeline only — fill in paths (below)
```

A single `pip install -r requirements.txt` sets up everything, including the OPT optimizer
**TROPT**. Use the provided copy — do **not** `pip install` TROPT from upstream (its current
API differs); see `third_party/tropt/`.

`.env` variables (all paths resolved only through these — no hardcoded paths):

| Variable | Meaning |
|---|---|
| `TELLTAIL_DATA_DIR` | corpus / query inputs |
| `TELLTAIL_INDEX_DIR` | precomputed FAISS indices (one per retriever) |
| `TELLTAIL_OUT_DIR` | pipeline + demo outputs |
| `OPENAI_API_KEY` | only for the `openai-3-small` model |
| `HF_TOKEN` | only for gated HF models |

## 🪶 TellTail-OPT

`opt/` crafts per-model **optimized trigger suffixes** so a benign query, once suffixed,
retrieves a chosen target under the victim retriever — a stronger fingerprint than the
Topic/Random query variants. See `opt/README.md` for the pipeline, CLI, and threat-model
status; how to run each reproduction is in [Full reproduction](#full-reproduction).

The optimizer is powered by **TROPT**, vendored under `third_party/` (see `third_party/tropt/`).

## Full reproduction

At paper scale, TellTail-OPT fingerprints the deployed retriever across the full 53-model
zoo — the diagonal (correct self-identification) dominates:

<p align="center"><img src="assets/heatmap_top3_rate.png" alt="TellTail-OPT top-3 rate (19 candidates x 53 deployed models)" width="860"/></p>
<p align="center"><sub>Top-3 rate <code>S[candidate, deployed model]</code> over the full MS MARCO index
(19 candidates × 53 deployed retrievers); outlined diagonal = correct identification.</sub></p>

All paths come from `.env`. Pick by how much you want to recompute.

### A. From scratch — build the indices, then run a variant (GPU)

```bash
python indexing/build_faiss.py --all        # one IVF+SQ8 index per model over ~8.8M MS MARCO passages
```
Build only a subset with `--models`, e.g. `python indexing/build_faiss.py --models minilm-l6,e5-small`
— the aliases are the `alias:` keys in `configs/models.yaml`.

> `retrieve`, `score`, and `opt.optimize` default to the **whole 53-model registry**. Scope to
> one model with `--models <alias>` (`retrieve`/`opt.optimize`) or `--candidates`/`--targets
> <alias>` (`score`).

**Topic / Random** — one script runs the whole procedure (both query sets, both threat models,
paper configs baked in):
```bash
bash scripts/reproduce_generic.sh                   # retrieve → fetch → score → evaluate → sweep
SKIP_UPSTREAM=1 bash scripts/reproduce_generic.sh   # reuse the score cache (skip retrieve/fetch/score)
```

The **`score`** stage writes a reusable **score cache** (the per-target proxy-corpus
similarities) to `$TELLTAIL_OUT_DIR/generic/<query_set>/scores/`. `evaluate`/`sweep` read only
that cache — so once it exists, re-running the metrics is cheap, which is exactly what
`SKIP_UPSTREAM=1` reuses.

Under the hood, the five stages (per query set — `msmarco_topic.csv`, `msmarco_random.csv`):
```bash
python -m generic_queries retrieve --queries generic_queries/queries/msmarco_topic.csv --top-k 50
python -m generic_queries fetch    --queries generic_queries/queries/msmarco_topic.csv --max-k 50   # streams the full BeIR/msmarco corpus — needs live network
python -m generic_queries score    --queries generic_queries/queries/msmarco_topic.csv --corpus-top-k 50
python -m generic_queries evaluate --interface tm1 --queries generic_queries/queries/msmarco_topic.csv --top-ks 1,2,3,5,10,20,50
python -m generic_queries sweep    --interface tm1 --queries generic_queries/queries/msmarco_topic.csv --top-ks 1,2,3,5,10,20,50
# ...then again with --interface tm2 (TM2 uses top-ks 1,2,3,4,5,10,20,50, n_repeats 500)
```

**OPT, TM1/TM2** — one script runs `optimize → evaluate → score` for both threat models:
```bash
bash scripts/reproduce_opt.sh                   # optimize → evaluate → score (TM1 + TM2)
SKIP_OPTIMIZE=1 bash scripts/reproduce_opt.sh   # score straight from the shipped ranks (no GPU, no indices)
```

Under the hood, the three stages:
```bash
python -m opt.optimize --queries opt/inputs/queries.csv --out outputs/phase1
python -m opt.evaluate --attacks outputs/phase1/phase1_attacks.csv --out outputs/opt_long.csv --eval-models <aliases>
python -m opt.score    --tm 1 --long outputs/opt_long.csv --out outputs/opt_tm1    # and --tm 2
```

**OPT, TM3 (response-only)** — opt with `RUN_TM3=1`:
```bash
RUN_TM3=1 bash scripts/reproduce_opt.sh         # optimize --mode topic → retrieve → llm → judge → heatmap
```

Needs the **full** passage store from `tools/build_passage_store.py` (no `--sample`: a sampled
store won't contain the retrieved docids, so `retrieve` returns empty passage text and `llm`
produces meaningless output), plus OpenAI (RAG) + DeepInfra (judge) keys. See
`opt/README.md`. (The bare `python plots/judge_heatmap.py` in Section C plots the **shipped**
verdicts in `results/opt/tm3_judgments/`. The `RUN_TM3=1` script instead points the heatmap at
**your own** run's verdicts — the judgements your `judge` step just wrote to
`outputs/opt/tm3_judge/` — via `plots/judge_heatmap.py --judgments outputs/opt/tm3_judge`.)

### B. From the shipped OPT ranks (+ your own generic score cache) (CPU, no indices)

Recompute just the **final metrics** — no GPU, no 8.8M-passage indices — from cached
intermediate results: the OPT ranks we ship, and (for the generic chain) the score cache
Section A's `score` wrote.

OPT TM1/TM2 scores come straight from the shipped ranks. For the generic (Topic/Random)
chain **no score cache is shipped** — `<cache>` is the `scores/` dir that Section A's `score`
writes (`$TELLTAIL_OUT_DIR/generic/<query_set>/scores`), which is also the default if you omit
the flag.

```bash
python -m opt.score --tm 1 --long results/opt/tm1_tm2_ranks/ranks.csv --out outputs/opt_tm1   # and --tm 2
python -m generic_queries evaluate --interface tm1 --queries generic_queries/queries/msmarco_topic.csv \
    --score-cache <cache> --top-ks 1,2,3,5,10,20,50 --n-repeats 100 --n-jobs 8
python -m generic_queries sweep    --interface tm2 --queries generic_queries/queries/msmarco_topic.csv \
    --score-cache <cache> --top-ks 1,2,3,4,5,10,20,50
```

### C. Figures only (from the published CSVs, CPU)

The plot scripts default to the paper's published CSVs in `results/paper/` — our exact reported
configs — so they reproduce the paper figures with no arguments beyond `--tm`:
```bash
python plots/asr_vs_budget.py   --tm 1     # ASR vs budget (OPT / Topic / Random)
python plots/plot_ccr_0_oscr.py --tm 2     # CCR@fa<=0 / CCR@fa<=2 / AUOSCR vs k
python plots/judge_heatmap.py              # TM3 response-only heatmap (from the shipped verdicts)
```

To double-check us, the raw results behind these figures are shipped too: the optimized queries
(`results/opt/optimized_queries/`), the OPT ranks (`results/opt/tm1_tm2_ranks/ranks.csv`), and the
TM3 RAG responses + judge verdicts (`results/opt/tm3_responses/`, `results/opt/tm3_judgments/`).

Exact paper configs: budgets `1,5,10,12,15,18,20`, seed `1337`, corpus `same_as_top_k`,
enumerate-when-≤-repeats on; TM1 `top_ks 1,2,3,5,10,20,50` `n_repeats 100`; TM2
`top_ks 1,2,3,4,5,10,20,50` `n_repeats 500`.

## Layout

```
telltail/
├── demo/                 # reviewer demo: fingerprint · topic · optimize
│   └── data/             #   small ~5.6k-passage corpus + ready optimized queries
├── opt/                  # TellTail-OPT: optimize → evaluate → score (+ retrieve/llm/judge for TM3)
│   └── inputs/           #   attack queries, topic passages, HP block words
├── generic_queries/      # Topic/Random pipeline: retrieve / fetch / score / evaluate / sweep
│   └── queries/          #   msmarco_topic.csv, msmarco_random.csv
├── telltail/             # core library: model registry, env paths, embedding, passage store
├── indexing/             # FAISS index builder (build_faiss.py)
│   └── openai/           #   OpenAI index via the Batch API
├── configs/              # models.yaml — 53-model registry (19 candidates), byte-exact prefixes
├── plots/                # figures: asr_vs_budget · plot_ccr_0_oscr · judge_heatmap
├── scripts/              # reproduce_generic.sh, reproduce_opt.sh
├── results/              # paper/ (published CSVs) · opt/ (ranks, verdicts, optimized queries)
├── tools/                # build_passage_store, download/package release, build_models_yaml
├── tests/                # pytest: registry + passage store
├── third_party/tropt/    # vendored GASLITE optimizer
├── assets/               # README figures
└── requirements.txt · pyproject.toml · .env.example · LICENSE · README.md
```
