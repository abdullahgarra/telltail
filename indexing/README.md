# indexing/ — build the FAISS indices

`indexing/` holds the **code that builds** the per-model FAISS indices.
`$TELLTAIL_INDEX_DIR` (locally, the gitignored `indexes/` symlink dir) holds the
**built indices themselves** — one multi-GB IVF+SQ8 index per model over all
~8.8M MS MARCO passages. Builder here; artifacts there.

## Build

```bash
# one model (e.g. to test retrieval against a single index)
python indexing/build_faiss.py --models minilm-l6

# all 53 registry models (huge; shard across GPUs/jobs with --models)
python indexing/build_faiss.py --all --device cuda
```

Each model produces `<sanitized_hf_id>.index`, `_docids.npy`, `_spec.json` in
`$TELLTAIL_INDEX_DIR` (or `--out`) — the filenames `retrieve` looks up. The index
is `IndexIVFScalarQuantizer(IndexFlatIP, dim, nlist=8192, QT_8bit,
METRIC_INNER_PRODUCT)`; passages are embedded through `telltail.models` (registry
prefixes + OpenAI/ST backend) and L2-normalized so inner product = cosine.

Indices are **not** committed (multi-GB each).

## OpenAI model (`openai-3-small`) — multi-stage Batch-API build

`build_faiss.py` **skips** the OpenAI model: embedding ~8.8M passages via
synchronous API calls is infeasible. Its index is built through the OpenAI
**Batch API** in three stages (final `.index`/`_docids.npy`/`_spec.json` land in
`$TELLTAIL_INDEX_DIR` as `openai_text-embedding-3-small.*`, matching `retrieve`):

1. **make batches** — chunk the corpus into JSONL batch-request files
   (≤50k inputs / ≤190 MB each) plus doc-id maps.
2. **run batches** — submit to the OpenAI Batch API, poll to completion, and
   download the embedding shards (`.npy` + `.docids.npy`).
3. **build from shards** — assemble the IVF + 8-bit SQ index (nlist=8192, IP)
   from the downloaded shards — same index type as the ST models.

Requires `OPENAI_API_KEY` in the environment. Reference implementation (not yet
ported into this repo): `closedSource/cache_faiss/cache_openai/`
(`make_msmarco_batches.py` → `run_openai_batches.py` → `build_faiss_from_shards.py`,
with `verifyBatches.py` / `reduceBatches.py` helpers).

## Passage prefixes — reproducing the frozen indices

The builder embeds passages with **`index_passage_prefix`** (in
`configs/models.yaml`) — the prefix each frozen index was *actually* built with —
so it reproduces the released indices rather than "fixing" them. This is separate
from `passage_prefix`, which the **scoring** stage used.

`index_passage_prefix` was determined **empirically** by an SQ8 code-match test
(`_local/tests/index_check`): for each model, sampled indexed passages are
re-embedded and re-compressed with the frozen index's own quantizer; the prefix
that reproduces the stored codes (~99%+ exact, rest off-by-1 from GPU float noise)
is the one the index was built with. It equals `passage_prefix` for **51/53**
models. Only **2 targets' indices were built with a different passage prefix than
the scoring stage used** — a documented property of the released indices, not a
correction:

| model | `index_passage_prefix` (frozen index, verified) | `passage_prefix` (scoring) |
|---|---|---|
| gemma-300m | `"title: none | text: "` (trailing space) | `"title: none | text:"` |
| jina-v5-small | `""` (empty) | `"Document:"` |

Verified code-match at the winning prefix: gemma **98.5% exact** (0% for the
no-trailing-space variant — the trailing space is *not* tokenizer-ignored),
jina-v5-small **99.7% exact** (0% for `Document:`). `build_faiss.py` uses
`index_passage_prefix`, so a rebuild matches these frozen indices.
