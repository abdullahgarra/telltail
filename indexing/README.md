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
from `passage_prefix`, which the **scoring** stage used (and which the regression
validated).

For 47/53 models the two agree. For **5 models the released index was built with a
different (usually empty) passage prefix than the scoring stage used** — a known
property of the released indices, not something the builder corrects:

| model | `index_passage_prefix` (frozen index) | `passage_prefix` (scoring) |
|---|---|---|
| e5-base | `""` | `"passage: "` |
| nomic-v1.5 | `""` | `"search_document: "` |
| nomic-v1 | `""` | `"search_document: "` |
| jina-v5-small | `""` | `"Document:"` |
| gemma-300m | `"title: none | text: "` (trailing space) | `"title: none | text:"` |

So these targets' indices were built **without** the recommended passage prefix;
their scoring caches used the correct one. `build_faiss.py` uses
`index_passage_prefix` to match the frozen indices exactly. (The `index_check`
test under `_local/tests/` empirically confirms which prefix each frozen index
was built with.)
