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

## Note on passage prefixes

Passage embeddings use the **registry** (`configs/models.yaml`) `passage_prefix`.
The original one-off build script omitted or altered the doc prefix for 5 models,
so a registry-correct rebuild will differ from those frozen indices:

| model | original builder | registry (used here) |
|---|---|---|
| e5-base | `""` (missing) | `"passage: "` |
| nomic-v1.5 | `""` (missing) | `"search_document: "` |
| nomic-v1 | `""` (missing) | `"search_document: "` |
| jina-v5-small | `""` (missing) | `"Document:"` |
| gemma-300m | `"title: none | text: "` | `"title: none | text:"` |

The registry prefixes are the model-correct ones; this builder standardizes on them.
