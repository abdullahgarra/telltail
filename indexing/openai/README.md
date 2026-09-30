# indexing/openai/ — build the OpenAI (`text-embedding-3-small`) index

The OpenAI model can't be embedded synchronously (8.8M passages × sync API calls
is infeasible), so its FAISS index is built through the OpenAI **Batch API** in
three steps. The final `.index`/`_docids.npy`/`_spec.json` land in
`$TELLTAIL_INDEX_DIR` as `openai_text-embedding-3-small.*`, which `retrieve` uses.

```bash
# 1. write Batch-API request files (streams the corpus; no API key needed)
python indexing/openai/1_make_batches.py --out-dir _local/openai_batches

# 2. submit to the Batch API, poll, download embedding shards (KEY FROM ENV)
OPENAI_API_KEY=... python indexing/openai/2_run_batches.py \
    --batch-dir _local/openai_batches --out-dir _local/openai_run

# 3. assemble the IVF+SQ8 index from the shards -> $TELLTAIL_INDEX_DIR
python indexing/openai/3_build_from_shards.py \
    --shard-dir _local/openai_run/emb_shards --out-dir $TELLTAIL_INDEX_DIR
```

Notes:
- The API key is read **only** from `OPENAI_API_KEY` in the environment; it is
  never hardcoded or passed on the command line.
- Step 2 is resumable (state in `--out-dir/batch_state.json`); it re-uses any
  shards already written.
- Index type matches the local models: `IndexIVFScalarQuantizer(IndexFlatIP, dim,
  nlist=8192, QT_8bit, METRIC_INNER_PRODUCT)`, L2-normalized (IP = cosine).

### Approximate cost

`text-embedding-3-small` is **$0.02 / 1M tokens**, and the Batch API is **50% off**
(~$0.01 / 1M). MS MARCO is ~8.8M passages; at a rough ~60–80 tokens/passage that's
~0.5–0.7B tokens, i.e. **roughly $5–7** for the full corpus via Batch (about
$10–14 without the batch discount). Actual cost depends on passage lengths.

> This step may fail if the OpenAI account has no available balance; if so, it is
> fine to skip — just report the failure. The rest of the pipeline (all local
> models) does not depend on it.
