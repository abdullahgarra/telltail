#!/usr/bin/env bash
# Reproduce the 8 paper CSVs for the generic-query attacks (TellTail-Random /
# TellTail-Topic) across TM1/TM2.
#
# Prerequisites:
#   * FAISS indices built first:  python indexing/build_faiss.py --all
#   * A populated .env (see .env.example): TELLTAIL_DATA_DIR, TELLTAIL_INDEX_DIR,
#     TELLTAIL_OUT_DIR. OPENAI_API_KEY is only needed if the openai-3-small model
#     is in the set.
#   * Query sets in generic_queries/queries/: msmarco_topic.csv, msmarco_random.csv.
#
# SKIP_UPSTREAM=1 skips retrieve/fetch/score and reuses the existing score cache
# ($TELLTAIL_OUT_DIR/generic/<query_set>/scores); evaluate/sweep then run off it:
#   SKIP_UPSTREAM=1 bash scripts/reproduce_generic.sh
#
# Exact paper configs (these reproduce the 8 goldens):
#   budgets = 1,5,10,12,15,18,20 ; seed 1337 ; corpus same_as_top_k ; enumerate on (default)
#   TM1: top_ks 1,2,3,5,10,20,50   n_repeats 100
#   TM2: top_ks 1,2,3,4,5,10,20,50 n_repeats 500
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_UPSTREAM="${SKIP_UPSTREAM:-0}"
BUDGETS="1,5,10,12,15,18,20"
TM1_TOPKS="1,2,3,5,10,20,50"
TM2_TOPKS="1,2,3,4,5,10,20,50"

for QS in msmarco_topic msmarco_random; do
  CSV="generic_queries/queries/${QS}.csv"
  echo "############ query set: ${QS} ############"

  # ---- upstream pipeline (needs FAISS indices + msmarco corpus + GPU) --------
  # Recreates the score cache from scratch. SKIP_UPSTREAM=1 skips these and reuses
  # the existing cache; evaluate/sweep below fall back to it by default.
  if [ "$SKIP_UPSTREAM" != "1" ]; then
    python -m generic_queries retrieve --queries "$CSV" --top-k 50
    python -m generic_queries fetch    --queries "$CSV" --max-k 50
    python -m generic_queries score    --queries "$CSV" --corpus-top-k 50
  else
    echo "  [SKIP_UPSTREAM=1] reusing existing score cache; skipping retrieve/fetch/score"
  fi

  # ---- TM1 (ordered exact match) --------------------------------------------
  python -m generic_queries evaluate --interface tm1 --queries "$CSV" \
    --top-ks "$TM1_TOPKS" --budgets "$BUDGETS" --n-repeats 100 \
    --seed 1337 --corpus same_as_top_k
  python -m generic_queries sweep    --interface tm1 --queries "$CSV" \
    --top-ks "$TM1_TOPKS"

  # ---- TM2 (unordered set match) --------------------------------------------
  python -m generic_queries evaluate --interface tm2 --queries "$CSV" \
    --top-ks "$TM2_TOPKS" --budgets "$BUDGETS" --n-repeats 500 \
    --seed 1337 --corpus same_as_top_k
  python -m generic_queries sweep    --interface tm2 --queries "$CSV" \
    --top-ks "$TM2_TOPKS"
done

echo "Done. Results under \$TELLTAIL_OUT_DIR/generic/<query_set>/results/."
echo "Plot with: python plots/asr_vs_budget.py --tm 1  (and --tm 2)"
