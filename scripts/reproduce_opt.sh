#!/usr/bin/env bash
# Reproduce the TellTail-OPT results.
#
#   TM1/TM2 : optimize -> evaluate -> score   (OSCR + ASR goldens, results/paper/tm{1,2}_*)
#   TM3     : optimize --mode topic -> retrieve -> llm -> judge -> heatmap  (response-only, opt-in)
#
# Prerequisites:
#   * FAISS indices built first:  python indexing/build_faiss.py --all
#   * A populated .env (see .env.example): TELLTAIL_DATA_DIR, TELLTAIL_INDEX_DIR,
#     TELLTAIL_OUT_DIR. OPENAI_API_KEY is only needed if openai-3-small is in the eval set.
#   * A GPU for the optimizer (GASLITE).
#   * TM3 only: the FULL passage store (python tools/build_passage_store.py, no --sample)
#     plus OPENAI_API_KEY (RAG generation) and DEEPINFRA_API_KEY (LLM judge).
#
# Knobs:
#   SKIP_OPTIMIZE=1  score TM1/TM2 straight from the shipped ranks
#                    (results/opt/tm1_tm2_ranks/ranks.csv) — no GPU, no indices.
#   RUN_TM3=1        also run the response-only TM3 chain (needs keys + full store).
#   EVAL_MODELS=...  comma-separated eval aliases (default: all registry models).
#   OUT=...          work dir for generated artifacts (default: outputs/opt).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-outputs/opt}"
SKIP_OPTIMIZE="${SKIP_OPTIMIZE:-0}"
RUN_TM3="${RUN_TM3:-0}"
RANKS="results/opt/tm1_tm2_ranks/ranks.csv"
EVAL_MODELS="${EVAL_MODELS:-$(python -c 'from telltail.models import load_registry; print(",".join(sorted(load_registry())))')}"
mkdir -p "$OUT"

# ---- TM1 / TM2 : optimize -> evaluate -> score ----------------------------
if [ "$SKIP_OPTIMIZE" = "1" ]; then
  echo "[SKIP_OPTIMIZE=1] scoring from shipped ranks: $RANKS"
  LONG="$RANKS"
else
  python -m opt.optimize --queries opt/inputs/queries.csv --out "$OUT/phase1"
  python -m opt.evaluate --attacks "$OUT/phase1/phase1_attacks.csv" \
    --out "$OUT/opt_long.csv" --eval-models "$EVAL_MODELS"
  LONG="$OUT/opt_long.csv"
fi
python -m opt.score --tm 1 --long "$LONG" --out "$OUT/opt_tm1"
python -m opt.score --tm 2 --long "$LONG" --out "$OUT/opt_tm2"

# ---- TM3 (response-only), opt-in ------------------------------------------
if [ "$RUN_TM3" = "1" ]; then
  echo "############ TM3 response-only chain ############"
  # Needs the FULL passage store (no --sample) + OPENAI_API_KEY + DEEPINFRA_API_KEY.
  python -m opt.optimize --mode topic --queries opt/inputs/query_passages.csv --out "$OUT/phase1_topic"
  python -m opt.retrieve --attacks "$OUT/phase1_topic/phase1_attacks.csv" \
    --out "$OUT/tm3_retrieved.csv" --eval-models "$EVAL_MODELS" --k 100
  python -m opt.llm   --retrieved "$OUT/tm3_retrieved.csv" \
    --attacks "$OUT/phase1_topic/phase1_attacks.csv" --out "$OUT/tm3_responses"
  python -m opt.judge --responses "$OUT/tm3_responses" --out "$OUT/tm3_judge"
  python plots/judge_heatmap.py --judgments "$OUT/tm3_judge" --out "$OUT/tm3_judge"
fi

echo "Done. OPT scores under $OUT/ (opt_tm1*, opt_tm2*)."
echo "Plot with: python plots/asr_vs_budget.py --tm 1 (and --tm 2); plots/plot_ccr_0_oscr.py --tm {1,2}"
