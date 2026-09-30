"""TellTail-OPT — optimizer configuration.

Prefixes are NOT duplicated here: they come from `configs/models.yaml` via
`telltail.models` (one canonical source for all of TellTail). This file holds only
the attack hyper-parameters and the per-model token-blocking budgets.
"""
from __future__ import annotations

# Reproducibility: the research optimizer seeded random + torch with 42.
SEED = 42

# ---- Token-blocking budget (% of the candidate model's vocab to forbid) ----
# Semantic "global weighted blocking": forbid the top-`percent`% of vocab tokens by
# max cosine similarity to the target passage's word embeddings (passage tokens get
# priority). Budgets are set directly (not derived from the golden queries):
#   gemma-300m      90%   openai-3-small  25%   every other model  50%
BLOCKING_PERCENT_DEFAULT = 50.0
BLOCKING_PERCENT = {
    "gemma-300m": 90.0,
    "openai-3-small": 25.0,
}


def blocking_percent(alias: str) -> float:
    return BLOCKING_PERCENT.get(alias, BLOCKING_PERCENT_DEFAULT)


# ---- White-box optimizer (GASLITE) — HF encoder models ----
# Ported verbatim from the research config (TRIGGER_LEN/N_GRAD/N_CAND/N_FLIP/N_ITER).
TRIGGER_LEN = 30           # initial trigger length, rendered as ("! " * N).strip()
WHITEBOX = dict(
    n_candidates=128,
    num_steps=100,
    n_grad=10,
    n_flip=20,
    use_retokenize=True,
)

# ---- Black-box optimizer (RASLITE+) — OpenAI / API-only models ----
# num_steps is the released setting. NOTE: the paper TM1/TM2 queries were optimized
# with 200 steps; released queries reproduce the paper numbers.
BLACKBOX = dict(
    num_steps=1000,
    n_candidates=100,
    n_flip=1,
    use_random_logits=True,
    flip_pos_method="ordered",
    buffer_size=10,
    n_bulk_flips=1,
    use_retokenize=True,
    util_lm_name="qwen/qwen2.5-0.5b",
)

# Passage-tokenization cap when computing word embeddings for blocking (research used 512).
BLOCKING_MAX_SEQ_LENGTH = 512
