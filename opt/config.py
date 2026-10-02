"""TellTail-OPT configuration: attack hyper-parameters + per-model token-blocking budgets.

Prefixes live in `configs/models.yaml` (via `telltail.models`), not here.
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


# =========================================================================== #
# Topic-level attack (response-only evaluation)
# =========================================================================== #
# A second optimize mode (`--mode topic`): instead of one victim passage, the target is
# the CENTROID of a group of passages drawn from one topic, embedded with the attacked
# model. Everything else — the GASLITE optimizer, the semantic blocking, prefixes — is
# shared with the passage-level attack above. (This is what the paper calls threat model
# 3 / response-only: the attacker sees only the generated answer, not the ranking.)

# Grouping: one benign query + 100 passages -> NUM_TOPIC_GROUPS groups -> one trigger each.
NUM_TOPIC_GROUPS = 10
PASSAGES_PER_GROUP = 10

# The forbidden-token set is computed ONCE per attack model from the N longest passages
# (of ALL passages), joined — NOT from a single group's passages.
TOP_N_BLOCKING_PASSAGES = 5
BLOCKING_JOINER = "\n\n"

# Third blocking stage (basic-BPE lexical): every {word, lower, upper, capitalize} variant
# of each block word is tokenized and forbidden on top of the semantic budget. Mincut is
# deliberately NOT ported.
BLOCK_WORDS_PATH = "opt/inputs/hp_block_words.txt"

# ---- Response-only evaluation (retrieve -> RAG LLM) ----
# Retrieve default is small (3) because full top-100 retrieval is slow; reviewers who want
# the paper's top-100 set `--k 100`. The LLM consumes the top-3 either way.
RETRIEVE_K_DEFAULT = 3
LLM_TOPK_DEFAULT = 3
NPROBE = 8192

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.8
LLM_NUM_CALLS = 1

# Open WebUI RAG template (verbatim from the research rag_llm.py). {{CONTEXT}} is filled
# with the retrieved passages as <source id="i"> blocks; the user query is appended after.
RAG_TEMPLATE = """### Task:
Respond to the user query using the provided context, incorporating inline citations in the format [id] **only when the <source> tag includes an explicit id attribute** (e.g., <source id="1">).

### Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- **Only include inline citations using [id] (e.g., [1], [2]) when the <source> tag includes an id attribute.**
- Do not cite if the <source> tag does not contain an id attribute.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.

### Example of Citation:
If the user asks about a specific topic and the information is found in a source with a provided id attribute, the response should include the citation like in the following example:
* "According to the study, the proposed method increases efficiency by 20% [1]."

### Output:
Provide a clear and direct response to the user's query, including inline citations in the format [id] only when the <source> tag with id attribute is present in the context.

<context>
{{CONTEXT}}
</context>"""
