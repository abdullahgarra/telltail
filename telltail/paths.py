"""Single source of truth for filesystem paths.

All large artifacts (the corpus, FAISS indices, experiment outputs) live outside
the repository and are located exclusively through environment variables, so no
absolute path is ever hardcoded in the code. Set them in a ``.env`` file at the
repo root (see ``.env.example``); values are loaded here via python-dotenv.

    TELLTAIL_DATA_DIR   corpus, queries, and other input data
    TELLTAIL_INDEX_DIR  precomputed FAISS indices (one per retriever)
    TELLTAIL_OUT_DIR    experiment outputs (rankings, scores, figures)
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv, find_dotenv
import os

# Load .env from the repo root (or nearest ancestor) if present. Real
# environment variables always take precedence over .env values.
load_dotenv(find_dotenv(usecwd=True))


def _require_dir(env_var: str) -> Path:
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Copy .env.example to .env and set {env_var} (or export it). "
            f"Paths are resolved only through environment variables."
        )
    return Path(val).expanduser().resolve()


def data_dir() -> Path:
    """Directory holding corpus / query inputs (TELLTAIL_DATA_DIR)."""
    return _require_dir("TELLTAIL_DATA_DIR")


def index_dir() -> Path:
    """Directory holding precomputed FAISS indices (TELLTAIL_INDEX_DIR)."""
    return _require_dir("TELLTAIL_INDEX_DIR")


def out_dir() -> Path:
    """Directory for experiment outputs (TELLTAIL_OUT_DIR)."""
    return _require_dir("TELLTAIL_OUT_DIR")
