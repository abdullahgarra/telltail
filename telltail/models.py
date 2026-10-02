"""Model registry, prefixing, and embedding.

Loads the 53-model registry from ``configs/models.yaml`` and provides a unified
embedding interface over two backends: local sentence-transformers models and
the OpenAI embedding API. Model-specific query/passage prefixes are applied
here, byte-exact as stored in the registry.

The OpenAI API key is read ONLY from the environment (OPENAI_API_KEY); there is
no CLI flag or function argument for it.
"""
from __future__ import annotations

import functools
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"

# Cache lazily-imported optional backends so we only pay for what we use.
try:  # torch is only needed to pick a device for local encoders
    import torch
except Exception:  # pragma: no cover - torch optional at import time
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from openai import OpenAI as _OpenAIClient
    _openai_available = True
except ImportError:  # pragma: no cover
    _openai_available = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_registry(config_path: Optional[Path] = None) -> Dict[str, dict]:
    """Return {alias: entry} from models.yaml.

    Each entry has: alias, hf_id, query_prefix, passage_prefix, candidate,
    backend. Prefixes are preserved exactly as written in the YAML.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    entries = raw["models"]
    registry = {e["alias"]: e for e in entries}
    if len(registry) != len(entries):
        raise ValueError("duplicate alias in models.yaml")
    return registry


def candidates() -> List[str]:
    """Aliases in the attacker candidate set S (candidate: true)."""
    reg = load_registry()
    return [a for a, e in reg.items() if e.get("candidate")]


def _entry(alias: str) -> dict:
    reg = load_registry()
    if alias not in reg:
        raise KeyError(f"unknown model alias: {alias!r}")
    return reg[alias]


def is_openai(alias: str) -> bool:
    return _entry(alias).get("backend") == "openai"


# ---------------------------------------------------------------------------
# Prefixing
# ---------------------------------------------------------------------------
def apply_prefix(alias: str, texts: List[str], kind: str) -> List[str]:
    """Prepend the model-specific prefix. ``kind`` is 'query' or 'passage'.

    OpenAI models use no external prefix (the API handles instruction).
    """
    prefix_key = {"query": "query_prefix", "passage": "passage_prefix",
                  "index_passage": "index_passage_prefix"}.get(kind)
    if prefix_key is None:
        raise ValueError("kind must be 'query', 'passage', or 'index_passage'")
    entry = _entry(alias)
    if entry.get("backend") == "openai":
        return list(texts)
    pref = entry.get(prefix_key, "")
    return [f"{pref}{t}" for t in texts] if pref else list(texts)


# ---------------------------------------------------------------------------
# Encoders / embedding
# ---------------------------------------------------------------------------
def _gpu_usable() -> bool:
    """True only if a CUDA device is present AND this torch build has a kernel for it.

    ``torch.cuda.is_available()`` can report True on a GPU whose compute capability
    the installed torch has no kernels for (e.g. an older card); the failure then
    surfaces only later, at the first op. Probe with a tiny op so ``device=None``
    (auto) falls back to CPU instead of crashing mid-encode.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        # A plain .cuda() copy can succeed on an unsupported card; force a real
        # compute kernel + sync so "no kernel image" surfaces here, not mid-encode.
        x = torch.ones(8, 8, device="cuda")
        (x @ x).sum().item()
        return True
    except Exception:
        return False


def _default_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if _gpu_usable() else "cpu"


def load_encoder(alias: str, device: Optional[str] = None):
    """Load a local sentence-transformers encoder. Returns None for OpenAI."""
    entry = _entry(alias)
    if entry.get("backend") == "openai":
        return None
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed.")
    return SentenceTransformer(
        entry["hf_id"], device=_default_device(device), trust_remote_code=True
    )


def _embed_openai(
    texts: List[str],
    oai_model: str,
    batch_size: int,
    normalize: bool,
    max_retries: int = 8,
    initial_backoff: float = 1.0,
) -> np.ndarray:
    if not _openai_available:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = _OpenAIClient(api_key=key)
    all_embs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        backoff = initial_backoff
        vecs = None
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(input=batch, model=oai_model)
                vecs = np.array(
                    [e.embedding for e in sorted(resp.data, key=lambda x: x.index)],
                    dtype=np.float32,
                )
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait = backoff * (2 ** attempt)
                print(
                    f"  [OpenAI] Attempt {attempt+1} failed ({exc}). "
                    f"Retrying in {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        all_embs.append(vecs)
    return np.vstack(all_embs).astype(np.float32, copy=False)


def embed(
    alias: str,
    texts: List[str],
    kind: str = "query",
    normalize: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
    encoder=None,
) -> np.ndarray:
    """Embed ``texts`` with model ``alias``, applying its prefix for ``kind``.

    For local models, pass a preloaded ``encoder`` to avoid reloading; if None,
    one is loaded on demand. OpenAI models ignore ``encoder``/``device``.
    """
    texts = apply_prefix(alias, list(texts), kind)
    if is_openai(alias):
        oai_name = _entry(alias)["hf_id"].split("openai/", 1)[-1]
        return _embed_openai(
            texts, oai_model=oai_name,
            batch_size=min(batch_size, 256), normalize=normalize,
        )
    model = encoder if encoder is not None else load_encoder(alias, device)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    ).astype(np.float32, copy=False)
