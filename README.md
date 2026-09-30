# TellTail

Fingerprinting retrievers in black-box systems.

> Placeholder README — setup and usage instructions to follow.

## Layout

- `telltail/` — core library (model registry, paths, embedding).
- `configs/models.yaml` — the 53-model registry (19 candidates).
- `indexing/` — build FAISS indices over the corpus (not shipped; too large).
- `generic_queries/` — TellTail-Random / TellTail-Topic.
- `opt/` — TellTail-OPT (model-specific optimized queries).
- `plots/` — figure generation.
- `tools/` — one-off build scripts (e.g. `build_models_yaml.py`).

## Setup

```bash
pip install -e .
cp .env.example .env   # then fill in the paths
```
