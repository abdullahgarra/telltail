"""Output-path layout for the generic-queries pipeline.

All artifacts live under ``$TELLTAIL_OUT_DIR/generic/<query_set>/`` in four
subdirs: retrieval/, passages/, scores/, results/. The query-set name is the
stem of the ``--queries`` CSV.
"""
from __future__ import annotations

from pathlib import Path

from telltail.paths import out_dir


def query_set_name(queries_csv) -> str:
    return Path(queries_csv).stem


def generic_dir(query_set: str) -> Path:
    return out_dir() / "generic" / query_set


def retrieval_dir(query_set: str) -> Path:
    return generic_dir(query_set) / "retrieval"


def passages_dir(query_set: str) -> Path:
    return generic_dir(query_set) / "passages"


def scores_dir(query_set: str) -> Path:
    return generic_dir(query_set) / "scores"


def results_dir(query_set: str) -> Path:
    return generic_dir(query_set) / "results"
