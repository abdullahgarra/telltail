"""Sanity checks on the model registry."""
from telltail.models import load_registry, candidates


def test_counts():
    reg = load_registry()
    assert len(reg) == 53, f"expected 53 models, got {len(reg)}"
    assert len(candidates()) == 19, f"expected 19 candidates, got {len(candidates())}"


def test_e5_prefixes_exact():
    reg = load_registry()
    for alias in ("e5-small", "e5-base", "e5-large"):
        assert reg[alias]["query_prefix"] == "query: "
        assert reg[alias]["passage_prefix"] == "passage: "


def test_gtr_alias_is_canonical():
    reg = load_registry()
    assert "gtr-t5-base" in reg
    assert "gtr-t5" not in reg
