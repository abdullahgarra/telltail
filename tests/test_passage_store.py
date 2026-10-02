"""Passage store (SQLite) + opt.retrieve text attachment.

Guarantees the happy path: when the store covers the retrieved ids, every retrieved row
gets the CORRECT passage_text for its id.
"""
import csv
import re
import sys
import types

import numpy as np

from telltail.passage_store import create_store, insert_batch, get_passages


def _make_store(path, mapping):
    conn = create_store(path)
    insert_batch(conn, list(mapping.items()))
    conn.commit()
    conn.close()


def test_get_passages_maps_id_to_text(tmp_path):
    db = tmp_path / "s.sqlite"
    mapping = {"123": "alpha", "456": "beta", "789": "gamma"}
    _make_store(db, mapping)
    got = get_passages(["456", "123"], db_path=db)
    assert got == {"456": "beta", "123": "alpha"}      # right text wired to right id


def test_get_passages_missing_and_empty(tmp_path):
    db = tmp_path / "s.sqlite"
    _make_store(db, {"1": "a", "2": "b"})
    assert get_passages(["1", "nope"], db_path=db) == {"1": "a"}   # missing id absent
    assert get_passages([], db_path=db) == {}


def test_get_passages_chunks_over_sqlite_var_limit(tmp_path):
    db = tmp_path / "s.sqlite"
    mapping = {str(i): f"t{i}" for i in range(2500)}   # > 900-var IN() chunk size
    _make_store(db, mapping)
    got = get_passages([str(i) for i in range(2500)], db_path=db)
    assert len(got) == 2500 and got["2499"] == "t2499"


def test_retrieve_attaches_store_text(tmp_path, monkeypatch):
    """Run opt.retrieve over a tiny real FAISS index; assert each row's passage_text
    equals the store text for that passage_id (mocks only the encoder + registry)."""
    import faiss
    import opt.retrieve as R

    ids = [f"p{i}" for i in range(5)]
    texts = {pid: f"text for {pid}" for pid in ids}
    db = tmp_path / "store.sqlite"
    _make_store(db, texts)

    dim = 8
    vecs = np.eye(5, dim, dtype="float32")            # p_i = e_i
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    hf = "org/model"
    san = re.sub(r"[^a-zA-Z0-9._-]+", "_", hf)
    idir = tmp_path / "indexes"
    idir.mkdir()
    faiss.write_index(idx, str(idir / f"{san}.index"))
    np.save(str(idir / f"{san}_docids.npy"), np.array(ids, dtype=object))

    # mock the registry (one OSS eval model) and the sentence-transformers encoder
    monkeypatch.setattr("telltail.models.load_registry",
                        lambda: {"m1": {"hf_id": hf, "query_prefix": ""}})

    class FakeST:
        def __init__(self, *a, **k):
            pass

        def encode(self, qs, convert_to_numpy=True):
            v = np.zeros((1, dim), dtype="float32")
            v[0, 0] = 1.0                              # closest to p0
            return v

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    attacks = tmp_path / "att.csv"
    with open(attacks, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_text", "query_group", "attack_model", "trigger_suffix"])
        w.writerow(["0", "q", "0", "m1", "trig"])
    out = tmp_path / "ret.csv"

    R.run(attacks, out, ["m1"], k=3, index_dir_override=idir, passage_store_path=db)

    rows = list(csv.DictReader(open(out)))
    assert rows, "retrieve produced no rows"
    for r in rows:
        assert r["passage_text"] == texts[r["passage_id"]]   # correct text for the id
    assert rows[0]["passage_id"] == "p0"                     # nearest came back rank 1
