import json

import numpy as np
import pytest

from src.analysis.search import load_index, search


class FakeEncoder:
    """Maps known queries to fixed unit vectors, so tests need no model download."""

    VECTORS = {
        "home": [1.0, 0.0, 0.0],
        "travel": [0.0, 1.0, 0.0],
    }

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self.VECTORS[t] for t in texts])


RECORDS = [
    {"essay_id": "1", "title": "No Place Like Home", "passage_no": 1, "text": "hostel food"},
    {"essay_id": "1", "title": "No Place Like Home", "passage_no": 2, "text": "friends"},
    {"essay_id": "2", "title": "Caught in a Landslide", "passage_no": 1, "text": "Sikkim trip"},
    {"essay_id": "3", "title": "Lost", "passage_no": 1, "text": "wandering"},
]

VECTORS = np.array([
    [0.95, 0.31, 0.0],   # essay 1, strongly "home"
    [0.90, 0.44, 0.0],   # essay 1, also "home"
    [0.10, 0.99, 0.0],   # essay 2, "travel"
    [0.60, 0.80, 0.0],   # essay 3, mixed
])


def test_search_ranks_by_cosine_similarity():
    results = search("home", RECORDS, VECTORS, FakeEncoder(), k=3)
    assert [r["passage_no"] for r in results[:2]] == [1, 2]
    assert results[0]["title"] == "No Place Like Home"
    assert results[0]["score"] == pytest.approx(0.95)
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_search_by_essay_returns_each_essay_once():
    results = search("home", RECORDS, VECTORS, FakeEncoder(), k=3, by_essay=True)
    ids = [r["essay_id"] for r in results]
    assert ids == ["1", "3", "2"]
    assert len(set(ids)) == len(ids)


def test_search_finds_other_topic():
    results = search("travel", RECORDS, VECTORS, FakeEncoder(), k=1)
    assert results[0]["title"] == "Caught in a Landslide"


def test_search_rejects_empty_query():
    with pytest.raises(ValueError):
        search("   ", RECORDS, VECTORS, FakeEncoder())


def test_load_index_detects_mismatch(tmp_path):
    with (tmp_path / "passages.jsonl").open("w", encoding="utf-8") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
    np.save(tmp_path / "embeddings.npy", VECTORS[:2])
    with pytest.raises(ValueError):
        load_index(tmp_path)


def test_load_index_reads_model_name(tmp_path):
    with (tmp_path / "passages.jsonl").open("w", encoding="utf-8") as f:
        for r in RECORDS:
            f.write(json.dumps(r) + "\n")
    np.save(tmp_path / "embeddings.npy", VECTORS)
    (tmp_path / "model.txt").write_text("my-model", encoding="utf-8")
    records, vectors, model = load_index(tmp_path)
    assert len(records) == 4 and vectors.shape == (4, 3) and model == "my-model"
