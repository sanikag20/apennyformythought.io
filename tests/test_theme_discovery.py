import importlib

import numpy as np
import pandas as pd
import pytest

themes = importlib.import_module("src.analysis.03_theme_discovery")


def test_vectorize_drops_filler_and_contraction_fragments():
    texts = [
        "Honestly guys, I couldn't stop thinking about the river and the mountains.",
        "Okay, literally the river again, and the mountains, guys.",
        "Spreadsheets and coffee and deadlines at the office.",
    ]
    _, terms = themes.vectorize(texts)
    assert "river" in terms and "mountains" in terms
    for word in ["guys", "okay", "literally", "couldn"]:
        assert word not in terms


def test_choose_k_prefers_smallest_k_near_the_best():
    selection = pd.DataFrame({"k": [2, 3, 4, 5], "stability": [0.40, 0.70, 0.71, 0.60]})
    assert themes.choose_k(selection) == 3
    assert themes.choose_k(selection, tolerance=0.0) == 4


def test_topic_stability_is_high_for_clearly_separated_topics():
    # Two blocks of documents that use disjoint vocabularies.
    rng = np.random.default_rng(0)
    block = rng.uniform(0.5, 1.0, size=(5, 12))
    X = np.zeros((10, 24))
    X[:5, :12] = block
    X[5:, 12:] = rng.uniform(0.5, 1.0, size=(5, 12))
    assert themes.topic_stability(X, 2, seeds=5, n_terms=5) > 0.9


def test_load_labels(tmp_path):
    f = tmp_path / "labels.csv"
    f.write_text("topic,label\n1,Home\n2,Work\n", encoding="utf-8")
    assert themes.load_labels(f, 2) == {1: "Home", 2: "Work"}
    assert themes.load_labels(None, 2) == {}
    with pytest.raises(ValueError):
        themes.load_labels(f, 1)
