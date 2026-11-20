"""Tests for the model-free helpers in 05_rag_writer.py and 06_llm_style_comparison.py.

The script names start with digits, so they are loaded with importlib.
No language model or embedding model is downloaded.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

rag = importlib.import_module("src.analysis.05_rag_writer")
style = importlib.import_module("src.analysis.06_llm_style_comparison")


def test_slugify_makes_safe_filenames():
    assert rag.slugify("What have I written about belonging?") == "what-have-i-written-about-belonging"
    assert rag.slugify("???") == "prompt"
    assert len(rag.slugify("word " * 40)) <= 50


def test_read_prompts_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("# header\n\nFirst prompt\n  Second prompt  \n# note\n", encoding="utf-8")
    assert rag.read_prompts(f) == ["First prompt", "Second prompt"]
    assert style.read_prompts(f) == ["First prompt", "Second prompt"]


def test_read_prompts_rejects_empty_file(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rag.read_prompts(f)


def test_build_context_labels_sources():
    matches = [({"title": "Lost", "passage_no": 2, "text": "body"}, 0.5)]
    context = rag.build_context(matches)
    assert "[Lost | passage 2 | retrieval score 0.500]" in context
    assert context.endswith("body")


def test_style_features_counts():
    text = "I went home. My friends were there, laughing! Were we lost?"
    f = style.style_features(text)
    assert f["avg_sentence_words"] == pytest.approx(11 / 3)
    assert f["first_person_per_100_words"] == pytest.approx(100 * 3 / 11)
    assert f["question_marks_per_1000_chars"] == pytest.approx(1000 / len(text))


def test_style_distance_is_zero_at_author_mean():
    means = pd.Series({"a": 1.0, "b": 2.0})
    stds = pd.Series({"a": 0.5, "b": 1.0})
    assert style.style_distance({"a": 1.0, "b": 2.0}, means, stds) == 0.0
    # one standard deviation away on both features -> distance 1
    assert style.style_distance({"a": 1.5, "b": 3.0}, means, stds) == pytest.approx(1.0)


def test_summarize_counts_prompts_where_rag_is_closer():
    metrics = pd.DataFrame([
        {"prompt": "(all)", "condition": "Author corpus mean", "style_distance_from_author_mean": 0.0},
        {"prompt": "p1", "condition": "Base LLM", "style_distance_from_author_mean": 2.0},
        {"prompt": "p1", "condition": "RAG-conditioned LLM", "style_distance_from_author_mean": 1.0},
        {"prompt": "p2", "condition": "Base LLM", "style_distance_from_author_mean": 1.0},
        {"prompt": "p2", "condition": "RAG-conditioned LLM", "style_distance_from_author_mean": 1.5},
    ])
    summary = style.summarize(metrics)
    base = summary[summary["condition"] == "Base LLM"].iloc[0]
    assert base["mean"] == pytest.approx(1.5)
    assert summary["rag_closer_on_prompts"].iloc[0] == "1/2"
