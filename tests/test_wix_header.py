import pandas as pd
import pytest

import src.ingestion.build_dataset as bd

WIX_TEXT = "World's Smallest Violin\nOct 29, 2025\n6 min read\n\nI don't have a topic in mind."


def test_parse_wix_header_strips_header():
    title, date, body = bd.parse_wix_header(WIX_TEXT)
    assert title == "World's Smallest Violin"
    assert date == "Oct 29, 2025"
    assert body == "I don't have a topic in mind."


def test_parse_wix_header_leaves_plain_text_alone():
    assert bd.parse_wix_header("Just an essay.") == (None, None, "Just an essay.")


def test_titles_match_tolerates_punctuation_and_copy_errors():
    assert bd.titles_match("ours Truly, Debby Downer.", "Yours Truly Debby Downer")
    assert bd.titles_match("Caught in a landslide", "Caught in a Landslide")
    assert not bd.titles_match("World's Smallest Violin", "Not Like Other Girls.")


def test_header_date_to_iso():
    assert bd.header_date_to_iso("Jun 20, 2024") == "2024-06-20"
    assert bd.header_date_to_iso("Mar 7") is None


def test_resolve_date_rules():
    assert bd.resolve_date("", "Oct 29, 2025", "t") == "2025-10-29"
    assert bd.resolve_date("2026-03-07", "Mar 7", "t") == "2026-03-07"
    assert bd.resolve_date("", "Mar 7", "t") == ""
    with pytest.raises(ValueError):
        bd.resolve_date("2024-01-01", "Oct 29, 2025", "t")


def _setup(tmp_path, monkeypatch, filename, text, title, date=""):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / filename).write_text(text, encoding="utf-8")
    pd.DataFrame([{
        "essay_id": "1", "title": title, "published_date": date,
        "filename": filename, "source_url": "",
    }]).to_csv(data_dir / "essays.csv", index=False)
    monkeypatch.setattr(bd, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bd, "METADATA_FILE", data_dir / "essays.csv")
    monkeypatch.setattr(bd, "OUTPUT_FILE", data_dir / "processed" / "essays.csv")
    return data_dir / "processed" / "essays.csv"


def test_pipeline_strips_header_and_fills_date(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch, "violin.txt", WIX_TEXT, "World's Smallest Violin")
    bd.main()
    row = pd.read_csv(out, keep_default_na=False).iloc[0]
    assert row["text"] == "I don't have a topic in mind."
    assert row["published_date"] == "2025-10-29"
    assert row["word_count"] == 7


def test_pipeline_rejects_swapped_essay_file(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, "girls.txt", WIX_TEXT, "Not Like Other Girls.")
    with pytest.raises(ValueError, match="Title mismatch"):
        bd.main()
