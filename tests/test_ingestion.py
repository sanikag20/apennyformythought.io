
import pandas as pd
import pytest

from src.ingestion.build_dataset import main


def test_pipeline_builds_dataset(tmp_path, monkeypatch):
    """The pipeline should create a valid dataset from essay files."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True)

    (raw_dir / "essay_one.txt").write_text(
        "I write to understand myself.",
        encoding="utf-8",
    )

    metadata = pd.DataFrame(
        [
            {
                "essay_id": "1",
                "title": "Essay One",
                "published_date": "",
                "filename": "essay_one.txt",
                "source_url": "https://example.com/essay-one",
            }
        ]
    )

    metadata_file = data_dir / "essays.csv"
    metadata.to_csv(metadata_file, index=False)

    # Redirect the pipeline to our temporary test data.
    monkeypatch.setattr(
        "src.ingestion.build_dataset.PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "src.ingestion.build_dataset.DATA_DIR",
        data_dir,
    )
    monkeypatch.setattr(
        "src.ingestion.build_dataset.RAW_DIR",
        raw_dir,
    )
    monkeypatch.setattr(
        "src.ingestion.build_dataset.METADATA_FILE",
        metadata_file,
    )
    monkeypatch.setattr(
        "src.ingestion.build_dataset.OUTPUT_FILE",
        processed_dir / "essays.csv",
    )

    main()

    output_file = processed_dir / "essays.csv"
    assert output_file.exists()

    result = pd.read_csv(output_file, dtype={"essay_id": str})

    assert len(result) == 1
    assert result.loc[0, "title"] == "Essay One"
    assert result.loc[0, "word_count"] == 5
    assert result.loc[0, "text"] == "I write to understand myself."


def test_pipeline_rejects_missing_essay_file(tmp_path, monkeypatch):
    """The pipeline should fail rather than silently skip a missing essay."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "essay_id": "1",
                "title": "Missing Essay",
                "published_date": "",
                "filename": "missing.txt",
                "source_url": "",
            }
        ]
    ).to_csv(data_dir / "essays.csv", index=False)

    monkeypatch.setattr(
        "src.ingestion.build_dataset.METADATA_FILE",
        data_dir / "essays.csv",
    )
    monkeypatch.setattr(
        "src.ingestion.build_dataset.RAW_DIR",
        raw_dir,
    )

    with pytest.raises(FileNotFoundError):
        main()
