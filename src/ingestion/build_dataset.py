
from pathlib import Path

import pandas as pd

# Project root: two directories above this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "essays.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "essays.csv"

def main():
    metadata = pd.read_csv(
        METADATA_FILE,
        dtype={"essay_id": "string"},
        keep_default_na=False,
    )

    required_columns = {
        "essay_id",
        "title",
        "published_date",
        "filename",
        "source_url",
    }

    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if metadata["essay_id"].duplicated().any():
        raise ValueError("Duplicate essay_id values found.")

    if metadata["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found in metadata.")

    records = []

    for _, row in metadata.iterrows():
        filename = row["filename"].strip()
        title = row["title"].strip()

        if not filename or not title:
            raise ValueError("Every record must have a title and filename.")

        essay_path = (RAW_DIR / filename).resolve()

        # Ensure paths stay inside data/raw
        if not essay_path.is_relative_to(RAW_DIR.resolve()):
            raise ValueError(f"Invalid essay path: {filename}")

        if not essay_path.is_file():
            raise FileNotFoundError(
                f"Essay file not found: {essay_path}\n"
                f"Add the original text to data/raw/{filename}"
            )

        text = essay_path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Essay is empty: {filename}")

        records.append(
            {
                "essay_id": row["essay_id"],
                "title": title,
                "published_date": row["published_date"],
                "text": text,
                "word_count": len(text.split()),
                "source_url": row["source_url"].strip(),
            }
        )

    dataset = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Successfully processed {len(dataset)} essays.")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(dataset[["essay_id", "title", "word_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
