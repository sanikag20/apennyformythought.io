
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"

EXPECTED_COLUMNS = {
    "essay_id",
    "title",
    "published_date",
    "text",
    "word_count",
    "source_url",
}


def main():
    df = pd.read_csv(
        DATASET_FILE,
        dtype={"essay_id": "string"},
        keep_default_na=False,
    )

    errors = []

    # 1. Check the dataset schema
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)

    # 2. Check identifiers and titles
    if df["essay_id"].eq("").any():
        errors.append("Some essays have missing IDs.")

    if df["essay_id"].duplicated().any():
        errors.append("Duplicate essay IDs found.")

    if df["title"].str.strip().eq("").any():
        errors.append("Some essays have missing titles.")

    # 3. Check text and word counts
    if df["text"].isna().any() or df["text"].str.strip().eq("").any():
        errors.append("Some essays have empty text.")

    for _, row in df.iterrows():
        actual_count = len(row["text"].split())

        if actual_count != int(row["word_count"]):
            errors.append(
                f"Word-count mismatch for essay {row['essay_id']}: "
                f"stored={row['word_count']}, actual={actual_count}"
            )

    # 4. Check for duplicate essay text
    normalized_text = df["text"].str.strip().str.casefold()

    if normalized_text.duplicated().any():
        errors.append("Identical essay texts found.")

    # 5. Report results
    print(f"Essays checked: {len(df)}")
    print(f"Total words: {df['word_count'].sum():,}")
    print(f"Unique essay IDs: {df['essay_id'].nunique()}")
    print(f"Essays with publication dates: "
          f"{df['published_date'].str.strip().ne('').sum()}")

    if errors:
        print("\nDATASET AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nDATASET AUDIT PASSED")


if __name__ == "__main__":
    main()
