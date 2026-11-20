"""Build the processed essay dataset from metadata plus raw text files.

Essays copied from the Wix blog begin with a page header:

    <title>
    <date>          e.g. "Oct 29, 2025" (Wix omits the year for current-year posts)
    <N> min read

When that header is present, this script:
  * checks the header title matches the metadata title (catches swapped files),
  * uses the header date to fill a blank published_date (or checks it agrees),
  * removes the header so it does not leak into the analysis.
"""
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import re

import pandas as pd

# Project root: two directories above this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "essays.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "essays.csv"

WIX_HEADER = re.compile(
    r"\A(?P<title>[^\n]+)\n"
    r"(?P<date>[A-Z][a-z]{2} \d{1,2}(?:, \d{4})?)\n"
    r"\d+ min read[ \t]*(?:\n|\Z)"
)
TITLE_MATCH_THRESHOLD = 0.75


def parse_wix_header(text):
    """Return (header_title, header_date, body). Title and date are None if no header."""
    match = WIX_HEADER.match(text)
    if not match:
        return None, None, text
    return match.group("title").strip(), match.group("date"), text[match.end():].strip()


def normalize_title(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def titles_match(a, b):
    """Tolerant comparison: ignores case and punctuation, allows small copy errors."""
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio() >= TITLE_MATCH_THRESHOLD


def header_date_to_iso(date_text):
    """'Oct 29, 2025' -> '2025-10-29'. Returns None when the year is missing."""
    if not date_text or "," not in date_text:
        return None
    return datetime.strptime(date_text, "%b %d, %Y").strftime("%Y-%m-%d")


def resolve_date(metadata_date, header_date, title):
    """Metadata wins; the header fills gaps; a disagreement is an error."""
    metadata_date = (metadata_date or "").strip()
    header_iso = header_date_to_iso(header_date)
    if metadata_date and header_iso and metadata_date != header_iso:
        raise ValueError(
            f"Date mismatch for '{title}': metadata {metadata_date}, page header {header_iso}"
        )
    if metadata_date:
        return metadata_date
    if header_iso:
        return header_iso
    if header_date:
        print(f"WARNING: '{title}' header date '{header_date}' has no year; "
              f"add published_date to {METADATA_FILE.name}.")
    return ""


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

        raw_text = essay_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        header_title, header_date, text = parse_wix_header(raw_text)

        if header_title is not None and not titles_match(header_title, title):
            raise ValueError(
                f"Title mismatch in {filename}: metadata says '{title}' but the file "
                f"starts with '{header_title}'. The file may hold a different essay."
            )

        if not text:
            raise ValueError(f"Essay is empty: {filename}")

        records.append(
            {
                "essay_id": row["essay_id"],
                "title": title,
                "published_date": resolve_date(row["published_date"], header_date, title),
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
    print(dataset[["essay_id", "title", "published_date", "word_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
