
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    df = pd.read_csv(DATASET_FILE, keep_default_na=False)

    # Basic dataset statistics
    print("\n=== ARCHIVE OVERVIEW ===")
    print(f"Number of essays: {len(df)}")
    print(f"Total words: {df['word_count'].sum():,}")
    print(f"Average words per essay: {df['word_count'].mean():,.1f}")
    print(f"Shortest essay: {df.loc[df['word_count'].idxmin(), 'title']}")
    print(f"Longest essay: {df.loc[df['word_count'].idxmax(), 'title']}")

    # Sentence-level statistics
    sentence_pattern = r"[.!?]+(?:\s+|$)"
    df["sentence_count"] = (
        df["text"].str.count(sentence_pattern)
    )
    df["avg_sentence_length"] = (
        df["word_count"] / df["sentence_count"].replace(0, pd.NA)
    )

    # Vocabulary statistics
    df["unique_word_count"] = df["text"].apply(
        lambda text: len(set(
            word.lower().strip(".,!?;:\"'()[]{}")
            for word in text.split()
            if word.strip(".,!?;:\"'()[]{}")
        ))
    )

    df["vocabulary_ratio"] = (
        df["unique_word_count"] / df["word_count"]
    )

    # Display per-essay results
    print("\n=== ESSAY-LEVEL STATISTICS ===")
    columns = [
        "title",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "unique_word_count",
        "vocabulary_ratio",
    ]
    print(
        df[columns]
        .round(3)
        .sort_values("word_count", ascending=False)
        .to_string(index=False)
    )

    # Save analysis results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "essay_statistics.csv"
    df[columns].to_csv(output_file, index=False)

    print(f"\nStatistics saved to: {output_file}")


if __name__ == "__main__":
    main()
