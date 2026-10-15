
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENS_FILE = PROJECT_ROOT / "data" / "processed" / "essay_tokens.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    tokens = pd.read_csv(TOKENS_FILE, keep_default_na=False)

    # Keep actual alphabetic words, excluding punctuation and numbers.
    words = tokens[
        tokens["is_alpha"].astype(str).str.lower().eq("true")
        & tokens["is_punctuation"].astype(str).str.lower().eq("false")
    ].copy()

    words["lowercase"] = words["lowercase"].str.lower()
    words["lemma"] = words["lemma"].str.lower()

    # 1. Most frequent surface forms, retaining stopwords.
    surface_counts = (
        words["lowercase"]
        .value_counts()
        .rename_axis("word")
        .reset_index(name="count")
    )

    # 2. Most frequent lemmas, retaining stopwords.
    lemma_counts = (
        words["lemma"]
        .value_counts()
        .rename_axis("lemma")
        .reset_index(name="count")
    )

    # 3. Most frequent content lemmas.
    content = words[~words["is_stop"].astype(str).str.lower().eq("true")]
    content_lemmas = (
        content["lemma"]
        .value_counts()
        .rename_axis("lemma")
        .reset_index(name="count")
    )

    # 4. Grammatical composition of each essay.
    pos_counts = (
        words.groupby(["essay_id", "title", "pos"])
        .size()
        .reset_index(name="count")
    )
    pos_counts["proportion"] = (
        pos_counts["count"]
        / pos_counts.groupby("essay_id")["count"].transform("sum")
    )

    # Save outputs locally.
    surface_counts.to_csv(OUTPUT_DIR / "surface_word_frequencies.csv", index=False)
    lemma_counts.to_csv(OUTPUT_DIR / "lemma_frequencies.csv", index=False)
    content_lemmas.to_csv(OUTPUT_DIR / "content_lemma_frequencies.csv", index=False)
    pos_counts.to_csv(OUTPUT_DIR / "essay_pos_distribution.csv", index=False)

    print("\n=== TOP 20 SURFACE WORDS ===")
    print(surface_counts.head(20).to_string(index=False))

    print("\n=== TOP 20 LEMMAS ===")
    print(lemma_counts.head(20).to_string(index=False))

    print("\n=== TOP 20 CONTENT LEMMAS ===")
    print(content_lemmas.head(20).to_string(index=False))

    print("\n=== GRAMMATICAL COMPOSITION BY ESSAY ===")
    summary = (
        pos_counts.pivot_table(
            index="title",
            columns="pos",
            values="proportion",
            fill_value=0,
        ) * 100
    )

    selected = [
        col for col in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN"]
        if col in summary.columns
    ]
    print(summary[selected].round(1).to_string())

    print("\nSaved four analysis files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
