
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ESSAYS_FILE = DATA_DIR / "essays.csv"
STATS_FILE = DATA_DIR / "essay_statistics.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def save_bar_chart(data, x_col, y_col, title, xlabel, ylabel, filename):
    data = data.sort_values(x_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data[y_col], data[x_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)
    stats = pd.read_csv(STATS_FILE)

    # Chart 1: essay length
    save_bar_chart(
        stats,
        "word_count",
        "title",
        "Essay Length",
        "Number of words",
        "Essay",
        "essay_lengths.png",
    )

    # Chart 2: average sentence length
    save_bar_chart(
        stats,
        "avg_sentence_length",
        "title",
        "Average Sentence Length",
        "Average words per sentence",
        "Essay",
        "sentence_lengths.png",
    )

    # Chart 3: vocabulary ratio
    save_bar_chart(
        stats,
        "vocabulary_ratio",
        "title",
        "Basic Vocabulary Ratio",
        "Unique words / total words",
        "Essay",
        "vocabulary_ratios.png",
    )

    # Word frequency: use standard English stopwords
    # to reduce the influence of common words such as "the" and "and".
    vectorizer = CountVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
        max_features=30,
    )

    matrix = vectorizer.fit_transform(essays["text"])
    frequencies = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    top_words = (
        pd.DataFrame({"word": words, "count": frequencies})
        .sort_values("count", ascending=False)
        .head(20)
    )

    print("\n=== TOP 20 WORDS (AFTER BASIC STOPWORD REMOVAL) ===")
    print(top_words.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 7))
    top_words_sorted = top_words.sort_values("count", ascending=True)
    ax.barh(top_words_sorted["word"], top_words_sorted["count"])
    ax.set_title("Most Frequent Words Across the Archive")
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Word")
    fig.tight_layout()

    output_path = FIGURES_DIR / "top_words.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
