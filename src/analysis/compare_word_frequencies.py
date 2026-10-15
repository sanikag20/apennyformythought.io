
from pathlib import Path
from collections import Counter
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESSAYS_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"


def tokenize(text):
    text = text.lower()

    # Expand common contractions before tokenization.
    replacements = {
        "won't": "will not",
        "can't": "cannot",
        "shan't": "shall not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "'s": " is",
    }

    for contraction, replacement in replacements.items():
        text = text.replace(contraction, replacement)

    return re.findall(r"\b[a-z]+\b", text)

def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)
    all_words = []

    for text in essays["text"]:
        all_words.extend(tokenize(text))

    all_counts = Counter(all_words)
    content_words = [
        word for word in all_words
        if word not in ENGLISH_STOP_WORDS
    ]
    content_counts = Counter(content_words)

    print(f"Total tokens: {len(all_words):,}")
    print(f"Unique tokens: {len(all_counts):,}")

    print("\n=== TOP 20 WORDS: INCLUDING STOPWORDS ===")
    print(pd.DataFrame(
        all_counts.most_common(20),
        columns=["word", "count"]
    ).to_string(index=False))

    print("\n=== TOP 20 CONTENT WORDS: STOPWORDS REMOVED ===")
    print(pd.DataFrame(
        content_counts.most_common(20),
        columns=["word", "count"]
    ).to_string(index=False))

if __name__ == "__main__":
    main()
