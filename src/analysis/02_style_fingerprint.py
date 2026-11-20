"""02 - Measure interpretable stylistic features for each essay."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "essays.csv"
OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
CONTRACTIONS = re.compile(r"\b\w+['’](?:t|s|re|ve|ll|d|m)\b", re.I)

def features(title, text):
    text = str(text)
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    sentences = [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]
    sentence_lengths = [len(WORD_RE.findall(s)) for s in sentences]
    n = max(len(words), 1)
    chars = max(len(text), 1)
    return {
        "title": title,
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]),
        "avg_sentence_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "sentence_length_std": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
        "lexical_diversity": len(set(lower)) / n,
        "hapax_ratio": sum(v == 1 for v in pd.Series(lower).value_counts()) / n if lower else 0.0,
        "avg_word_characters": float(np.mean([len(w) for w in words])) if words else 0.0,
        "first_person_per_100_words": 100 * sum(w in FIRST_PERSON for w in lower) / n,
        "second_person_per_100_words": 100 * sum(w in SECOND_PERSON for w in lower) / n,
        "question_marks_per_1000_chars": 1000 * text.count("?") / chars,
        "exclamation_marks_per_1000_chars": 1000 * text.count("!") / chars,
        "commas_per_1000_chars": 1000 * text.count(",") / chars,
        "parentheses_per_1000_chars": 1000 * (text.count("(") + text.count(")")) / chars,
        "contractions_per_100_words": 100 * len(CONTRACTIONS.findall(text)) / n,
    }

def main():
    df = pd.read_csv(INPUT, keep_default_na=False)
    if not {"title", "text"}.issubset(df.columns):
        raise ValueError("essays.csv must contain title and text columns.")
    result = pd.DataFrame([features(r["title"], r["text"]) for _, r in df.iterrows()])
    result.to_csv(OUT / "style_fingerprint_by_essay.csv", index=False)

    numeric = result.select_dtypes(include="number")
    summary = pd.DataFrame({
        "feature": numeric.columns,
        "corpus_mean": numeric.mean().values,
        "corpus_std": numeric.std(ddof=0).values,
        "minimum": numeric.min().values,
        "maximum": numeric.max().values,
    })
    summary.to_csv(OUT / "style_fingerprint_summary.csv", index=False)

    print("\n=== STYLE FINGERPRINT: CORPUS AVERAGES ===")
    print(summary[["feature", "corpus_mean", "corpus_std"]].round(3).to_string(index=False))
    print("\nSaved per-essay and summary CSVs in reports/analysis/")

if __name__ == "__main__":
    main()
