"""01 - Audit the personal essay corpus before modeling."""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "essays.csv"
OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = {"title", "text"}

def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", str(text)))

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Could not find {INPUT}")
    df = pd.read_csv(INPUT, keep_default_na=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    df["word_count"] = df["text"].map(word_count)
    df["character_count"] = df["text"].str.len()
    df["paragraph_count"] = df["text"].map(
        lambda x: len([p for p in re.split(r"\n\s*\n", x.strip()) if p.strip()])
    )
    df["empty_text"] = df["text"].str.strip().eq("")

    print("\n=== CORPUS AUDIT ===")
    print(f"Essays: {len(df)}")
    print(f"Total words: {df['word_count'].sum():,}")
    print(f"Mean words per essay: {df['word_count'].mean():,.1f}")
    print(f"Median words per essay: {df['word_count'].median():,.1f}")
    print(f"Total paragraphs: {df['paragraph_count'].sum():,}")
    print(f"Empty essays: {df['empty_text'].sum()}")
    print("\nEssay lengths:")
    print(df[["title", "word_count", "paragraph_count"]].to_string(index=False))

    if "date" in df.columns:
        nonempty_dates = df["date"].astype(str).str.strip().ne("").sum()
        print(f"\nRows with dates: {nonempty_dates}/{len(df)}")
        if nonempty_dates == 0:
            print("Note: dates are absent, so chronological change cannot yet be studied.")

    output = OUT / "corpus_audit.csv"
    df.drop(columns=["text"]).to_csv(output, index=False)
    print(f"\nSaved audit (metadata only): {output}")

if __name__ == "__main__":
    main()
