
from pathlib import Path

import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESSAYS_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "essay_tokens.csv"
)

# Disable components we do not need for this step.
# Keep the tagger, parser, and lemmatizer.
nlp = spacy.load("en_core_web_sm")


def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)
    records = []

    for _, essay in essays.iterrows():
        doc = nlp(essay["text"])

        for sentence_id, sentence in enumerate(doc.sents):
            for token in sentence:
                if token.is_space:
                    continue

                records.append({
                    "essay_id": essay["essay_id"],
                    "title": essay["title"],
                    "sentence_id": sentence_id,
                    "token": token.text,
                    "lowercase": token.lower_,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "is_stop": token.is_stop,
                    "is_punctuation": token.is_punct,
                    "is_alpha": token.is_alpha,
                })

        print(f"Annotated: {essay['title']}")

    tokens = pd.DataFrame(records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_csv(OUTPUT_FILE, index=False)

    print("\n=== ANNOTATION SUMMARY ===")
    print(f"Essays processed: {essays['essay_id'].nunique()}")
    print(f"Tokens annotated: {len(tokens):,}")
    print(f"Unique surface forms: {tokens['lowercase'].nunique():,}")
    print(f"Unique lemmas: {tokens['lemma'].nunique():,}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\n=== SAMPLE TOKENS ===")
    print(
        tokens[["token", "lemma", "pos", "is_stop"]]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
