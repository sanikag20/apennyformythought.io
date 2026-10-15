
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESSAYS_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)

    # Analyze each essay as a separate document.
    # Use spaCy lemmas, excluding punctuation and stopwords.
    tokens = pd.read_csv(
        OUTPUT_DIR / "essay_tokens.csv",
        keep_default_na=False,
    )

    content = tokens[
        tokens["is_alpha"].astype(str).str.lower().eq("true")
        & tokens["is_stop"].astype(str).str.lower().eq("false")
        & tokens["pos"].isin(
            ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]
        )
    ].copy()

    content["lemma"] = content["lemma"].str.lower()

    documents = (
        content.groupby("essay_id")["lemma"]
        .apply(lambda words: " ".join(words))
        .rename("document")
        .reset_index()
    )

    # Keep the original essay ordering and titles.
    documents = essays[
        ["essay_id", "title"]
    ].merge(documents, on="essay_id", how="left")

    documents["document"] = documents["document"].fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(documents["document"])
    terms = vectorizer.get_feature_names_out()

    rows = []

    for i, essay in documents.iterrows():
        scores = matrix[i].toarray().ravel()
        top_indices = scores.argsort()[::-1][:10]

        for rank, term_idx in enumerate(top_indices, start=1):
            if scores[term_idx] <= 0:
                continue

            rows.append({
                "essay_id": essay["essay_id"],
                "title": essay["title"],
                "rank": rank,
                "term": terms[term_idx],
                "tfidf_score": scores[term_idx],
            })

    results = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "essay_tfidf_terms.csv"
    results.to_csv(output_path, index=False)

    print("=== DISTINCTIVE TERMS BY ESSAY ===")

    for title, group in results.groupby("title", sort=False):
        terms_display = ", ".join(group["term"].head(10))
        print(f"\n{title}\n  {terms_display}")

    print(f"\nSaved TF-IDF results to: {output_path}")


if __name__ == "__main__":
    main()
