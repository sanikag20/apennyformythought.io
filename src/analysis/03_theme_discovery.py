
"""03 - Discover recurring themes using TF-IDF and NMF."""
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "essays.csv"
OUT = ROOT / "reports" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

N_TOPICS = 3
TOP_TERMS = 12

# Remove fragments created when contractions are tokenized.
CUSTOM_STOPWORDS = {
    "ve", "ll", "re", "don", "doesn", "didn", "isn",
    "wasn", "wouldn", "couldn", "shouldn", "won",
    "ain", "aren", "weren", "haven", "hasn", "hadn"
}

STOPWORDS = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))


def main():
    df = pd.read_csv(INPUT, keep_default_na=False)
    texts = df["text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        max_features=3000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(texts)

    print("\n=== VECTORIZATION CHECK ===")
    print(f"Essays: {X.shape[0]}")
    print(f"Vocabulary features: {X.shape[1]}")

    if X.shape[1] < 5:
        raise ValueError(
            "Too few vocabulary features. Check preprocessing settings."
        )

    n_topics = min(N_TOPICS, len(texts), X.shape[1])

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=2000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0
    )

    W = model.fit_transform(X)
    terms = vectorizer.get_feature_names_out()

    # Discover the strongest terms associated with each topic.
    topic_rows = []

    for topic_id, weights in enumerate(model.components_, start=1):
        top_indices = weights.argsort()[::-1][:TOP_TERMS]
        top_terms = [terms[i] for i in top_indices]

        topic_rows.append({
            "topic": topic_id,
            "top_terms": ", ".join(top_terms)
        })

    topics_df = pd.DataFrame(topic_rows)
    topics_df.to_csv(OUT / "discovered_topics.csv", index=False)

    # Each essay gets a score for every topic.
    essay_topics = pd.DataFrame(
        W,
        columns=[f"topic_{i}" for i in range(1, n_topics + 1)]
    )

    essay_topics.insert(0, "title", df["title"])

    # Normalize topic weights within each essay.
    row_sums = essay_topics.iloc[:, 1:].sum(axis=1).replace(0, 1)
    essay_topics.iloc[:, 1:] = essay_topics.iloc[:, 1:].div(
        row_sums, axis=0
    )

    essay_topics["dominant_topic"] = (
        essay_topics.iloc[:, 1:n_topics + 1].idxmax(axis=1)
    )

    essay_topics.to_csv(
        OUT / "essay_topic_weights.csv",
        index=False
    )

    print("\n=== DISCOVERED TOPICS ===")

    for row in topic_rows:
        print(f"\nTopic {row['topic']}: {row['top_terms']}")

    print("\n=== TOPIC WEIGHTS BY ESSAY ===")
    print(essay_topics.round(3).to_string(index=False))

    unique_topics = topics_df["top_terms"].nunique()

    print("\n=== MODEL DIAGNOSTICS ===")
    print(f"Distinct topic term lists: {unique_topics}/{n_topics}")

    if unique_topics < n_topics:
        print(
            "WARNING: Some topics have identical top-term lists. "
            "Treat the topic model as unresolved."
        )

    print("\nSaved results in reports/analysis/")


if __name__ == "__main__":
    main()
