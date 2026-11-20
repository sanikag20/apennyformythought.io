"""03 - Discover recurring themes with TF-IDF and NMF topic modeling.

With a small corpus (11 essays), topic models are exploratory and easy to
over-read, so this script:
  * removes conversational filler ("guys", "okay", "literally") as well as
    standard English stopwords, so topics reflect content rather than voice,
  * chooses the number of topics by stability: for each k it refits NMF from
    many random starts and measures how consistently the same top terms
    reappear (matched Jaccard overlap). It picks the smallest k whose stability
    is within STABILITY_TOLERANCE of the best,
  * reports the most representative essays for each topic.

Usage:
    python src/analysis/03_theme_discovery.py                 # choose k automatically
    python src/analysis/03_theme_discovery.py --k 4           # force a k
    python src/analysis/03_theme_discovery.py --labels experiments/topic_labels.csv

Outputs (private, in reports/analysis/):
    topic_k_selection.csv      stability and reconstruction error for each k
    discovered_topics.csv      top terms and representative essays per topic
    essay_topic_weights.csv    normalized topic weights and dominant topic per essay
"""
from pathlib import Path
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "essays.csv"
OUT = ROOT / "reports" / "analysis"

K_RANGE = range(2, 7)
STABILITY_SEEDS = 20
STABILITY_TERMS = 10
STABILITY_TOLERANCE = 0.02
TOP_TERMS = 12
TOP_ESSAYS = 3
MIN_ESSAY_WEIGHT = 0.05   # list an essay under a topic only if it carries real weight
RANDOM_STATE = 42

# Words that carry the author's conversational voice but not a theme.
# Contraction fragments are covered too ("couldn't" tokenizes to "couldn").
FILLER_WORDS = {
    "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn",
    "shouldn", "wasn", "weren", "won", "wouldn",
    "yes", "yeah", "okay", "hey", "heyo", "guys", "really", "just", "like", "lot",
    "lots", "thing", "things", "bit", "gonna", "wanna", "toodles", "lowkey",
    "literally", "probably", "truly", "sure", "let", "maybe", "actually", "basically",
    "pretty", "kind", "sort", "stuff", "way", "got", "get", "gets", "getting", "know",
    "think", "thought", "feel", "feels", "said", "say", "says", "going", "went",
    "make", "made", "want", "wanted", "does", "did", "doing", "time", "times", "day",
    "days", "year", "years", "today", "right", "good", "great", "bad", "new", "old",
    "little", "big", "long", "wow", "haha", "lol", "min", "read", "until", "else",
    "someone", "something", "everything", "anything", "nothing", "everyone",
    "people", "person", "mean", "means", "tell", "told", "come", "came", "look",
    "looks", "looking", "use", "used", "need", "needs", "thank", "thanks", "hope",
    "wonder", "shit", "damn", "fuck", "bitch",
}
STOPWORDS = sorted(ENGLISH_STOP_WORDS | FILLER_WORDS)


def vectorize(texts):
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        min_df=2,          # a theme should appear in at least two essays
        max_df=0.7,        # words in most essays describe the writer, not a theme
        sublinear_tf=True,
    )
    texts = [str(t).replace("’", "'") for t in texts]
    X = vectorizer.fit_transform(texts)
    return X, vectorizer.get_feature_names_out()


def top_term_indices(components, n):
    return [weights.argsort()[::-1][:n] for weights in components]


def topic_stability(X, k, seeds=STABILITY_SEEDS, n_terms=STABILITY_TERMS):
    """Mean matched Jaccard overlap of top terms across NMF runs from random starts.

    1.0 means every run finds the same topics; values near 0 mean the topics are
    artifacts of the starting point.
    """
    runs = []
    for seed in range(seeds):
        model = NMF(n_components=k, init="random", random_state=seed, max_iter=3000)
        model.fit(X)
        runs.append([set(idx) for idx in top_term_indices(model.components_, n_terms)])

    scores = []
    for a, b in itertools.combinations(runs, 2):
        overlap = np.array([[len(x & y) / len(x | y) for y in b] for x in a])
        rows, cols = linear_sum_assignment(-overlap)
        scores.append(overlap[rows, cols].mean())
    return float(np.mean(scores))


def choose_k(selection, tolerance=STABILITY_TOLERANCE):
    """Smallest k whose stability is within tolerance of the most stable k."""
    best = selection["stability"].max()
    eligible = selection[selection["stability"] >= best - tolerance]
    return int(eligible["k"].min())


def fit_topics(X, k):
    model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=3000)
    W = model.fit_transform(X)
    return model, W


def load_labels(path, k):
    if not path:
        return {}
    labels = pd.read_csv(path)
    if not {"topic", "label"}.issubset(labels.columns):
        raise ValueError("Labels file needs 'topic' and 'label' columns.")
    mapping = dict(zip(labels["topic"].astype(int), labels["label"].astype(str)))
    unknown = set(mapping) - set(range(1, k + 1))
    if unknown:
        raise ValueError(f"Labels refer to topics that do not exist for k={k}: {sorted(unknown)}")
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, help="Number of topics. Default: chosen by stability.")
    parser.add_argument("--labels", help="Optional CSV with columns topic,label.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, keep_default_na=False)
    X, terms = vectorize(df["text"])

    print("\n=== VECTORIZATION ===")
    print(f"Essays: {X.shape[0]} | vocabulary after filtering: {X.shape[1]}")
    if X.shape[1] < 10:
        raise ValueError("Too few vocabulary features. Check preprocessing settings.")

    k_values = [k for k in K_RANGE if k < X.shape[0]]
    print(f"\n=== CHOOSING THE NUMBER OF TOPICS ({STABILITY_SEEDS} random starts per k) ===")
    rows = []
    for k in k_values:
        model, _ = fit_topics(X, k)
        rows.append({"k": k, "stability": topic_stability(X, k),
                     "reconstruction_error": float(model.reconstruction_err_)})
        print(f"k={k}: stability {rows[-1]['stability']:.3f}, "
              f"reconstruction error {rows[-1]['reconstruction_error']:.3f}")
    selection = pd.DataFrame(rows)
    selection.to_csv(OUT / "topic_k_selection.csv", index=False)

    k = args.k or choose_k(selection)
    print(f"\nUsing k={k}" + (" (set by --k)" if args.k else " (most stable, smallest)"))
    labels = load_labels(args.labels, k)

    model, W = fit_topics(X, k)
    weights = W / np.where(W.sum(axis=1, keepdims=True) == 0, 1, W.sum(axis=1, keepdims=True))

    topic_rows = []
    for t, idx in enumerate(top_term_indices(model.components_, TOP_TERMS), start=1):
        order = np.argsort(weights[:, t - 1])[::-1]
        essays = [f"{df['title'][j]} ({weights[j, t - 1]:.2f})"
                  for j in order[:TOP_ESSAYS] if weights[j, t - 1] >= MIN_ESSAY_WEIGHT]
        topic_rows.append({
            "topic": t,
            "label": labels.get(t, ""),
            "top_terms": ", ".join(terms[i] for i in idx),
            "top_essays": "; ".join(essays),
        })
    topics = pd.DataFrame(topic_rows)
    topics.to_csv(OUT / "discovered_topics.csv", index=False)

    essay_topics = pd.DataFrame(weights, columns=[f"topic_{i}" for i in range(1, k + 1)])
    essay_topics.insert(0, "title", df["title"])
    if "published_date" in df.columns:
        essay_topics.insert(1, "published_date", df["published_date"])
    dominant = weights.argmax(axis=1) + 1
    essay_topics["dominant_topic"] = dominant
    essay_topics["dominant_label"] = [labels.get(int(t), "") for t in dominant]
    essay_topics["dominant_weight"] = weights.max(axis=1)
    essay_topics.to_csv(OUT / "essay_topic_weights.csv", index=False)

    print("\n=== DISCOVERED TOPICS ===")
    for row in topic_rows:
        name = f" - {row['label']}" if row["label"] else ""
        print(f"\nTopic {row['topic']}{name}\n  terms:  {row['top_terms']}\n  essays: {row['top_essays']}")

    print("\nNote: with a small corpus these topics are exploratory; read them alongside the essays.")
    print(f"Saved results in {OUT}")


if __name__ == "__main__":
    main()
