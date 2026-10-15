
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESSAYS_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)

    # A compact pretrained model for general-purpose sentence embeddings.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        essays["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarity = cosine_similarity(embeddings)

    # Save the complete pairwise similarity matrix.
    similarity_df = pd.DataFrame(
        similarity,
        index=essays["title"],
        columns=essays["title"],
    )
    similarity_df.to_csv(OUTPUT_DIR / "essay_similarity_matrix.csv")

    # Save embeddings separately for future analysis.
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.insert(0, "title", essays["title"])
    embedding_df.insert(0, "essay_id", essays["essay_id"])
    embedding_df.to_csv(OUTPUT_DIR / "essay_embeddings.csv", index=False)

    # Report the most similar essay pairs, excluding self-similarity.
    pairs = []

    for i in range(len(essays)):
        for j in range(i + 1, len(essays)):
            pairs.append({
                "essay_1": essays.iloc[i]["title"],
                "essay_2": essays.iloc[j]["title"],
                "similarity": float(similarity[i, j]),
            })

    pairs_df = pd.DataFrame(pairs).sort_values(
        "similarity",
        ascending=False,
    )
    pairs_df.to_csv(OUTPUT_DIR / "essay_similarity_pairs.csv", index=False)

    print("\n=== TOP 10 MOST SIMILAR ESSAY PAIRS ===")
    print(pairs_df.head(10).to_string(index=False))

    print("\n=== TOP 5 NEIGHBORS FOR EACH ESSAY ===")

    for i, title in enumerate(essays["title"]):
        neighbors = np.argsort(similarity[i])[::-1]
        neighbors = [j for j in neighbors if j != i][:5]

        print(f"\n{title}")
        for j in neighbors:
            print(
                f"  {essays.iloc[j]['title']}: "
                f"{similarity[i, j]:.3f}"
            )

    print("\nSaved similarity matrix, pairs, and embeddings.")


if __name__ == "__main__":
    main()
