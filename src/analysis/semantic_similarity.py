
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESSAYS_FILE = PROJECT_ROOT / "data" / "processed" / "essays.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_OVERLAP = 64


def split_into_chunks(text, tokenizer, max_tokens):
    """Split text into overlapping token chunks without truncating it."""
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if not token_ids:
        return []

    step = max_tokens - CHUNK_OVERLAP
    chunks = []

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))

        if start + max_tokens >= len(token_ids):
            break

    return chunks


def normalize(vector):
    """Return a unit-length vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)

    if essays.empty:
        raise ValueError("The essay dataset is empty.")

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    # Reserve room for special tokens added by the model.
    max_chunk_tokens = model.max_seq_length - 2

    if CHUNK_OVERLAP >= max_chunk_tokens:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    texts = essays["text"].astype(str).tolist()

    # Baseline: explicitly encode only the opening portion of each essay.
    opening_texts = [
        tokenizer.decode(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"][:max_chunk_tokens]
        )
        for text in texts
    ]

    opening_embeddings = model.encode(
        opening_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Improved method: encode every chunk, then aggregate by essay.
    essay_embeddings = []
    chunk_counts = []

    for title, text in zip(essays["title"], texts):
        chunks = split_into_chunks(
            text,
            tokenizer,
            max_chunk_tokens,
        )

        if not chunks:
            raise ValueError(f"Essay has no usable text: {title}")

        chunk_embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Weight each chunk by its token count before averaging.
        chunk_weights = np.array([
            len(tokenizer(
                chunk,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"])
            for chunk in chunks
        ], dtype=float)

        average_embedding = np.average(
            chunk_embeddings,
            axis=0,
            weights=chunk_weights,
        )

        essay_embeddings.append(normalize(average_embedding))
        chunk_counts.append(len(chunks))

    opening_similarity = cosine_similarity(opening_embeddings)
    chunked_embeddings = np.vstack(essay_embeddings)
    chunked_similarity = cosine_similarity(chunked_embeddings)

    titles = essays["title"].tolist()

    # Save new outputs separately so the original results remain available.
    pd.DataFrame(
        chunked_similarity,
        index=titles,
        columns=titles,
    ).to_csv(OUTPUT_DIR / "essay_similarity_matrix_chunked.csv")

    embedding_df = pd.DataFrame(chunked_embeddings)
    embedding_df.insert(0, "title", titles)
    embedding_df.insert(0, "essay_id", essays["essay_id"])
    embedding_df["chunk_count"] = chunk_counts
    embedding_df.to_csv(
        OUTPUT_DIR / "essay_embeddings_chunked.csv",
        index=False,
    )

    pairs = []

    for i in range(len(essays)):
        for j in range(i + 1, len(essays)):
            pairs.append({
                "essay_1": titles[i],
                "essay_2": titles[j],
                "opening_similarity": float(opening_similarity[i, j]),
                "chunked_similarity": float(chunked_similarity[i, j]),
                "score_change": float(
                    chunked_similarity[i, j] - opening_similarity[i, j]
                ),
            })

    pairs_df = pd.DataFrame(pairs).sort_values(
        "chunked_similarity",
        ascending=False,
    )

    pairs_df.to_csv(
        OUTPUT_DIR / "essay_similarity_pairs_comparison.csv",
        index=False,
    )

    print("\n=== CHUNK COUNTS ===")
    for title, count in zip(titles, chunk_counts):
        print(f"{title}: {count} chunks")

    print("\n=== TOP 10 PAIRS BY CHUNKED SIMILARITY ===")
    print(
        pairs_df[
            ["essay_1", "essay_2", "opening_similarity",
             "chunked_similarity", "score_change"]
        ].head(10).round(3).to_string(index=False)
    )

    print("\n=== LARGEST CHANGES IN SIMILARITY ===")
    changed = pairs_df.reindex(
        pairs_df["score_change"].abs().sort_values(ascending=False).index
    )
    print(
        changed[
            ["essay_1", "essay_2", "opening_similarity",
             "chunked_similarity", "score_change"]
        ].head(10).round(3).to_string(index=False)
    )

    print("\nSaved new chunked similarity outputs.")


if __name__ == "__main__":
    main()
