
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
TOP_PAIRS = 20


def split_into_chunks(text, tokenizer, max_tokens):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    step = max_tokens - CHUNK_OVERLAP
    chunks = []

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + max_tokens]
        if not chunk_ids:
            break

        chunks.append(tokenizer.decode(chunk_ids))

        if start + max_tokens >= len(token_ids):
            break

    return chunks


def main():
    essays = pd.read_csv(ESSAYS_FILE, keep_default_na=False)

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length - 2

    passages = []

    for _, essay in essays.iterrows():
        chunks = split_into_chunks(
            str(essay["text"]),
            tokenizer,
            max_tokens,
        )

        for chunk_index, chunk in enumerate(chunks):
            passages.append({
                "essay_id": essay["essay_id"],
                "title": essay["title"],
                "chunk_index": chunk_index,
                "text": chunk,
            })

    print(f"Essays: {len(essays)}")
    print(f"Passages: {len(passages)}")

    embeddings = model.encode(
        [p["text"] for p in passages],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    similarity = cosine_similarity(embeddings)
    pairs = []

    for i, first in enumerate(passages):
        for j in range(i + 1, len(passages)):
            second = passages[j]

            # Only compare passages from different essays.
            if first["essay_id"] == second["essay_id"]:
                continue

            pairs.append({
                "essay_1": first["title"],
                "passage_1": first["chunk_index"] + 1,
                "text_1": first["text"],
                "essay_2": second["title"],
                "passage_2": second["chunk_index"] + 1,
                "text_2": second["text"],
                "similarity": float(similarity[i, j]),
            })

    pairs_df = pd.DataFrame(pairs).sort_values(
        "similarity",
        ascending=False,
    )

    output_file = OUTPUT_DIR / "passage_similarity_pairs.csv"
    pairs_df.to_csv(output_file, index=False)

    print(f"\n=== TOP {TOP_PAIRS} CROSS-ESSAY PASSAGE MATCHES ===")

    for rank, (_, pair) in enumerate(
        pairs_df.head(TOP_PAIRS).iterrows(),
        start=1,
    ):
        print(f"\n{rank}. Similarity: {pair['similarity']:.3f}")
        print(
            f"{pair['essay_1']} (passage {pair['passage_1']}):"
        )
        print(pair["text_1"][:450].replace("\n", " "))
        print(
            f"{pair['essay_2']} (passage {pair['passage_2']}):"
        )
        print(pair["text_2"][:450].replace("\n", " "))

    print(f"\nSaved all passage matches to: {output_file}")


if __name__ == "__main__":
    main()
