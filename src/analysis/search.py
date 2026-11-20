"""Semantic search over the writing archive.

Ask the archive a question in plain language and get back the most relevant
passages (or essays), ranked by meaning rather than shared keywords.

Requires the passage index built by 04_build_rag.py.

Usage:
    python src/analysis/search.py "something I wrote about feeling lost"
    python src/analysis/search.py "belonging" --k 3 --by-essay
    python src/analysis/search.py            # interactive mode
"""
from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "processed" / "rag_index"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SNIPPET_CHARS = 220


def load_index(index_dir=INDEX):
    """Load passages, their embeddings, and the embedding model name."""
    passages_file = index_dir / "passages.jsonl"
    if not passages_file.exists():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run 04_build_rag.py first."
        )
    with passages_file.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vectors = np.load(index_dir / "embeddings.npy")
    if len(records) != len(vectors):
        raise ValueError(
            f"Index mismatch: {len(records)} passages but {len(vectors)} embeddings. "
            "Rebuild with 04_build_rag.py."
        )
    model_file = index_dir / "model.txt"
    model_name = (
        model_file.read_text(encoding="utf-8").strip()
        if model_file.exists() else DEFAULT_MODEL
    )
    return records, vectors, model_name


def search(query, records, vectors, encoder, k=5, by_essay=False):
    """Return the top-k results for a query.

    Embeddings are L2-normalized, so the dot product equals cosine similarity.
    With by_essay=True, each essay is scored by its best-matching passage and
    appears at most once.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    q = np.asarray(encoder.encode([query], normalize_embeddings=True))[0]
    scores = np.asarray(vectors) @ q
    order = np.argsort(scores)[::-1]

    results = []
    seen_essays = set()
    for i in order:
        record = records[i]
        if by_essay:
            if record["essay_id"] in seen_essays:
                continue
            seen_essays.add(record["essay_id"])
        results.append({
            "rank": len(results) + 1,
            "essay_id": record["essay_id"],
            "title": record["title"],
            "passage_no": record["passage_no"],
            "score": float(scores[i]),
            "text": record["text"],
        })
        if len(results) == k:
            break
    return results


def print_results(query, results):
    print(f'\nQuery: "{query}"\n')
    for r in results:
        snippet = " ".join(r["text"].split())[:SNIPPET_CHARS]
        print(f"{r['rank']}. {r['title']} (passage {r['passage_no']}) "
              f"score {r['score']:.3f}")
        print(f"   {snippet}...\n")


def main():
    parser = argparse.ArgumentParser(description="Search the writing archive by meaning.")
    parser.add_argument("query", nargs="?", help="Question or phrase. Omit for interactive mode.")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default 5).")
    parser.add_argument("--by-essay", action="store_true",
                        help="Return each essay at most once, scored by its best passage.")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    records, vectors, model_name = load_index()
    encoder = SentenceTransformer(model_name)
    print(f"Loaded {len(records)} passages from "
          f"{len({r['essay_id'] for r in records})} essays.")

    if args.query:
        print_results(args.query, search(args.query, records, vectors, encoder,
                                         args.k, args.by_essay))
        return

    print("Type a question (blank line to quit).")
    while True:
        query = input("\nSearch: ").strip()
        if not query:
            break
        print_results(query, search(query, records, vectors, encoder,
                                    args.k, args.by_essay))


if __name__ == "__main__":
    main()
