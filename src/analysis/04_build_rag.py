"""04 - Build a local semantic passage index for retrieval-augmented generation."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "essays.csv"
OUT = ROOT / "data" / "processed" / "rag_index"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 180
OVERLAP_WORDS = 35

def split_passages(text):
    words = re.findall(r"\S+", str(text))
    passages = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_WORDS])
        if chunk.strip():
            passages.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return passages

def main():
    df = pd.read_csv(INPUT, keep_default_na=False)
    if not {"title", "text"}.issubset(df.columns):
        raise ValueError("essays.csv must contain title and text columns.")

    records = []
    for idx, row in df.iterrows():
        essay_id = str(row["essay_id"]) if "essay_id" in df.columns else str(idx + 1)
        for passage_no, passage in enumerate(split_passages(row["text"]), start=1):
            records.append({
                "essay_id": essay_id,
                "title": str(row["title"]),
                "passage_no": passage_no,
                "text": passage
            })

    if not records:
        raise ValueError("No passages were created. Check the essay text column.")

    print(f"Essays indexed: {len(df)}")
    print(f"Passages created: {len(records)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [r["text"] for r in records],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    ).astype("float32")

    with (OUT / "passages.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(OUT / "embeddings.npy", embeddings)
    (OUT / "model.txt").write_text(EMBEDDING_MODEL, encoding="utf-8")

    print(f"\nRAG index saved privately under: {OUT}")
    print("Next: run 05_rag_writer.py with a question or writing prompt.")

if __name__ == "__main__":
    main()
