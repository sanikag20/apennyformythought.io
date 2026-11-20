"""05 - Retrieve relevant personal passages and generate a grounded response.

Usage:
    python src/analysis/05_rag_writer.py --prompt "What have I written about belonging?"
    python src/analysis/05_rag_writer.py --prompts-file experiments/prompts.txt
    python src/analysis/05_rag_writer.py            # asks for a prompt

Each run is saved to reports/generations/rag_runs/ (one text file per prompt),
and a row per prompt is appended to reports/generations/rag_runs.csv with the
retrieved sources and timing (retrieval seconds, generation seconds, tokens/sec).
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "processed" / "rag_index"
OUT = ROOT / "reports" / "generations"
RUNS_DIR = OUT / "rag_runs"
RUNS_LOG = OUT / "rag_runs.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 4
MAX_NEW_TOKENS = 350

SYSTEM_PROMPT = (
    "You are an assistant helping the author explore and develop their ideas. "
    "Use the supplied excerpts as context and optional style examples. Do not copy "
    "sentences verbatim. Do not claim that an idea appears in the archive unless "
    "the excerpts support it. Preserve the author's agency and do not invent personal facts."
)

LOG_FIELDS = [
    "timestamp", "prompt", "generation_model", "embedding_model", "top_k",
    "sources", "top_score", "retrieval_seconds", "generation_seconds",
    "new_tokens", "tokens_per_second", "output_file",
]


def slugify(text, max_len=50):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def read_prompts(path):
    """One prompt per line; blank lines and lines starting with # are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_index():
    with (INDEX / "passages.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vectors = np.load(INDEX / "embeddings.npy")
    return records, vectors


def retrieve(query, records, vectors, encoder, k=TOP_K):
    q = encoder.encode([query], normalize_embeddings=True)
    scores = vectors @ q[0]
    chosen = np.argsort(scores)[::-1][:k]
    return [(records[i], float(scores[i])) for i in chosen]


def build_context(matches):
    return "\n\n---\n\n".join(
        f"[{record['title']} | passage {record['passage_no']} | retrieval score {score:.3f}]\n"
        f"{record['text']}"
        for record, score in matches
    )


def generate(prompt, context, tokenizer, model, seed):
    """Return (text, number of new tokens, generation seconds)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"AUTHOR'S REQUEST:\n{prompt}\n\n"
            f"RETRIEVED EXCERPTS FROM THE AUTHOR'S PRIVATE WRITING:\n{context}\n\n"
            "Respond to the request. If asked to draft prose, use broad stylistic cues from "
            "the excerpts without copying them. If evidence is insufficient, say so."
        )}
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    torch.manual_seed(seed)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=0.75, top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id
        )
    seconds = time.perf_counter() - start
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, int(len(new_tokens)), seconds


def append_log(row):
    new_file = not RUNS_LOG.exists()
    with RUNS_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Question or writing prompt.")
    parser.add_argument("--prompts-file", help="Text file with one prompt per line.")
    parser.add_argument("--model", default=GENERATION_MODEL, help="Hugging Face causal LM model ID.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Passages to retrieve (default 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    if args.prompts_file:
        prompts = read_prompts(args.prompts_file)
    else:
        prompt = args.prompt or input("What would you like to ask or write about? ").strip()
        if not prompt:
            raise ValueError("Please provide a non-empty prompt.")
        prompts = [prompt]

    if not (INDEX / "passages.jsonl").exists():
        raise FileNotFoundError("Run 04_build_rag.py first.")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    records, vectors = load_index()
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nLoading generation model: {args.model}")
    print("First run downloads model weights; CPU generation may be slow.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    for n, prompt in enumerate(prompts, start=1):
        print(f"\n[{n}/{len(prompts)}] {prompt}")
        start = time.perf_counter()
        matches = retrieve(prompt, records, vectors, encoder, args.k)
        retrieval_seconds = time.perf_counter() - start
        context = build_context(matches)

        answer, new_tokens, gen_seconds = generate(prompt, context, tokenizer, model, args.seed)
        tps = new_tokens / gen_seconds if gen_seconds > 0 else 0.0

        print("\n=== RAG OUTPUT ===\n")
        print(answer)
        print("\n=== RETRIEVED SOURCES ===")
        for record, score in matches:
            print(f"- {record['title']} (passage {record['passage_no']}, score {score:.3f})")
        print(f"\nRetrieval {retrieval_seconds:.2f}s | generation {gen_seconds:.1f}s | "
              f"{new_tokens} tokens | {tps:.1f} tokens/s")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = RUNS_DIR / f"{stamp}_{slugify(prompt)}.txt"
        output_file.write_text(
            f"PROMPT\n{prompt}\n\nMODEL\n{args.model} (seed {args.seed})\n\n"
            f"RETRIEVED SOURCES\n{context}\n\nGENERATED TEXT\n{answer}\n",
            encoding="utf-8"
        )
        append_log({
            "timestamp": stamp,
            "prompt": prompt,
            "generation_model": args.model,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": args.k,
            "sources": "; ".join(f"{r['title']} p{r['passage_no']}" for r, _ in matches),
            "top_score": round(matches[0][1], 4),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(gen_seconds, 2),
            "new_tokens": new_tokens,
            "tokens_per_second": round(tps, 2),
            "output_file": output_file.name,
        })
        print(f"Saved privately to: {output_file}")

    print(f"\nRun log: {RUNS_LOG}")


if __name__ == "__main__":
    main()
