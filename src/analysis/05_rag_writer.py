"""05 - Retrieve relevant personal passages and generate a grounded response."""
from pathlib import Path
import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "processed" / "rag_index"
OUT = ROOT / "reports" / "generations"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 4

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

def generate(prompt, context, tokenizer, model):
    messages = [
        {"role": "system", "content": (
            "You are an assistant helping the author explore and develop their ideas. "
            "Use the supplied excerpts as context and optional style examples. Do not copy "
            "sentences verbatim. Do not claim that an idea appears in the archive unless "
            "the excerpts support it. Preserve the author's agency and do not invent personal facts."
        )},
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
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=350, do_sample=True,
            temperature=0.75, top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Question or writing prompt. If omitted, you'll be asked.")
    parser.add_argument("--model", default=GENERATION_MODEL, help="Hugging Face causal LM model ID.")
    args = parser.parse_args()
    prompt = args.prompt or input("What would you like to ask or write about? ").strip()
    if not prompt:
        raise ValueError("Please provide a non-empty prompt.")
    if not (INDEX / "passages.jsonl").exists():
        raise FileNotFoundError("Run 04_build_rag.py first.")

    records, vectors = load_index()
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    matches = retrieve(prompt, records, vectors, encoder)

    context_parts = []
    for record, score in matches:
        context_parts.append(
            f"[{record['title']} | passage {record['passage_no']} | retrieval score {score:.3f}]\n"
            f"{record['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    print(f"\nLoading generation model: {args.model}")
    print("First run downloads model weights; CPU generation may be slow.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    answer = generate(prompt, context, tokenizer, model)
    print("\n=== RAG OUTPUT ===\n")
    print(answer)
    print("\n=== RETRIEVED SOURCES ===")
    for record, score in matches:
        print(f"- {record['title']} (passage {record['passage_no']}, score {score:.3f})")

    output_file = OUT / "rag_generation.txt"
    output_file.write_text(
        f"PROMPT\n{prompt}\n\nRETRIEVED SOURCES\n{context}\n\nGENERATED TEXT\n{answer}\n",
        encoding="utf-8"
    )
    print(f"\nSaved output privately to: {output_file}")

if __name__ == "__main__":
    main()
