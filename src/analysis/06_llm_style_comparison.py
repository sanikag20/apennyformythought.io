"""06 - Compare the same base LLM with and without retrieved personal writing."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
ESSAYS = ROOT / "data" / "processed" / "essays.csv"
INDEX = ROOT / "data" / "processed" / "rag_index"
OUT = ROOT / "reports" / "generations"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def style_features(text):
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    sentences = [s for s in SENTENCE_RE.split(text.strip()) if s.strip()]
    lengths = [len(WORD_RE.findall(s)) for s in sentences]
    n = max(len(words), 1)
    return {
        "avg_sentence_words": float(np.mean(lengths)) if lengths else 0.0,
        "sentence_length_std": float(np.std(lengths)) if lengths else 0.0,
        "lexical_diversity": len(set(lower)) / n,
        "avg_word_characters": float(np.mean([len(w) for w in words])) if words else 0.0,
        "first_person_per_100_words": 100 * sum(w in {"i","me","my","mine","we","us","our"} for w in lower) / n,
        "question_marks_per_1000_chars": 1000 * text.count("?") / max(len(text), 1),
        "exclamation_marks_per_1000_chars": 1000 * text.count("!") / max(len(text), 1),
        "commas_per_1000_chars": 1000 * text.count(",") / max(len(text), 1),
    }

def retrieve(query, records, vectors, encoder, k=4):
    q = encoder.encode([query], normalize_embeddings=True)
    scores = vectors @ q[0]
    ids = np.argsort(scores)[::-1][:k]
    return [(records[i], float(scores[i])) for i in ids]

def run_model(prompt, context, tokenizer, model, use_context):
    if use_context:
        user_text = (
            f"REQUEST:\n{prompt}\n\n"
            f"STYLE AND IDEA EXAMPLES FROM THE AUTHOR'S WRITING:\n{context}\n\n"
            "Write a fresh response to the request. Use broad stylistic tendencies only. "
            "Do not copy phrases or claim the author personally experienced anything not stated."
        )
        system = "You are a careful writing assistant. Preserve the author's voice without imitation by copying."
    else:
        user_text = prompt
        system = "You are a helpful writing assistant. Respond naturally to the user's request."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            repetition_penalty=1.08, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Use the exact same prompt for baseline and RAG.")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()
    prompt = args.prompt or input("Enter one writing prompt for both conditions: ").strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if not (INDEX / "passages.jsonl").exists():
        raise FileNotFoundError("Run 04_build_rag.py first.")

    essays = pd.read_csv(ESSAYS, keep_default_na=False)
    with (INDEX / "passages.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vectors = np.load(INDEX / "embeddings.npy")

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    matches = retrieve(prompt, records, vectors, encoder)
    context = "\n\n".join(
        f"[{r['title']}]\n{r['text']}" for r, _ in matches
    )

    print(f"\nLoading one shared model for both conditions: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    baseline = run_model(prompt, "", tokenizer, model, use_context=False)
    rag = run_model(prompt, context, tokenizer, model, use_context=True)

    real_features = pd.DataFrame([style_features(t) for t in essays["text"]])
    baseline_features = style_features(baseline)
    rag_features = style_features(rag)

    # Standardize distances using variation across the author's essays.
    cols = list(real_features.columns)
    means = real_features.mean()
    stds = real_features.std(ddof=0).replace(0, 1.0)

    rows = []
    for label, values in [("Author corpus mean", means), ("Base LLM", pd.Series(baseline_features)),
                          ("RAG-conditioned LLM", pd.Series(rag_features))]:
        z = (values[cols] - means[cols]) / stds[cols]
        distance = float(np.sqrt(np.mean(np.square(z)))) if label != "Author corpus mean" else 0.0
        row = {"condition": label, "style_distance_from_author_mean": distance}
        row.update({c: float(values[c]) for c in cols})
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "llm_style_comparison_metrics.csv", index=False)

    (OUT / "llm_style_comparison_text.txt").write_text(
        f"PROMPT\n{prompt}\n\nBASE LLM (NO PERSONAL CONTEXT)\n{baseline}\n\n"
        f"RAG-CONDITIONED LLM\n{rag}\n\nRETRIEVED PASSAGES\n{context}\n",
        encoding="utf-8"
    )

    print("\n=== BASE LLM: NO PERSONAL CONTEXT ===\n")
    print(baseline)
    print("\n=== RAG-CONDITIONED LLM ===\n")
    print(rag)
    print("\n=== STYLE METRICS ===")
    print(comparison.round(3).to_string(index=False))
    print("\nA smaller style distance means closer on these measured features only; it does not prove human-perceived voice similarity.")
    print(f"Saved comparison files under: {OUT}")

if __name__ == "__main__":
    main()
