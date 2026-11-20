"""06 - Compare the same base LLM with and without retrieved personal writing.

For each prompt, the same model writes two responses: one with no personal
context (baseline) and one conditioned on retrieved passages (RAG). Both are
scored on interpretable style features and compared with the author's own
essays, using z-scores relative to variation across the author's essays.

Usage:
    python src/analysis/06_llm_style_comparison.py --prompt "Write about coming home"
    python src/analysis/06_llm_style_comparison.py --prompts-file experiments/prompts.txt

Outputs (private, in reports/generations/):
    llm_style_comparison_metrics.csv   one row per prompt x condition
    llm_style_comparison_summary.csv   mean style distance per condition
    style_comparison_runs/*.txt        the generated texts for each prompt
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ESSAYS = ROOT / "data" / "processed" / "essays.csv"
INDEX = ROOT / "data" / "processed" / "rag_index"
OUT = ROOT / "reports" / "generations"
TEXT_DIR = OUT / "style_comparison_runs"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORD_RE = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}


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
        "first_person_per_100_words": 100 * sum(w in FIRST_PERSON for w in lower) / n,
        "question_marks_per_1000_chars": 1000 * text.count("?") / max(len(text), 1),
        "exclamation_marks_per_1000_chars": 1000 * text.count("!") / max(len(text), 1),
        "commas_per_1000_chars": 1000 * text.count(",") / max(len(text), 1),
    }


def style_distance(features, means, stds):
    """Root-mean-square z-score distance from the author's mean style."""
    cols = list(means.index)
    z = (pd.Series(features)[cols] - means[cols]) / stds[cols]
    return float(np.sqrt(np.mean(np.square(z.astype(float)))))


def read_prompts(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def slugify(text, max_len=50):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "prompt"


def retrieve(query, records, vectors, encoder, k=4):
    q = encoder.encode([query], normalize_embeddings=True)
    scores = vectors @ q[0]
    ids = np.argsort(scores)[::-1][:k]
    return [(records[i], float(scores[i])) for i in ids]


def run_model(prompt, context, tokenizer, model, use_context):
    import torch

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


def summarize(metrics):
    """Mean and spread of style distance per condition, plus how often RAG was closer."""
    gen = metrics[metrics["condition"] != "Author corpus mean"]
    summary = (
        gen.groupby("condition")["style_distance_from_author_mean"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    pivot = gen.pivot(index="prompt", columns="condition",
                      values="style_distance_from_author_mean")
    if {"Base LLM", "RAG-conditioned LLM"}.issubset(pivot.columns):
        closer = int((pivot["RAG-conditioned LLM"] < pivot["Base LLM"]).sum())
        summary["rag_closer_on_prompts"] = f"{closer}/{len(pivot)}"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="One prompt, used for both conditions.")
    parser.add_argument("--prompts-file", help="Text file with one prompt per line.")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    if args.prompts_file:
        prompts = read_prompts(args.prompts_file)
    else:
        prompt = args.prompt or input("Enter one writing prompt for both conditions: ").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        prompts = [prompt]

    if not (INDEX / "passages.jsonl").exists():
        raise FileNotFoundError("Run 04_build_rag.py first.")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    essays = pd.read_csv(ESSAYS, keep_default_na=False)
    with (INDEX / "passages.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    vectors = np.load(INDEX / "embeddings.npy")

    real_features = pd.DataFrame([style_features(t) for t in essays["text"]])
    means = real_features.mean()
    stds = real_features.std(ddof=0).replace(0, 1.0)

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"\nLoading one shared model for both conditions: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    rows = [{"prompt": "(all)", "condition": "Author corpus mean",
             "style_distance_from_author_mean": 0.0, **means.to_dict()}]

    for n, prompt in enumerate(prompts, start=1):
        print(f"\n[{n}/{len(prompts)}] {prompt}")
        matches = retrieve(prompt, records, vectors, encoder)
        context = "\n\n".join(f"[{r['title']}]\n{r['text']}" for r, _ in matches)

        baseline = run_model(prompt, "", tokenizer, model, use_context=False)
        rag = run_model(prompt, context, tokenizer, model, use_context=True)

        for label, text in [("Base LLM", baseline), ("RAG-conditioned LLM", rag)]:
            feats = style_features(text)
            rows.append({"prompt": prompt, "condition": label,
                         "style_distance_from_author_mean": style_distance(feats, means, stds),
                         **feats})

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (TEXT_DIR / f"{stamp}_{slugify(prompt)}.txt").write_text(
            f"PROMPT\n{prompt}\n\nBASE LLM (NO PERSONAL CONTEXT)\n{baseline}\n\n"
            f"RAG-CONDITIONED LLM\n{rag}\n\nRETRIEVED PASSAGES\n{context}\n",
            encoding="utf-8"
        )
        base_d, rag_d = rows[-2]["style_distance_from_author_mean"], rows[-1]["style_distance_from_author_mean"]
        print(f"Style distance from your writing: base {base_d:.2f} | RAG {rag_d:.2f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "llm_style_comparison_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT / "llm_style_comparison_summary.csv", index=False)

    print("\n=== SUMMARY: STYLE DISTANCE FROM YOUR ESSAYS (lower = closer) ===")
    print(summary.round(3).to_string(index=False))
    print("\nA smaller style distance means closer on these measured features only; "
          "it does not prove human-perceived voice similarity.")
    print(f"Saved metrics, summary and texts under: {OUT}")


if __name__ == "__main__":
    main()
