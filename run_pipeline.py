"""Run the whole writing-analysis pipeline, in order, with one command.

    python run_pipeline.py                  # tests + every stage except the slow LLM ones
    python run_pipeline.py --with-llm       # also run RAG answers and the LLM style comparison
    python run_pipeline.py --from topics    # resume from a stage
    python run_pipeline.py --only rag_index search_check
    python run_pipeline.py --list           # show stages
    python run_pipeline.py --dry-run        # print what would run

Each stage runs as its own Python process. The run stops at the first failing
stage. Timings are appended to reports/analysis/pipeline_runs.csv (private).
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import argparse
import csv
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN_LOG = ROOT / "reports" / "analysis" / "pipeline_runs.csv"
PROMPTS = "experiments/prompts.txt"
TOPIC_LABELS = "experiments/topic_labels.csv"


@dataclass
class Stage:
    name: str
    script: str
    description: str
    args: list = field(default_factory=list)
    llm: bool = False          # slow: runs a local language model


STAGES = [
    Stage("tests", "-m pytest -q", "Unit tests"),
    Stage("build", "src/ingestion/build_dataset.py", "Build dataset: strip Wix headers, check titles, fill dates"),
    Stage("validate", "src/ingestion/validate_dataset.py", "Audit dataset schema, IDs, word counts, duplicates"),
    Stage("audit", "src/analysis/01_corpus_audit.py", "Corpus size and paragraph audit"),
    Stage("explore", "src/analysis/explore_writing.py", "Per-essay length, sentence and vocabulary statistics"),
    Stage("annotate", "src/analysis/annotate_writing.py", "spaCy tokens, lemmas and parts of speech"),
    Stage("frequencies", "src/analysis/analyze_annotated_writing.py", "Word, lemma and part-of-speech frequencies"),
    Stage("tfidf", "src/analysis/tf_idf_analysis.py", "Distinctive TF-IDF terms per essay"),
    Stage("style", "src/analysis/02_style_fingerprint.py", "Style fingerprint per essay"),
    Stage("topics", "src/analysis/03_theme_discovery.py", "NMF topic modeling with stability-based k",
          args=["--labels", TOPIC_LABELS]),
    Stage("charts", "src/analysis/visualise_writing.py", "Charts of length, sentences, vocabulary, top words"),
    Stage("essay_similarity", "src/analysis/semantic_similarity.py", "Essay-to-essay embedding similarity"),
    Stage("passage_similarity", "src/analysis/passage_similarity.py", "Similar passages across essays"),
    Stage("rag_index", "src/analysis/04_build_rag.py", "Split essays into passages and embed them"),
    Stage("search_check", "src/analysis/search.py", "Smoke test: one semantic search",
          args=["feeling like I did not belong", "--k", "3", "--by-essay"]),
    Stage("rag_answers", "src/analysis/05_rag_writer.py", "RAG answers for every prompt (slow)",
          args=["--prompts-file", PROMPTS], llm=True),
    Stage("style_compare", "src/analysis/06_llm_style_comparison.py",
          "Base vs RAG LLM style comparison (slow)", args=["--prompts-file", PROMPTS], llm=True),
]
STAGE_NAMES = [s.name for s in STAGES]


def select_stages(stages, start=None, only=None, with_llm=False, skip_tests=False):
    """Pick which stages to run, keeping pipeline order."""
    names = [s.name for s in stages]
    for name in ([start] if start else []) + list(only or []):
        if name not in names:
            raise ValueError(f"Unknown stage '{name}'. Choose from: {', '.join(names)}")
    if only:
        return [s for s in stages if s.name in set(only)]
    chosen = stages[names.index(start):] if start else list(stages)
    if not with_llm:
        chosen = [s for s in chosen if not s.llm]
    if skip_tests:
        chosen = [s for s in chosen if s.name != "tests"]
    return chosen


def command_for(stage):
    if stage.script.startswith("-m "):
        return [sys.executable, *stage.script.split(), *stage.args]
    return [sys.executable, stage.script, *stage.args]


def log_run(rows):
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RUN_LOG.exists()
    with RUN_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_started", "stage", "status", "seconds"])
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the writing-analysis pipeline.")
    parser.add_argument("--from", dest="start", help="Start at this stage.")
    parser.add_argument("--only", nargs="+", help="Run only these stages.")
    parser.add_argument("--with-llm", action="store_true", help="Include the slow LLM stages.")
    parser.add_argument("--no-tests", action="store_true", help="Skip the unit tests.")
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args()

    if args.list:
        for i, s in enumerate(STAGES, 1):
            print(f"{i:>2}. {s.name:<19} {s.description}")
        return

    stages = select_stages(STAGES, args.start, args.only, args.with_llm, args.no_tests)
    if args.dry_run:
        for s in stages:
            print(" ".join(command_for(s)))
        return

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, total = [], time.perf_counter()
    for i, stage in enumerate(stages, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(stages)}] {stage.name}: {stage.description}\n{'=' * 70}")
        t0 = time.perf_counter()
        result = subprocess.run(command_for(stage), cwd=ROOT)
        seconds = time.perf_counter() - t0
        status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        rows.append({"run_started": started, "stage": stage.name,
                     "status": status, "seconds": round(seconds, 2)})
        if result.returncode != 0:
            log_run(rows)
            print(f"\nStage '{stage.name}' failed. Fix it, then resume with:\n"
                  f"    python run_pipeline.py --from {stage.name}")
            sys.exit(result.returncode)
    log_run(rows)

    print(f"\n{'=' * 70}\nPIPELINE COMPLETE in {time.perf_counter() - total:.1f}s\n{'=' * 70}")
    for r in rows:
        print(f"  {r['stage']:<19} {r['seconds']:>8.1f}s  {r['status']}")
    if not args.with_llm and not args.only:
        print("\nSkipped slow LLM stages. Run them with: python run_pipeline.py --only rag_answers style_compare")


if __name__ == "__main__":
    main()
