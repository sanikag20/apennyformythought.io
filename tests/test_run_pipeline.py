from pathlib import Path
import sys

import pytest

import run_pipeline as rp

ROOT = Path(rp.__file__).resolve().parent


def names(stages):
    return [s.name for s in stages]


def test_every_stage_script_exists():
    for stage in rp.STAGES:
        if not stage.script.startswith("-m "):
            assert (ROOT / stage.script).is_file(), stage.script


def test_default_run_skips_llm_stages_and_keeps_order():
    chosen = names(rp.select_stages(rp.STAGES))
    assert chosen[0] == "tests" and chosen[1] == "build"
    assert "rag_answers" not in chosen and "style_compare" not in chosen
    assert chosen == [n for n in rp.STAGE_NAMES if n in chosen]


def test_with_llm_includes_slow_stages():
    assert names(rp.select_stages(rp.STAGES, with_llm=True))[-2:] == ["rag_answers", "style_compare"]


def test_from_resumes_at_stage():
    chosen = names(rp.select_stages(rp.STAGES, start="topics"))
    assert chosen[0] == "topics" and "build" not in chosen


def test_only_runs_named_stages_in_pipeline_order():
    chosen = names(rp.select_stages(rp.STAGES, only=["rag_index", "build"]))
    assert chosen == ["build", "rag_index"]


def test_skip_tests():
    assert "tests" not in names(rp.select_stages(rp.STAGES, skip_tests=True))


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        rp.select_stages(rp.STAGES, start="nope")


def test_command_for_uses_current_python():
    tests = rp.STAGES[0]
    assert rp.command_for(tests)[:3] == [sys.executable, "-m", "pytest"]
    topics = next(s for s in rp.STAGES if s.name == "topics")
    assert rp.command_for(topics) == [sys.executable, topics.script, "--labels", rp.TOPIC_LABELS]
