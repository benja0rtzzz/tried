"""
Judge client integration test.

Live tests are skipped automatically if the codex CLI is not on PATH.
Covers:
  - Judge prompt builder (AttemptContext rendering)
  - Live codex CLI call and structured output parsing
  - JudgeResult field types and constraints
"""
import shutil

import pytest

from orchestrator.clients.judge_client import JudgeResult, judge
from orchestrator.prompts.judge import AttemptContext, build_user_prompt
from shared.enums import JudgeClassification

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_CODEX_AVAILABLE = pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI not on PATH — skipping live judge tests",
)

# ---------------------------------------------------------------------------
# Prompt builder tests (offline — no API call)
# ---------------------------------------------------------------------------

def test_prompt_includes_pytorch_code(simple_pytorch_code, compile_failure_context):
    prompt = build_user_prompt(simple_pytorch_code, [compile_failure_context])
    assert simple_pytorch_code in prompt


def test_prompt_includes_attempt_block(simple_pytorch_code, compile_failure_context):
    prompt = build_user_prompt(simple_pytorch_code, [compile_failure_context])
    assert "Attempt 0" in prompt
    assert "current" in prompt


def test_prompt_marks_only_last_as_current(simple_pytorch_code, compile_failure_context, correctness_failure_context):
    prior = AttemptContext(**{**compile_failure_context, "fix_suggestion": "Add masks."})
    current = AttemptContext(**{**correctness_failure_context, "attempt_n": 1})
    prompt = build_user_prompt(simple_pytorch_code, [prior, current])
    # attempt 0 is prior — should NOT be marked current
    assert "Attempt 0 (current)" not in prompt
    # attempt 1 is the last — should be marked current
    assert "Attempt 1 (current)" in prompt


def test_prompt_includes_compile_error(simple_pytorch_code, compile_failure_context):
    prompt = build_user_prompt(simple_pytorch_code, [compile_failure_context])
    assert compile_failure_context["compile_error"] in prompt


def test_prompt_includes_correctness_stats(simple_pytorch_code, correctness_failure_context):
    prompt = build_user_prompt(simple_pytorch_code, [correctness_failure_context])
    assert "max_abs_diff" in prompt
    assert "pct_exceeding" in prompt


def test_prompt_includes_prior_fix_suggestion(simple_pytorch_code, compile_failure_context):
    ctx_with_fix = AttemptContext(**{**compile_failure_context, "fix_suggestion": "Add masks."})
    next_ctx = AttemptContext(**{**compile_failure_context, "attempt_n": 1, "fix_suggestion": None})
    prompt = build_user_prompt(simple_pytorch_code, [ctx_with_fix, next_ctx])
    assert "Add masks." in prompt


# ---------------------------------------------------------------------------
# Live judge tests (require codex CLI on PATH)
# ---------------------------------------------------------------------------

@_CODEX_AVAILABLE
def test_judge_compile_failure_returns_valid_result(simple_pytorch_code, compile_failure_context):
    result = judge(simple_pytorch_code, [compile_failure_context])

    assert isinstance(result, JudgeResult)
    assert isinstance(result.classification, JudgeClassification)
    assert isinstance(result.latency_ms, int) and result.latency_ms > 0
    assert isinstance(result.prompt_tokens, int) and result.prompt_tokens > 0
    assert isinstance(result.completion_tokens, int) and result.completion_tokens > 0


@_CODEX_AVAILABLE
def test_judge_compile_failure_suggests_fix(simple_pytorch_code, compile_failure_context):
    """A compile failure should always produce a fix suggestion."""
    result = judge(simple_pytorch_code, [compile_failure_context])
    assert result.fix_suggestion is not None
    assert len(result.fix_suggestion) > 0


@_CODEX_AVAILABLE
def test_judge_correctness_failure_suggests_fix(simple_pytorch_code, correctness_failure_context):
    """A correctness failure should always produce a fix suggestion."""
    result = judge(simple_pytorch_code, [correctness_failure_context])
    assert result.fix_suggestion is not None


@_CODEX_AVAILABLE
def test_judge_classification_is_closed_vocabulary(simple_pytorch_code, compile_failure_context):
    """Classification must be one of the 9 enum values — no free-text leakage."""
    result = judge(simple_pytorch_code, [compile_failure_context])
    assert result.classification in JudgeClassification


@_CODEX_AVAILABLE
def test_judge_multi_attempt_context(simple_pytorch_code, compile_failure_context, correctness_failure_context):
    """Judge should handle multiple prior attempts without error."""
    prior = AttemptContext(**{**compile_failure_context, "fix_suggestion": "Add masks to load and store."})
    current = AttemptContext(**{**correctness_failure_context, "attempt_n": 1})
    result = judge(simple_pytorch_code, [prior, current])

    assert isinstance(result.classification, JudgeClassification)
