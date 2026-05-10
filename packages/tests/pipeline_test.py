"""
End-to-end pipeline tests: real generator (Ollama) + real judge (codex CLI) + mock verification.

The mock server (mock_server.py) runs locally at localhost:8765 and returns
scripted compile/run/benchmark responses — the actual generated Triton code is
ignored. This lets us test the full agent loop logic (retries, judge feedback,
dataset writes) without needing the Lenovo CUDA server.

Prerequisites:
  - codex CLI on PATH with profile gpt-5-3-codex configured
  - Ollama running with qwen2.5-coder:14b pulled
  - uv sync --all-packages done

Run:
  TRIED_ROLE=orchestrator uv run pytest tests/pipeline_test.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

import orchestrator.clients.verification_client as _vc_module
from orchestrator.dataset.agent import run_job
from orchestrator.clients.verification_client import VerificationClient
from shared.enums import CompileStatus, CorrectnessStatus, JudgeClassification, Split
from shared.logging import get_logger
from shared.models import CorpusRecord, DatasetRow

_log = get_logger("tests.pipeline")

# Speed up benchmark polling — the mock returns done immediately, no need to wait 5s.
_vc_module._POLL_INTERVAL = 0.2

_MOCK_URL  = "http://127.0.0.1:8765"
_TEST_KEY  = "test-key"
_DATA_DIR  = Path(__file__).parent.parent.parent / ".testdata"

# ---------------------------------------------------------------------------
# Shared response blocks
# ---------------------------------------------------------------------------

_ZERO_STATS: dict[str, Any] = {
    "max_abs_diff": 0.0,
    "max_rel_diff": 0.0,
    "mean_abs_diff": 0.0,
    "n_elements_exceeding_tol": 0,
    "pct_elements_exceeding_tol": 0.0,
}

_BAD_STATS: dict[str, Any] = {
    "max_abs_diff": 0.47,
    "max_rel_diff": 0.33,
    "mean_abs_diff": 0.19,
    "n_elements_exceeding_tol": 8192,
    "pct_elements_exceeding_tol": 3.1,
}

_COMPILE_OK:  dict[str, Any] = {"status": "success", "error_message": None}
_COMPILE_ERR: dict[str, Any] = {
    "status": "failed",
    "error_message": (
        "triton.compiler.errors.CompilationError: "
        "invalid use of tl.load — pointer arithmetic produced wrong shape"
    ),
}

_RUN_PASS: dict[str, Any] = {
    "correctness_status": "passed",
    "tolerance_policy_used": "default_fp32",
    "vs_eager":    _ZERO_STATS,
    "vs_inductor": _ZERO_STATS,
    "error_message": None,
}

_RUN_FAIL: dict[str, Any] = {
    "correctness_status": "failed",
    "tolerance_policy_used": "default_fp32",
    "vs_eager":    _BAD_STATS,
    "vs_inductor": _BAD_STATS,
    "error_message": None,
}

_BENCHMARK_GOOD: dict[str, Any] = {
    "triton_ms":           0.48,
    "eager_ms":            1.05,
    "inductor_ms":         0.82,
    "speedup_vs_eager":    2.19,
    "speedup_vs_inductor": 1.71,
    "triton_std_ms":       0.01,
    "eager_std_ms":        0.02,
    "inductor_std_ms":     0.015,
}

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

# easy: attempt 0 compiles and runs correctly → success in one shot
_EASY_SCRIPTS = [
    {"compile": _COMPILE_OK, "run": _RUN_PASS, "benchmark": _BENCHMARK_GOOD},
]

# medium: attempt 0 compiles but fails correctness, attempt 1 passes
_MEDIUM_SCRIPTS = [
    {"compile": _COMPILE_OK, "run": _RUN_FAIL},
    {"compile": _COMPILE_OK, "run": _RUN_PASS, "benchmark": _BENCHMARK_GOOD},
]

# hard: attempt 0 compile fails, attempt 1 run fails, attempt 2 passes
_HARD_SCRIPTS = [
    {"compile": _COMPILE_ERR},
    {"compile": _COMPILE_OK, "run": _RUN_FAIL},
    {"compile": _COMPILE_OK, "run": _RUN_PASS, "benchmark": _BENCHMARK_GOOD},
]

# ---------------------------------------------------------------------------
# Corpus fixtures  (real records from the training corpus)
# ---------------------------------------------------------------------------

_EASY_RECORD = CorpusRecord(
    example_id="test-easy-clamp-square",
    split=Split.TRAIN,
    origin="curated/train/elementwise_clamp_square_fp32",
    op_category="elementwise_chain",
    difficulty=None,
    pytorch_code=(
        "import torch\n\n"
        "def op(x: torch.Tensor) -> torch.Tensor:\n"
        "    y = torch.clamp(x, -3.0, 3.0)\n"
        "    return y * y + 0.25 * y\n"
    ),
    input_shapes=[[1048576]],
    input_dtypes=["float32"],
    rng_seed=42,
)

_MEDIUM_RECORD = CorpusRecord(
    example_id="test-medium-gelu-residual",
    split=Split.TRAIN,
    origin="curated/train/elementwise_gelu_residual_fp32",
    op_category="elementwise_chain",
    difficulty=None,
    pytorch_code=(
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        "def op(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:\n"
        "    return F.gelu(x) + residual * 0.5\n"
    ),
    input_shapes=[[32, 128, 768], [32, 128, 768]],
    input_dtypes=["float32", "float32"],
    rng_seed=42,
)

_HARD_RECORD = CorpusRecord(
    example_id="test-hard-rope-pair",
    split=Split.TRAIN,
    origin="curated/train/elementwise_rope_pair_fp16",
    op_category="elementwise_chain",
    difficulty=None,
    pytorch_code=(
        "import torch\n\n"
        "def op(\n"
        "    x: torch.Tensor,\n"
        "    cos: torch.Tensor,\n"
        "    sin: torch.Tensor,\n"
        ") -> torch.Tensor:\n"
        "    half = x.shape[-1] // 2\n"
        "    x1, x2 = x[..., :half], x[..., half:]\n"
        "    rotated = torch.cat([-x2, x1], dim=-1)\n"
        "    return x * cos + rotated * sin\n"
    ),
    input_shapes=[[2, 32, 128, 64], [128, 64], [128, 64]],
    input_dtypes=["float16", "float32", "float32"],
    rng_seed=42,
)

# ---------------------------------------------------------------------------
# Server fixture
# ---------------------------------------------------------------------------

def _configure(scripts: list[dict]) -> None:
    resp = httpx.post(f"{_MOCK_URL}/admin/configure", json={"scripts": scripts}, timeout=5.0)
    resp.raise_for_status()


@pytest.fixture(scope="session", autouse=True)
def mock_server():
    """Start the mock verification server once for the whole test session."""
    os.environ.setdefault("TRIED_ROLE", "orchestrator")
    _DATA_DIR.mkdir(exist_ok=True)

    from mock_server import app  # local import to avoid circular issues at module level

    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        daemon=True,
    )
    thread.start()

    # Wait until server is accepting connections
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{_MOCK_URL}/docs", timeout=1.0)
            break
        except httpx.ConnectError:
            time.sleep(0.1)

    yield

    server.should_exit = True


def _make_client() -> VerificationClient:
    return VerificationClient(_MOCK_URL, _TEST_KEY)


def _log_summary(label: str, scripts: list[dict], row: DatasetRow) -> None:
    _log.info("━━━ %s — %d script(s) configured ━━━", label, len(scripts))
    _log.info("total attempts: %d", len(row.attempts))
    for a in row.attempts:
        advice = a.judge_fix_suggestion
        advice_preview = f'"{advice[:120]}{"…" if len(advice) > 120 else ""}"' if advice else "none"
        _log.info(
            "  attempt %d  compile=%-7s  correctness=%-6s  judge=%-35s  fix=%s",
            a.attempt_n,
            a.compile.status.value,
            a.correctness.status.value if a.correctness else "—",
            a.judge_classification.value,
            advice_preview,
        )


def _read_row(subdir: str) -> DatasetRow:
    path = _DATA_DIR / subdir / "dataset.jsonl"
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert lines, f"dataset.jsonl is empty in {path}"
    return DatasetRow.model_validate_json(lines[-1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEasyScenario:
    """
    Mock: compile OK → run passes on attempt 0.
    Expect: success in one attempt, no prior advice applied.
    """

    def test_success_on_first_attempt(self):
        _configure(_EASY_SCRIPTS)
        _log.info("configuring mock: %d attempt script(s)", len(_EASY_SCRIPTS))
        run_job(_EASY_RECORD, _make_client(), _DATA_DIR / "easy")

        row = _read_row("easy")
        _log_summary("EASY", _EASY_SCRIPTS, row)
        assert len(row.attempts) == 1
        assert row.attempts[0].judge_classification == JudgeClassification.COMPILED_CORRECT

    def test_no_prior_advice_on_first_attempt(self):
        row = _read_row("easy")
        assert row.attempts[0].prior_advice_applied is None

    def test_attempt_0_compile_succeeded(self):
        row = _read_row("easy")
        assert row.attempts[0].compile.status == CompileStatus.SUCCESS

    def test_attempt_0_correctness_passed(self):
        row = _read_row("easy")
        assert row.attempts[0].correctness is not None
        assert row.attempts[0].correctness.status == CorrectnessStatus.PASSED


class TestMediumScenario:
    """
    Mock: attempt 0 correctness fails, attempt 1 passes.
    Expect: judge gives fix advice, attempt 1 carries that advice, pipeline succeeds.
    """

    def test_success_on_second_attempt(self):
        _configure(_MEDIUM_SCRIPTS)
        _log.info("configuring mock: %d attempt script(s)", len(_MEDIUM_SCRIPTS))
        run_job(_MEDIUM_RECORD, _make_client(), _DATA_DIR / "medium")

        row = _read_row("medium")
        _log_summary("MEDIUM", _MEDIUM_SCRIPTS, row)
        assert len(row.attempts) == 2
        assert row.attempts[1].judge_classification == JudgeClassification.COMPILED_CORRECT

    def test_attempt_0_correctness_failed(self):
        row = _read_row("medium")
        assert row.attempts[0].correctness is not None
        assert row.attempts[0].correctness.status == CorrectnessStatus.FAILED

    def test_judge_gave_fix_advice_after_attempt_0(self):
        row = _read_row("medium")
        assert row.attempts[0].judge_fix_suggestion is not None
        assert len(row.attempts[0].judge_fix_suggestion) > 0

    def test_attempt_1_received_prior_advice(self):
        row = _read_row("medium")
        assert row.attempts[1].prior_advice_applied is not None

    def test_attempt_1_correctness_passed(self):
        row = _read_row("medium")
        assert row.attempts[1].correctness is not None
        assert row.attempts[1].correctness.status == CorrectnessStatus.PASSED


class TestHardScenario:
    """
    Mock: attempt 0 compile fails, attempt 1 run fails, attempt 2 passes.
    Expect: judge recovers from both error types; pipeline succeeds on attempt 2.
    """

    def test_success_on_third_attempt(self):
        _configure(_HARD_SCRIPTS)
        _log.info("configuring mock: %d attempt script(s)", len(_HARD_SCRIPTS))
        run_job(_HARD_RECORD, _make_client(), _DATA_DIR / "hard")

        row = _read_row("hard")
        _log_summary("HARD", _HARD_SCRIPTS, row)
        assert len(row.attempts) == 3
        assert row.attempts[2].judge_classification == JudgeClassification.COMPILED_CORRECT

    def test_attempt_0_compile_status_failed(self):
        row = _read_row("hard")
        assert row.attempts[0].compile.status == CompileStatus.FAILED
        assert row.attempts[0].correctness is None

    def test_attempt_1_compile_ok_but_run_failed(self):
        row = _read_row("hard")
        assert row.attempts[1].compile.status == CompileStatus.SUCCESS
        assert row.attempts[1].correctness is not None
        assert row.attempts[1].correctness.status == CorrectnessStatus.FAILED

    def test_all_attempts_carry_prior_advice_from_attempt_1_onwards(self):
        row = _read_row("hard")
        assert row.attempts[1].prior_advice_applied is not None
        assert row.attempts[2].prior_advice_applied is not None

    def test_attempt_2_correctness_passed(self):
        row = _read_row("hard")
        assert row.attempts[2].correctness is not None
        assert row.attempts[2].correctness.status == CorrectnessStatus.PASSED
