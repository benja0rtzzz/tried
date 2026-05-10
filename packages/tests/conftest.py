"""
Shared pytest fixtures and env setup for the TRIED test suite.
Loads packages/orchestrator/.env so tests can run without manually sourcing it.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from orchestrator.prompts.judge import AttemptContext

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent.parent / "packages" / "orchestrator" / ".env")
os.environ.setdefault("TRIED_ROLE", "orchestrator")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pytorch_code() -> str:
    return (
        "import torch\n\n"
        "def op(x):\n"
        "    return x * 2.0 + 1.0\n"
    )


@pytest.fixture
def compile_failure_context() -> AttemptContext:
    """Attempt that failed to compile — most basic judge input."""
    return AttemptContext(
        attempt_n=0,
        triton_code=(
            "@triton.jit\n"
            "def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    x = tl.load(x_ptr + offs)  # missing mask\n"
            "    tl.store(out_ptr + offs, x * 2.0 + 1.0)  # missing mask\n"
        ),
        compile_status="failed",
        compile_error="triton.compiler.errors.CompilationError: invalid memory access at line 5",
        correctness_status=None,
        max_abs_diff=None,
        pct_exceeding=None,
        speedup_vs_eager=None,
        speedup_vs_inductor=None,
        fix_suggestion=None,
    )


@pytest.fixture
def correctness_failure_context() -> AttemptContext:
    """Attempt that compiled but produced wrong numeric output."""
    return AttemptContext(
        attempt_n=0,
        triton_code=(
            "@triton.jit\n"
            "def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    mask = offs < n\n"
            "    x = tl.load(x_ptr + offs, mask=mask)\n"
            "    tl.store(out_ptr + offs, x * 2.0, mask=mask)  # missing + 1.0\n"
        ),
        compile_status="success",
        compile_error=None,
        correctness_status="failed",
        max_abs_diff=1.0,
        pct_exceeding=100.0,
        speedup_vs_eager=None,
        speedup_vs_inductor=None,
        fix_suggestion=None,
    )
