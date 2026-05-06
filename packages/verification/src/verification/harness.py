"""
Execution harness for the Lenovo verification server.
Handles code loading, input generation, correctness comparison, and benchmarking.
All functions are synchronous and GPU-blocking — callers are responsible for
threading/async wrapping.

NOTE on _load_wrapper: code is written to a real .py tempfile before import.
Triton's @triton.jit decorator calls inspect.getsource() at decoration time,
which requires the function to live in a real file — exec("<string>") raises
ValueError: @jit functions should be defined in a Python file.
The tempfile is deleted after exec_module() returns; by then Triton has already
read and cached the source inside the JITFunction object.

_load_wrapper relies on @triton.jit producing a triton.JITFunction that is
NOT a types.FunctionType. If that assumption is wrong, _load_wrapper returns
the kernel instead of the wrapper and every /run call silently fails.
Confirm with:
    import triton, types
    @triton.jit
    def k(): pass
    assert not isinstance(k, types.FunctionType), "assumption broken — fix _load_wrapper"
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import sys
import tempfile
import types
import uuid
from typing import Any

import torch
from shared.enums import CompileStatus, CorrectnessStatus, Dtype, TolerancePolicy
from shared.logging import get_logger
from shared.models import CorrectnessStats
from shared.verification.api import (
    BenchmarkResponse,
    CompileResponse,
    PreflightResponse,
    RunResponse,
)
from shared.verification.tolerance import ComparisonMode
from shared.verification.tolerance import get as get_tolerance

_log = get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    Dtype.FLOAT16.value: torch.float16,
    Dtype.FLOAT32.value: torch.float32,
    Dtype.BFLOAT16.value: torch.bfloat16,
    Dtype.FLOAT64.value: torch.float64,
    Dtype.INT8.value: torch.int8,
    Dtype.INT16.value: torch.int16,
    Dtype.INT32.value: torch.int32,
    Dtype.INT64.value: torch.int64,
    Dtype.BOOL.value: torch.bool,
}

_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.bool}
_WARMUP_ITERS = 10
_TIMED_ITERS = 100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    input_shapes: list[list[int]],
    input_dtypes: list[Dtype],
    rng_seed: int,
) -> list[torch.Tensor]:
    torch.manual_seed(rng_seed)
    tensors = []
    for shape, dtype in zip(input_shapes, input_dtypes):
        dt = _DTYPE_MAP[dtype.value]
        if dt in _INTEGER_DTYPES:
            t = torch.randint(0, 127, shape, dtype=dt, device="cuda")
        else:
            t = torch.randn(shape, dtype=dt, device="cuda")
        tensors.append(t)
    return tensors


def _load_wrapper(code: str) -> types.FunctionType:
    """Write code to a temp .py file, import it, and return the wrapper function.

    For Triton code the @triton.jit kernel becomes a JITFunction (not
    types.FunctionType), so the last FunctionType is always the wrapper.
    The tempfile is deleted after the module is loaded; Triton caches source
    internally during decoration so the file is not needed after import.
    SyntaxError and all other exceptions propagate to the caller.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="_triton_"
        ) as f:
            f.write(code)
            tmp_path = f.name
        mod_name = f"_triton_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
        if spec is None:
            raise ImportError(f"Could not create module spec for {tmp_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    fns = [v for v in vars(module).values() if isinstance(v, types.FunctionType)]
    if not fns:
        raise ValueError("No callable wrapper function found in code")
    return fns[-1]


def _to_flat_tensor(output: Any) -> torch.Tensor:
    """Flatten a function's return value to a 1-D tensor for comparison.

    Handles single tensors and tuples/lists of tensors.
    """
    if isinstance(output, torch.Tensor):
        return output.flatten()
    if isinstance(output, (tuple, list)):
        parts = [t.flatten() for t in output if isinstance(t, torch.Tensor)]
        if not parts:
            raise ValueError("Function returned no tensors")
        return torch.cat(parts)
    raise TypeError(f"Unexpected output type: {type(output)}")


def _compute_stats(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    tol,
) -> tuple[CorrectnessStats, bool]:
    """Compute all 5 correctness stats and a pass/fail bool for one pair."""
    c = candidate.detach().float()
    r = reference.detach().float()

    if tol.comparison == ComparisonMode.EXACT:
        passed = torch.equal(candidate, reference)
        abs_diff = (c - r).abs()
        rel_diff = abs_diff / r.abs().clamp(min=1e-8)
        n_exceed = int((abs_diff > 0).sum().item())
        total = max(c.numel(), 1)
        return CorrectnessStats(
            max_abs_diff=float(abs_diff.max().item()),
            max_rel_diff=float(rel_diff.max().item()),
            mean_abs_diff=float(abs_diff.mean().item()),
            n_elements_exceeding_tol=n_exceed,
            pct_elements_exceeding_tol=100.0 * n_exceed / total,
        ), passed

    if tol.comparison == ComparisonMode.INF_AWARE_NUMERIC:
        inf_mismatch = int(
            (
                (torch.isposinf(c) != torch.isposinf(r))
                | (torch.isneginf(c) != torch.isneginf(r))
            )
            .sum()
            .item()
        )
        nan_count = int((torch.isnan(c) | torch.isnan(r)).sum().item())
        bad = inf_mismatch + nan_count
        if bad > 0:
            total = max(c.numel(), 1)
            return CorrectnessStats(
                max_abs_diff=math.inf,
                max_rel_diff=math.inf,
                mean_abs_diff=math.inf,
                n_elements_exceeding_tol=bad,
                pct_elements_exceeding_tol=100.0 * bad / total,
            ), False
        # Restrict numeric comparison to finite positions
        finite_mask = ~(torch.isinf(c) | torch.isinf(r))
        c = c[finite_mask]
        r = r[finite_mask]

    # Standard numeric comparison
    abs_diff = (c - r).abs()
    rel_diff = abs_diff / r.abs().clamp(min=1e-8)
    exceeds = abs_diff > (tol.atol + tol.rtol * r.abs())
    n_exceed = int(exceeds.sum().item())
    total = max(c.numel(), 1)

    return CorrectnessStats(
        max_abs_diff=float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0,
        max_rel_diff=float(rel_diff.max().item()) if rel_diff.numel() > 0 else 0.0,
        mean_abs_diff=float(abs_diff.mean().item()) if abs_diff.numel() > 0 else 0.0,
        n_elements_exceeding_tol=n_exceed,
        pct_elements_exceeding_tol=100.0 * n_exceed / total,
    ), n_exceed == 0


def _time_fn(fn: types.FunctionType, inputs: list[torch.Tensor]) -> tuple[float, float]:
    """Time fn over _TIMED_ITERS iterations using CUDA events.

    Returns (median_ms, stdev_ms). Warmup of _WARMUP_ITERS runs is done first.
    """
    with torch.no_grad():
        for _ in range(_WARMUP_ITERS):
            fn(*inputs)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_TIMED_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_TIMED_ITERS)]

    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(_TIMED_ITERS):
            starts[i].record()
            fn(*inputs)
            ends[i].record()
    torch.cuda.synchronize()

    times = [starts[i].elapsed_time(ends[i]) for i in range(_TIMED_ITERS)]
    return statistics.median(times), statistics.stdev(times)


# ---------------------------------------------------------------------------
# Public API — called by server endpoints
# ---------------------------------------------------------------------------


def compile_check(triton_code: str) -> CompileResponse:
    """Load the Triton code from a temp file and verify a wrapper function exists."""
    try:
        wrapper = _load_wrapper(triton_code)  # noqa: F841 — existence check only
        return CompileResponse(status=CompileStatus.SUCCESS)
    except ValueError as e:
        return CompileResponse(
            status=CompileStatus.FAILED,
            error_message=str(e),
        )
    except SyntaxError as e:
        return CompileResponse(
            status=CompileStatus.FAILED,
            error_message=f"SyntaxError: {e}",
        )
    except Exception as e:
        return CompileResponse(
            status=CompileStatus.FAILED,
            error_message=f"{type(e).__name__}: {e}",
        )


def preflight(
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[Dtype],
    rng_seed: int,
    tolerance_policy: TolerancePolicy,
) -> PreflightResponse:
    """Run eager vs Inductor sanity check on the PyTorch reference code."""
    try:
        torch_fn = _load_wrapper(pytorch_code)
        tol = get_tolerance(tolerance_policy)
        inputs = _make_inputs(input_shapes, input_dtypes, rng_seed)

        with torch.no_grad():
            eager_out = _to_flat_tensor(torch_fn(*inputs))

        inductor_fn = torch.compile(torch_fn, backend="inductor")
        with torch.no_grad():
            inductor_out = _to_flat_tensor(inductor_fn(*inputs))

        stats, passed = _compute_stats(inductor_out, eager_out, tol)
        return PreflightResponse(passed=passed, vs_eager_inductor=stats)
    except Exception as e:
        _log.error("preflight failed: %s: %s", type(e).__name__, e)
        return PreflightResponse(passed=False, error_message=f"{type(e).__name__}: {e}")


def run_verification(
    triton_code: str,
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[Dtype],
    rng_seed: int,
    tolerance_policy: TolerancePolicy,
) -> RunResponse:
    """Run Triton candidate against eager and Inductor, returning all 10 stats."""
    try:
        torch_fn = _load_wrapper(pytorch_code)
        triton_fn = _load_wrapper(triton_code)
        tol = get_tolerance(tolerance_policy)
        inputs = _make_inputs(input_shapes, input_dtypes, rng_seed)

        with torch.no_grad():
            eager_out = _to_flat_tensor(torch_fn(*inputs))
            inductor_fn = torch.compile(torch_fn, backend="inductor")
            inductor_out = _to_flat_tensor(inductor_fn(*inputs))
            triton_out = _to_flat_tensor(triton_fn(*inputs))

        stats_eager, passed_eager = _compute_stats(triton_out, eager_out, tol)
        stats_inductor, passed_inductor = _compute_stats(triton_out, inductor_out, tol)

        status = (
            CorrectnessStatus.PASSED
            if passed_eager and passed_inductor
            else CorrectnessStatus.FAILED
        )
        return RunResponse(
            correctness_status=status,
            tolerance_policy_used=tolerance_policy,
            vs_eager=stats_eager,
            vs_inductor=stats_inductor,
        )
    except Exception as e:
        _log.error("run failed: %s: %s", type(e).__name__, e)
        return RunResponse(
            correctness_status=CorrectnessStatus.FAILED,
            error_message=f"{type(e).__name__}: {e}",
        )


def run_benchmark(
    triton_code: str,
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[Dtype],
    rng_seed: int,
) -> BenchmarkResponse:
    """Time Triton, eager, and Inductor over _TIMED_ITERS iterations each.

    Returns median and stdev for all three. Inductor is compiled once before
    the warmup so compilation latency doesn't inflate timed results.
    """
    torch_fn = _load_wrapper(pytorch_code)
    triton_fn = _load_wrapper(triton_code)
    inputs = _make_inputs(input_shapes, input_dtypes, rng_seed)
    inductor_fn = torch.compile(torch_fn, backend="inductor")

    # Trigger Inductor JIT compilation before the warmup loop
    with torch.no_grad():
        inductor_fn(*inputs)

    triton_med, triton_std = _time_fn(triton_fn, inputs)
    eager_med, eager_std = _time_fn(torch_fn, inputs)
    inductor_med, inductor_std = _time_fn(inductor_fn, inputs)

    return BenchmarkResponse(
        triton_ms=triton_med,
        eager_ms=eager_med,
        inductor_ms=inductor_med,
        speedup_vs_eager=eager_med / triton_med,
        speedup_vs_inductor=inductor_med / triton_med,
        triton_std_ms=triton_std,
        eager_std_ms=eager_std,
        inductor_std_ms=inductor_std,
    )
