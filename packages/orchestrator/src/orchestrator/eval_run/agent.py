"""Eval-side raw runner. Single attempt per row, no judge calls.

The eval pipeline measures raw model capability: one shot, no feedback.
final_outcome is derived directly from compile / correctness / benchmark results.

Inputs:  EvalCorpusRecord rows from the locked holdout set.
Outputs: EvalRecord rows (one attempt per row).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    EvalFinalOutcome,
)
from shared.logging import get_logger
from shared.models import (
    CompileResult,
    CorrectnessCheck,
    EvalAttempt,
    EvalBenchmark,
    EvalCorpusRecord,
    EvalLatency,
    EvalRecord,
    EvalTokens,
)
from shared.verification.api import (
    BenchmarkRequest,
    BenchmarkResponse,
    CompileRequest,
    CompileResponse,
    PreflightRequest,
    RunRequest,
    RunResponse,
)

from orchestrator.clients.generator_client import GeneratorResult, generate
from orchestrator.clients.verification_client import VerificationClient

_log = get_logger(__name__)


def run_eval_job(
    record: EvalCorpusRecord,
    client: VerificationClient,
    model_label: str,
    run_id: str,
) -> EvalRecord | None:
    """Run one raw attempt on an EvalCorpusRecord. Returns the EvalRecord
    on completion, or None if /preflight didn't pass on this run.

    `model_label` is used only for logging and is the parent directory the
    caller writes to; it is not embedded in the record.

    Transport exceptions propagate to the caller.
    """
    spec = record.spec
    _log.info(
        "eval job start  example_id=%s  form=%s  model=%s",
        record.example_id, spec.form, model_label,
    )

    preflight = client.preflight(PreflightRequest(
        pytorch_code=record.pytorch_code,
        input_shapes=spec.input_shapes,
        input_dtypes=spec.input_dtypes,
        rng_seed=spec.rng_seed,
        tolerance_policy=spec.tolerance_policy,
    ))
    if not preflight.passed:
        reason = preflight.error_message or "preflight regression on accepted eval row"
        _log.warning(
            "eval preflight skipped  example_id=%s  reason=%s",
            record.example_id, reason,
        )
        return None

    timestamp = datetime.now(timezone.utc)
    gen = generate(
        pytorch_code=record.pytorch_code,
        input_shapes=spec.input_shapes,
        input_dtypes=[d.value for d in spec.input_dtypes],
        prior_code=None,
        prior_advice=None,
    )

    t0 = time.monotonic()
    compile_resp = client.compile(CompileRequest(triton_code=gen.triton_code))
    compile_ms = int((time.monotonic() - t0) * 1000)

    run_resp: RunResponse | None = None
    run_ms: int | None = None
    benchmark_resp: BenchmarkResponse | None = None

    if compile_resp.status == CompileStatus.SUCCESS:
        t0 = time.monotonic()
        try:
            run_resp = client.run(RunRequest(
                triton_code=gen.triton_code,
                pytorch_code=record.pytorch_code,
                input_shapes=spec.input_shapes,
                input_dtypes=spec.input_dtypes,
                rng_seed=spec.rng_seed,
                tolerance_policy=spec.tolerance_policy,
            ))
        except httpx.TimeoutException:
            _log.warning("/run timed out  example_id=%s", record.example_id)
            run_resp = RunResponse(
                error_message="ReadTimeout: /run did not respond within the client timeout",
            )
        run_ms = int((time.monotonic() - t0) * 1000)

        if run_resp.correctness_status == CorrectnessStatus.PASSED:
            benchmark_resp = client.benchmark(BenchmarkRequest(
                triton_code=gen.triton_code,
                pytorch_code=record.pytorch_code,
                input_shapes=spec.input_shapes,
                input_dtypes=spec.input_dtypes,
                rng_seed=spec.rng_seed,
            ))

    attempt = _build_eval_attempt(
        gen=gen,
        compile_resp=compile_resp,
        run_resp=run_resp,
        benchmark_resp=benchmark_resp,
        compile_ms=compile_ms,
        run_ms=run_ms,
        timestamp=timestamp,
        tolerance_policy=spec.tolerance_policy,
    )

    final_outcome, winning_n = _derive_outcome(compile_resp, run_resp, benchmark_resp, attempt.attempt_n)

    _log.info(
        "eval job done  example_id=%s  outcome=%s",
        record.example_id, final_outcome.value,
    )

    return EvalRecord(
        example_id=record.example_id,
        run_id=run_id,
        spec=spec,
        attempts=[attempt],
        final_outcome=final_outcome,
        final_winning_attempt_n=winning_n,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_outcome(
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
    attempt_n: int,
) -> tuple[EvalFinalOutcome, int | None]:
    if compile_resp.status == CompileStatus.FAILED:
        return EvalFinalOutcome.ALL_ATTEMPTS_FAILED, None
    if _run_error(run_resp) is not None:
        return EvalFinalOutcome.RUNTIME_FAIL, None
    if run_resp is None or run_resp.correctness_status != CorrectnessStatus.PASSED:
        return EvalFinalOutcome.CORRECTNESS_FAILED, None
    if benchmark_resp is None:
        return EvalFinalOutcome.ALL_ATTEMPTS_FAILED, None
    speedup = benchmark_resp.speedup_vs_inductor or 0.0
    if speedup >= 1.1:
        return EvalFinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR, attempt_n
    if speedup >= 1.0:
        return EvalFinalOutcome.COMPILED_CORRECT_PARITY, attempt_n
    return EvalFinalOutcome.COMPILED_CORRECT_SLOW, attempt_n


def _build_eval_attempt(
    *,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
    compile_ms: int,
    run_ms: int | None,
    timestamp: datetime,
    tolerance_policy,
) -> EvalAttempt:
    run_error = _run_error(run_resp)

    correctness: CorrectnessCheck | None = None
    if (
        run_resp is not None
        and run_resp.correctness_status is not None
        and run_resp.vs_eager is not None
        and run_resp.vs_inductor is not None
    ):
        correctness = CorrectnessCheck(
            status=run_resp.correctness_status,
            tolerance_policy_used=run_resp.tolerance_policy_used or tolerance_policy,
            vs_eager=run_resp.vs_eager,
            vs_inductor=run_resp.vs_inductor,
        )

    benchmark: EvalBenchmark | None = None
    if benchmark_resp is not None:
        benchmark = EvalBenchmark(
            triton_ms=benchmark_resp.triton_ms,
            eager_ms=benchmark_resp.eager_ms,
            inductor_ms=benchmark_resp.inductor_ms,
            speedup_vs_eager=benchmark_resp.speedup_vs_eager,
            speedup_vs_inductor=benchmark_resp.speedup_vs_inductor,
            triton_std_ms=benchmark_resp.triton_std_ms,
            eager_std_ms=benchmark_resp.eager_std_ms,
            inductor_std_ms=benchmark_resp.inductor_std_ms,
            triton_samples_ms=benchmark_resp.triton_samples_ms,
            eager_samples_ms=benchmark_resp.eager_samples_ms,
            inductor_samples_ms=benchmark_resp.inductor_samples_ms,
        )

    return EvalAttempt(
        attempt_n=0,
        prior_advice_applied=None,
        triton_code=gen.triton_code,
        compile=CompileResult(
            status=compile_resp.status,
            error=compile_resp.error_message,
        ),
        run_error=run_error,
        correctness=correctness,
        benchmark=benchmark,
        latency=EvalLatency(
            generator_ms=gen.latency_ms,
            compile_ms=compile_ms,
            run_ms=run_ms,
        ),
        tokens=EvalTokens(
            generator_prompt=gen.prompt_tokens,
            generator_completion=gen.completion_tokens,
        ),
        timestamp=timestamp,
    )


def _run_error(run_resp: RunResponse | None) -> str | None:
    if run_resp is None:
        return None
    if run_resp.vs_eager is not None and run_resp.vs_inductor is not None:
        return None
    return (
        run_resp.error_message
        or "Runtime verification failed before producing correctness stats"
    )
