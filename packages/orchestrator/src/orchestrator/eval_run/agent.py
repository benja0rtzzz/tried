"""Eval-side raw runner. Single attempt per row, no judge calls.

Unlike the dataset agent loop (which uses up to 5 judge-assisted retries
to collect SFT data), the eval pipeline measures raw model capability:
one shot, no feedback. The `judge_classification` field is derived from
compile / correctness / benchmark signals using the same threshold rules
the judge prompt enforces (see prompts/judge/judge_system.txt), so the
schema stays consistent and the stats analyses still work.

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
    FinalOutcome,
    JudgeClassification,
)
from shared.logging import get_logger
from shared.models import (
    CompileResult,
    CorrectnessCheck,
    EvalAttempt,
    EvalBenchmark,
    EvalCorpusRecord,
    EvalRecord,
    Latency,
    Tokens,
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

_SUCCESS_MAP: dict[JudgeClassification, FinalOutcome] = {
    JudgeClassification.CORRECT_AND_FASTER:               FinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    JudgeClassification.CORRECT_AND_COMPETITIVE:          FinalOutcome.COMPILED_CORRECT_PARITY,
    JudgeClassification.CORRECT_BUT_SLOWER_THAN_INDUCTOR: FinalOutcome.COMPILED_CORRECT_SLOW,
}


def run_eval_job(
    record: EvalCorpusRecord,
    client: VerificationClient,
    model_label: str,
    run_id: str,
) -> EvalRecord | None:
    """Run one raw attempt on an EvalCorpusRecord. Returns the EvalRecord
    on completion, or None if /preflight didn't pass on this run.

    `model_label` is used only for logging and is the parent directory the
    caller writes to; it is no longer embedded in the record.

    Transport exceptions propagate to the caller."""
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
                correctness_status=CorrectnessStatus.FAILED,
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

    classification = _derive_classification(compile_resp, run_resp, benchmark_resp)
    attempt = _build_eval_attempt(
        gen=gen,
        compile_resp=compile_resp,
        run_resp=run_resp,
        benchmark_resp=benchmark_resp,
        classification=classification,
        compile_ms=compile_ms,
        run_ms=run_ms,
        timestamp=timestamp,
        tolerance_policy=spec.tolerance_policy,
    )

    final_outcome = _SUCCESS_MAP.get(classification)
    if final_outcome is None:
        if classification == JudgeClassification.CORRECTNESS_FAILED_NUMERIC:
            final_outcome = FinalOutcome.CORRECTNESS_FAILED
        else:
            final_outcome = FinalOutcome.ALL_ATTEMPTS_FAILED
    winning_n = attempt.attempt_n if classification in _SUCCESS_MAP else None

    _log.info(
        "eval job done  example_id=%s  classification=%s  outcome=%s",
        record.example_id, classification.value, final_outcome.value,
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
# Classification derivation (mirrors prompts/judge/judge_system.txt rules)
# ---------------------------------------------------------------------------

def _derive_classification(
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
) -> JudgeClassification:
    eff_status, _ = _effective_compile(compile_resp, run_resp)
    if eff_status == CompileStatus.FAILED:
        return JudgeClassification.OTHER
    if run_resp is None or run_resp.correctness_status != CorrectnessStatus.PASSED:
        return JudgeClassification.CORRECTNESS_FAILED_NUMERIC
    if benchmark_resp is None or benchmark_resp.speedup_vs_inductor is None:
        return JudgeClassification.OTHER
    speedup = benchmark_resp.speedup_vs_inductor
    if speedup < 1.0:
        return JudgeClassification.CORRECT_BUT_SLOWER_THAN_INDUCTOR
    if speedup < 1.1:
        return JudgeClassification.CORRECT_AND_COMPETITIVE
    return JudgeClassification.CORRECT_AND_FASTER


# ---------------------------------------------------------------------------
# Helpers (parallel to orchestrator.dataset.agent's; intentional duplication)
# ---------------------------------------------------------------------------

def _effective_compile(
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
) -> tuple[CompileStatus, str | None]:
    if run_resp is not None and run_resp.vs_eager is None and run_resp.error_message is not None:
        return CompileStatus.FAILED, run_resp.error_message
    return compile_resp.status, compile_resp.error_message


def _build_eval_attempt(
    *,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
    classification: JudgeClassification,
    compile_ms: int,
    run_ms: int | None,
    timestamp: datetime,
    tolerance_policy,
) -> EvalAttempt:
    eff_compile_status, eff_compile_error = _effective_compile(compile_resp, run_resp)

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
        compile=CompileResult(status=eff_compile_status, error=eff_compile_error),
        correctness=correctness,
        benchmark=benchmark,
        judge_classification=classification,
        judge_fix_suggestion=None,
        latency=Latency(
            generator_ms=gen.latency_ms,
            judge_ms=0,
            compile_ms=compile_ms,
            run_ms=run_ms,
        ),
        tokens=Tokens(
            generator_prompt=gen.prompt_tokens,
            generator_completion=gen.completion_tokens,
            judge_prompt=0,
            judge_completion=0,
        ),
        timestamp=timestamp,
    )
