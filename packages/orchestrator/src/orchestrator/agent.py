"""
Agent loop for the TRIED orchestrator.

Public API
----------
run_job(record, client, data_dir) -> FinalOutcome | None
    Runs the full preflight → generate → compile → run → benchmark → judge
    retry loop for one corpus record. Returns the FinalOutcome on completion,
    or None if the preflight check failed (written to skipped.jsonl).
    Transport exceptions propagate to the caller.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from shared.dataset import append_dataset_row, append_skipped
from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    Dtype,
    FinalOutcome,
    JudgeClassification,
    OpCategory,
    TolerancePolicy,
)
from shared.logging import get_logger
from shared.models import (
    Attempt,
    Benchmark,
    CompileResult,
    CorpusRecord,
    CorrectnessCheck,
    DatasetRow,
    Latency,
    Source,
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
from orchestrator.clients.judge_client import JudgeResult, judge
from orchestrator.clients.verification_client import VerificationClient
from orchestrator.prompts.judge import AttemptContext

_log = get_logger(__name__)

MAX_ATTEMPTS = 5

_SUCCESS_MAP: dict[JudgeClassification, FinalOutcome] = {
    JudgeClassification.CORRECT_AND_FASTER:               FinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    JudgeClassification.CORRECT_AND_COMPETITIVE:          FinalOutcome.COMPILED_CORRECT_PARITY,
    JudgeClassification.CORRECT_BUT_SLOWER_THAN_INDUCTOR: FinalOutcome.COMPILED_CORRECT_SLOW,
}


def run_job(
    record: CorpusRecord,
    client: VerificationClient,
    data_dir: Path,
) -> FinalOutcome | None:
    """Run the full agent loop for one corpus record.

    Returns the FinalOutcome written to dataset.jsonl, or None if the preflight
    check failed (record written to skipped.jsonl instead).
    Transport exceptions (server unreachable, OpenAI down) propagate to the
    caller, which is responsible for writing to skipped.jsonl.
    """
    dataset_path = data_dir / "dataset.jsonl"
    skipped_path = data_dir / "skipped.jsonl"

    policy = _select_policy(record.input_dtypes, record.op_category)
    _log.info("job start  example_id=%s  policy=%s", record.example_id, policy.value)

    # --- Preflight ---
    preflight = client.preflight(PreflightRequest(
        pytorch_code=record.pytorch_code,
        input_shapes=record.input_shapes,
        input_dtypes=record.input_dtypes,
        rng_seed=record.rng_seed,
        tolerance_policy=policy,
    ))
    if not preflight.passed:
        reason = preflight.error_message or "preflight: eager vs inductor disagreement"
        _log.warning("preflight failed  example_id=%s  reason=%s", record.example_id, reason)
        append_skipped(skipped_path, record.example_id, reason)
        return None

    # --- Retry loop ---
    attempts: list[Attempt] = []
    prior_code: str | None = None
    prior_advice: str | None = None

    for attempt_n in range(MAX_ATTEMPTS):
        _log.info(
            "attempt %d/%d  example_id=%s",
            attempt_n, MAX_ATTEMPTS - 1, record.example_id,
        )
        timestamp = datetime.now(timezone.utc)

        gen = generate(
            pytorch_code=record.pytorch_code,
            input_shapes=record.input_shapes,
            input_dtypes=[d.value for d in record.input_dtypes],
            prior_code=prior_code,
            prior_advice=prior_advice,
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
                    input_shapes=record.input_shapes,
                    input_dtypes=record.input_dtypes,
                    rng_seed=record.rng_seed,
                    tolerance_policy=policy,
                ))
            except httpx.TimeoutException:
                _log.warning(
                    "attempt %d  /run timed out — recording as correctness FAILED",
                    attempt_n,
                )
                run_resp = RunResponse(
                    correctness_status=CorrectnessStatus.FAILED,
                    error_message="ReadTimeout: /run did not respond within the client timeout",
                )
            run_ms = int((time.monotonic() - t0) * 1000)

            if run_resp.correctness_status == CorrectnessStatus.PASSED:
                benchmark_resp = client.benchmark(BenchmarkRequest(
                    triton_code=gen.triton_code,
                    pytorch_code=record.pytorch_code,
                    input_shapes=record.input_shapes,
                    input_dtypes=record.input_dtypes,
                    rng_seed=record.rng_seed,
                ))

        # Build judge context from all attempts so far + current one
        judge_contexts = [_attempt_to_context(a) for a in attempts]
        judge_contexts.append(
            _results_to_context(attempt_n, gen, compile_resp, run_resp, benchmark_resp)
        )
        judge_result = judge(record.pytorch_code, judge_contexts)

        attempts.append(_build_attempt(
            attempt_n=attempt_n,
            prior_advice=prior_advice,
            gen=gen,
            compile_resp=compile_resp,
            run_resp=run_resp,
            benchmark_resp=benchmark_resp,
            judge_result=judge_result,
            compile_ms=compile_ms,
            run_ms=run_ms,
            timestamp=timestamp,
            policy=policy,
        ))

        _log.info(
            "attempt %d  classification=%s  example_id=%s",
            attempt_n, judge_result.classification.value, record.example_id,
        )
        if judge_result.fix_suggestion is not None:
            _log.info("judge advice: %s", judge_result.fix_suggestion)

        if judge_result.classification in _SUCCESS_MAP:
            break

        prior_code = gen.triton_code
        prior_advice = judge_result.fix_suggestion

    final_outcome, winning_n = _determine_outcome(attempts)
    _log.info(
        "job done  example_id=%s  outcome=%s  attempts=%d",
        record.example_id, final_outcome.value, len(attempts),
    )

    append_dataset_row(dataset_path, DatasetRow(
        example_id=record.example_id,
        source=Source(
            pytorch_code=record.pytorch_code,
            origin=record.origin,
            input_shapes=record.input_shapes,
            input_dtypes=record.input_dtypes,
            rng_seed=record.rng_seed,
            op_category=record.op_category,
        ),
        attempts=attempts,
        final_outcome=final_outcome,
        final_winning_attempt_n=winning_n,
    ))

    return final_outcome


# ---------------------------------------------------------------------------
# Private helpers — data mapping only, no control flow
# ---------------------------------------------------------------------------

def _select_policy(dtypes: list[Dtype], op_category: OpCategory) -> TolerancePolicy:
    """Pick a tolerance policy from the corpus record's dtype and op category."""
    fp16_dtypes = {Dtype.FLOAT16, Dtype.BFLOAT16}
    int_dtypes  = {Dtype.INT8, Dtype.INT16, Dtype.INT32, Dtype.INT64, Dtype.BOOL}
    dtype_set = set(dtypes)

    if dtype_set <= int_dtypes:
        return TolerancePolicy.EXACT_INTEGER

    has_fp16 = bool(dtype_set & fp16_dtypes)

    if op_category == OpCategory.QUANTIZATION:
        return TolerancePolicy.LOW_PRECISION_DEQUANT
    if op_category == OpCategory.FUSED_ATTENTION and has_fp16:
        return TolerancePolicy.ATTENTION_SOFTMAX_FP16
    if op_category == OpCategory.REDUCTION:
        return TolerancePolicy.REDUCTION_FP16 if has_fp16 else TolerancePolicy.REDUCTION_FP32
    return TolerancePolicy.DEFAULT_FP16 if has_fp16 else TolerancePolicy.DEFAULT_FP32


def _effective_compile(
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
) -> tuple[CompileStatus, str | None]:
    """Return the effective compile status and error for an attempt.

    Triton JIT compilation is lazy — a kernel that passes the shallow Python
    import check can still crash on first execution. When run_resp has no
    vs_eager (exception before any tensor comparison), treat it as a compile
    failure so the schema invariant holds:
        correctness null iff compile.status == failed
    """
    if run_resp is not None and run_resp.vs_eager is None:
        return CompileStatus.FAILED, run_resp.error_message
    return compile_resp.status, compile_resp.error_message


def _determine_outcome(attempts: list[Attempt]) -> tuple[FinalOutcome, int | None]:
    """Map the completed attempt list to a FinalOutcome and winning attempt index."""
    last = attempts[-1]
    outcome = _SUCCESS_MAP.get(last.judge_classification)
    if outcome is not None:
        return outcome, last.attempt_n

    if all(a.compile.status == CompileStatus.FAILED for a in attempts):
        return FinalOutcome.ALL_ATTEMPTS_FAILED, None

    if all(
        a.correctness is not None and a.correctness.status == CorrectnessStatus.FAILED
        for a in attempts
    ):
        return FinalOutcome.CORRECTNESS_FAILED, None

    return FinalOutcome.ALL_ATTEMPTS_FAILED, None


def _build_attempt(
    *,
    attempt_n: int,
    prior_advice: str | None,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
    judge_result: JudgeResult,
    compile_ms: int,
    run_ms: int | None,
    timestamp: datetime,
    policy: TolerancePolicy,
) -> Attempt:
    """Assemble an Attempt model from the raw results of one loop iteration."""
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
            tolerance_policy_used=run_resp.tolerance_policy_used or policy,
            vs_eager=run_resp.vs_eager,
            vs_inductor=run_resp.vs_inductor,
        )

    benchmark: Benchmark | None = None
    if benchmark_resp is not None:
        benchmark = Benchmark(
            triton_ms=benchmark_resp.triton_ms,
            eager_ms=benchmark_resp.eager_ms,
            inductor_ms=benchmark_resp.inductor_ms,
            speedup_vs_eager=benchmark_resp.speedup_vs_eager,
            speedup_vs_inductor=benchmark_resp.speedup_vs_inductor,
            triton_std_ms=benchmark_resp.triton_std_ms,
            eager_std_ms=benchmark_resp.eager_std_ms,
            inductor_std_ms=benchmark_resp.inductor_std_ms,
        )

    return Attempt(
        attempt_n=attempt_n,
        prior_advice_applied=prior_advice,
        triton_code=gen.triton_code,
        compile=CompileResult(
            status=eff_compile_status,
            error=eff_compile_error,
        ),
        correctness=correctness,
        benchmark=benchmark,
        judge_classification=judge_result.classification,
        judge_fix_suggestion=judge_result.fix_suggestion,
        latency=Latency(
            generator_ms=gen.latency_ms,
            judge_ms=judge_result.latency_ms,
            compile_ms=compile_ms,
            run_ms=run_ms,
        ),
        tokens=Tokens(
            generator_prompt=gen.prompt_tokens,
            generator_completion=gen.completion_tokens,
            judge_prompt=judge_result.prompt_tokens,
            judge_completion=judge_result.completion_tokens,
        ),
        timestamp=timestamp,
    )


def _attempt_to_context(attempt: Attempt) -> AttemptContext:
    """Convert a completed Attempt model to an AttemptContext for the judge prompt."""
    return AttemptContext(
        attempt_n=attempt.attempt_n,
        triton_code=attempt.triton_code,
        compile_status=attempt.compile.status.value,
        compile_error=attempt.compile.error,
        correctness_status=attempt.correctness.status.value if attempt.correctness else None,
        max_abs_diff=attempt.correctness.vs_eager.max_abs_diff if attempt.correctness else None,
        pct_exceeding=attempt.correctness.vs_eager.pct_elements_exceeding_tol if attempt.correctness else None,
        speedup_vs_eager=attempt.benchmark.speedup_vs_eager if attempt.benchmark else None,
        speedup_vs_inductor=attempt.benchmark.speedup_vs_inductor if attempt.benchmark else None,
        fix_suggestion=attempt.judge_fix_suggestion,
    )


def _results_to_context(
    attempt_n: int,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    benchmark_resp: BenchmarkResponse | None,
) -> AttemptContext:
    """Build an AttemptContext from the raw results of the current in-flight attempt."""
    eff_status, eff_error = _effective_compile(compile_resp, run_resp)
    return AttemptContext(
        attempt_n=attempt_n,
        triton_code=gen.triton_code,
        compile_status=eff_status.value,
        compile_error=eff_error,
        correctness_status=(
            run_resp.correctness_status.value
            if run_resp and run_resp.correctness_status else None
        ),
        max_abs_diff=(
            run_resp.vs_eager.max_abs_diff
            if run_resp and run_resp.vs_eager else None
        ),
        pct_exceeding=(
            run_resp.vs_eager.pct_elements_exceeding_tol
            if run_resp and run_resp.vs_eager else None
        ),
        speedup_vs_eager=benchmark_resp.speedup_vs_eager if benchmark_resp else None,
        speedup_vs_inductor=benchmark_resp.speedup_vs_inductor if benchmark_resp else None,
        fix_suggestion=None,
    )
