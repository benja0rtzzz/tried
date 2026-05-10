"""
Agent loop for the TRIED orchestrator.

Public API
----------
run_job(record, client, data_dir) -> None
    Runs the generate → compile → run → judge retry loop for one corpus record
    and writes the result to dataset.jsonl. All records loaded by main.py have
    already passed the eager-vs-Inductor preflight check (see preflight_driver).
    Transport exceptions propagate to the caller.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx

from shared.dataset import append_dataset_row
from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    DatasetOutcome,
    Dtype,
    JudgeClassification,
    OpCategory,
    TolerancePolicy,
)
from shared.logging import get_logger
from shared.models import (
    Attempt,
    CompileResult,
    CorpusRecord,
    CorrectnessCheck,
    DatasetRow,
    Source,
)
from shared.verification.api import (
    CompileRequest,
    CompileResponse,
    RunRequest,
    RunResponse,
)

from orchestrator.clients.generator_client import GeneratorResult, generate
from orchestrator.clients.judge_client import JudgeResult, judge
from orchestrator.clients.verification_client import VerificationClient
from orchestrator.prompts.judge import AttemptContext

_log = get_logger(__name__)

MAX_ATTEMPTS = 5


def run_job(
    record: CorpusRecord,
    client: VerificationClient,
    data_dir: Path,
) -> None:
    """Run the agent loop for one corpus record and write to dataset.jsonl.

    All records reaching this function have already passed the eager-vs-Inductor
    preflight check (run by preflight_driver before the agent loop starts).
    Transport exceptions propagate to the caller.
    """
    dataset_path = data_dir / "dataset.jsonl"

    policy = _select_policy(record.input_dtypes, record.op_category)
    _log.info("job start  example_id=%s  policy=%s", record.example_id, policy.value)

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

        compile_resp = client.compile(CompileRequest(triton_code=gen.triton_code))

        run_resp: RunResponse | None = None

        if compile_resp.status == CompileStatus.SUCCESS:
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

        judge_contexts = [_attempt_to_context(a) for a in attempts]
        judge_contexts.append(
            _results_to_context(attempt_n, gen, compile_resp, run_resp)
        )
        judge_result = judge(record.pytorch_code, judge_contexts)

        attempts.append(_build_attempt(
            attempt_n=attempt_n,
            prior_advice=prior_advice,
            gen=gen,
            compile_resp=compile_resp,
            run_resp=run_resp,
            judge_result=judge_result,
            timestamp=timestamp,
            policy=policy,
        ))

        _log.info(
            "attempt %d  classification=%s  example_id=%s",
            attempt_n, judge_result.classification.value, record.example_id,
        )
        if judge_result.fix_suggestion is not None:
            _log.info("judge advice: %s", judge_result.fix_suggestion)

        if judge_result.classification == JudgeClassification.COMPILED_CORRECT:
            break

        prior_code = gen.triton_code
        prior_advice = judge_result.fix_suggestion

    final_outcome = _compute_outcome(attempts)
    _log.info(
        "job done  example_id=%s  attempts=%d  outcome=%s",
        record.example_id, len(attempts), final_outcome.value,
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
    ))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _select_policy(dtypes: list[Dtype], op_category: OpCategory) -> TolerancePolicy:
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
    if run_resp is not None and run_resp.vs_eager is None:
        return CompileStatus.FAILED, run_resp.error_message
    return compile_resp.status, compile_resp.error_message


def _compute_outcome(attempts: list[Attempt]) -> DatasetOutcome:
    for a in attempts:
        if a.judge_classification == JudgeClassification.COMPILED_CORRECT:
            return DatasetOutcome.COMPILED_CORRECT
    for a in attempts:
        if a.compile.status == CompileStatus.SUCCESS:
            return DatasetOutcome.NUMERIC_FAIL
    return DatasetOutcome.COMPILE_FAIL


def _build_attempt(
    *,
    attempt_n: int,
    prior_advice: str | None,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    judge_result: JudgeResult,
    timestamp: datetime,
    policy: TolerancePolicy,
) -> Attempt:
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

    return Attempt(
        attempt_n=attempt_n,
        prior_advice_applied=prior_advice,
        triton_code=gen.triton_code,
        compile=CompileResult(
            status=eff_compile_status,
            error=eff_compile_error,
        ),
        correctness=correctness,
        judge_classification=judge_result.classification,
        judge_fix_suggestion=judge_result.fix_suggestion,
        timestamp=timestamp,
    )


def _attempt_to_context(attempt: Attempt) -> AttemptContext:
    return AttemptContext(
        attempt_n=attempt.attempt_n,
        triton_code=attempt.triton_code,
        compile_status=attempt.compile.status.value,
        compile_error=attempt.compile.error,
        correctness_status=attempt.correctness.status.value if attempt.correctness else None,
        max_abs_diff=attempt.correctness.vs_eager.max_abs_diff if attempt.correctness else None,
        pct_exceeding=attempt.correctness.vs_eager.pct_elements_exceeding_tol if attempt.correctness else None,
        fix_suggestion=attempt.judge_fix_suggestion,
    )


def _results_to_context(
    attempt_n: int,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
) -> AttemptContext:
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
        fix_suggestion=None,
    )
