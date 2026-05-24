"""
Agent loop for the TRIED orchestrator.

Public API
----------
run_job(record, client, data_dir) -> None
    Runs the generate → compile → run retry loop for one corpus record and
    writes the result to dataset.jsonl. Failed attempts call the judge for
    taxonomy labels and retry advice; passing attempts skip it. All records
    loaded by main.py have already passed the eager-vs-Inductor preflight check
    (see preflight_driver). Transport exceptions propagate to the caller.
"""
from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from shared.dataset import append_dataset_row
from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    DatasetOutcome,
    FailureSymptom,
    TolerancePolicy,
)
from shared.logging import get_logger
from shared.models import (
    Attempt,
    CompileResult,
    CorpusRecord,
    DatasetCorrectnessCheck,
    DatasetRow,
    Source,
    derive_dataset_id,
)
from shared.verification.api import (
    CompileRequest,
    CompileResponse,
    RunRequest,
    RunResponse,
)

from orchestrator.clients.dataset.generator_client import GeneratorResult, generate
from orchestrator.clients.dataset.judge_client import JudgeResult, judge
from orchestrator.clients.verification_client import VerificationClient
from orchestrator.prompts.judge import AttemptContext

_log = get_logger(__name__)

MAX_ATTEMPTS = 3


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

    policy = record.tolerance_policy
    dataset_id = record.dataset_id or derive_dataset_id(
        source_id=record.example_id,
        op_category=record.op_category,
        pytorch_code=record.pytorch_code,
        input_shapes=record.input_shapes,
        input_dtypes=record.input_dtypes,
        rng_seed=record.rng_seed,
        tolerance_policy=policy,
    )
    _log.info(
        "job start  dataset_id=%s  source_id=%s  policy=%s",
        dataset_id,
        record.example_id,
        policy.value,
    )

    # --- Retry loop ---
    attempts: list[Attempt] = []
    prior_code: str | None = None
    prior_advice: str | None = None

    for attempt_n in range(MAX_ATTEMPTS):
        _log.info(
            "attempt %d/%d  dataset_id=%s  source_id=%s",
            attempt_n, MAX_ATTEMPTS - 1, dataset_id, record.example_id,
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
                    "attempt %d  /run timed out — recording as runtime failure",
                    attempt_n,
                )
                run_resp = RunResponse(
                    error_message="ReadTimeout: /run did not respond within the client timeout",
                )

        attempt_passed = _verification_passed(run_resp)
        final_attempt = attempt_n == MAX_ATTEMPTS - 1
        judge_result = None
        if not attempt_passed:
            judge_contexts = [_attempt_to_context(a) for a in attempts]
            judge_contexts.append(
                _results_to_context(attempt_n, gen, compile_resp, run_resp)
            )
            judge_result = judge(record.pytorch_code, judge_contexts)

        attempts.append(_build_attempt(
            attempt_n=attempt_n,
            prior_advice=prior_advice,
            previous_code=prior_code,
            gen=gen,
            compile_resp=compile_resp,
            run_resp=run_resp,
            judge_result=judge_result,
            timestamp=timestamp,
            policy=policy,
        ))

        if attempt_passed:
            break

        assert judge_result is not None
        _log.info(
            "attempt %d  classification=%s  root_cause=%s  repair_action=%s  "
            "dataset_id=%s",
            attempt_n,
            judge_result.classification.value,
            judge_result.root_cause.value if judge_result.root_cause else None,
            judge_result.repair_action.value if judge_result.repair_action else None,
            dataset_id,
        )
        if judge_result.fix_suggestion is not None:
            _log.info("judge advice: %s", judge_result.fix_suggestion)

        if final_attempt:
            break

        prior_code = gen.triton_code
        prior_advice = judge_result.fix_suggestion

    final_outcome = _compute_outcome(attempts)
    _log.info(
        "job done  dataset_id=%s  source_id=%s  attempts=%d  outcome=%s",
        dataset_id, record.example_id, len(attempts), final_outcome.value,
    )

    append_dataset_row(dataset_path, DatasetRow(
        dataset_id=dataset_id,
        example_id=record.example_id,
        source=Source(
            example_id=record.example_id,
            pytorch_code=record.pytorch_code,
            origin=record.origin,
            input_shapes=record.input_shapes,
            input_dtypes=record.input_dtypes,
            rng_seed=record.rng_seed,
            op_category=record.op_category,
            tolerance_policy=policy,
        ),
        attempts=attempts,
        final_outcome=final_outcome,
    ))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_outcome(attempts: list[Attempt]) -> DatasetOutcome:
    for a in attempts:
        if (
            a.correctness is not None
            and a.correctness.status == CorrectnessStatus.PASSED
        ):
            return DatasetOutcome.COMPILED_CORRECT
    for a in attempts:
        if (
            a.correctness is not None
            and a.correctness.status == CorrectnessStatus.FAILED
        ):
            return DatasetOutcome.NUMERIC_FAIL
    for a in attempts:
        if a.run_error is not None:
            return DatasetOutcome.RUNTIME_FAIL
    return DatasetOutcome.COMPILE_FAIL


def _verification_passed(run_resp: RunResponse | None) -> bool:
    return (
        run_resp is not None
        and run_resp.correctness_status == CorrectnessStatus.PASSED
        and run_resp.vs_eager is not None
        and run_resp.vs_inductor is not None
    )


def _build_attempt(
    *,
    attempt_n: int,
    prior_advice: str | None,
    previous_code: str | None,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    judge_result: JudgeResult | None,
    timestamp: datetime,
    policy: TolerancePolicy,
) -> Attempt:
    run_error = _run_error(run_resp)

    correctness: DatasetCorrectnessCheck | None = None
    if (
        run_resp is not None
        and run_resp.correctness_status is not None
        and run_resp.vs_eager is not None
        and run_resp.vs_inductor is not None
    ):
        correctness = DatasetCorrectnessCheck(
            status=run_resp.correctness_status,
            tolerance_policy_used=run_resp.tolerance_policy_used or policy,
        )

    return Attempt(
        attempt_n=attempt_n,
        prior_advice_applied=prior_advice,
        patch_from_previous=_patch_from_previous(previous_code, gen.triton_code),
        triton_code=gen.triton_code,
        compile=CompileResult(
            status=compile_resp.status,
            error=compile_resp.error_message,
        ),
        run_error=run_error,
        correctness=correctness,
        failure_symptom=_failure_symptom(compile_resp, run_resp, correctness),
        judge_classification=(
            judge_result.classification if judge_result is not None else None
        ),
        judge_root_cause=(
            judge_result.root_cause if judge_result is not None else None
        ),
        judge_repair_action=(
            judge_result.repair_action if judge_result is not None else None
        ),
        judge_fix_suggestion=(
            judge_result.fix_suggestion if judge_result is not None else None
        ),
        timestamp=timestamp,
    )


def _attempt_to_context(attempt: Attempt) -> AttemptContext:
    return AttemptContext(
        attempt_n=attempt.attempt_n,
        triton_code=attempt.triton_code,
        compile_status=attempt.compile.status.value,
        compile_error=attempt.compile.error,
        run_error=attempt.run_error,
        correctness_status=attempt.correctness.status.value if attempt.correctness else None,
        max_abs_diff=None,
        pct_exceeding=None,
        fix_suggestion=attempt.judge_fix_suggestion,
    )


def _results_to_context(
    attempt_n: int,
    gen: GeneratorResult,
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
) -> AttemptContext:
    return AttemptContext(
        attempt_n=attempt_n,
        triton_code=gen.triton_code,
        compile_status=compile_resp.status.value,
        compile_error=compile_resp.error_message,
        run_error=_run_error(run_resp),
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


def _run_error(run_resp: RunResponse | None) -> str | None:
    if run_resp is None:
        return None
    if run_resp.vs_eager is not None and run_resp.vs_inductor is not None:
        return None
    return (
        run_resp.error_message
        or "Runtime verification failed before producing correctness stats"
    )


def _patch_from_previous(previous_code: str | None, current_code: str) -> str | None:
    if previous_code is None:
        return None
    if previous_code == current_code:
        return ""
    return "\n".join(
        difflib.unified_diff(
            previous_code.splitlines(),
            current_code.splitlines(),
            fromfile="previous_attempt.py",
            tofile="current_attempt.py",
            lineterm="",
        )
    ) + "\n"


def _failure_symptom(
    compile_resp: CompileResponse,
    run_resp: RunResponse | None,
    correctness: DatasetCorrectnessCheck | None,
) -> FailureSymptom | None:
    if compile_resp.status == CompileStatus.FAILED:
        return _classify_failure_text(compile_resp.error_message)
    if correctness is not None:
        if correctness.status == CorrectnessStatus.FAILED:
            return FailureSymptom.NUMERIC_MISMATCH
        return None
    return _classify_failure_text(_run_error(run_resp))


def _classify_failure_text(message: str | None) -> FailureSymptom | None:
    if not message:
        return None
    text = message.lower()

    if "timed out" in text:
        return FailureSymptom.TIMEOUT
    if "illegal memory access" in text:
        return FailureSymptom.CUDA_ILLEGAL_MEMORY_ACCESS
    if "unsupported ptr type" in text or "cannot be accessed from triton" in text:
        return FailureSymptom.UNSUPPORTED_PTR_TYPE
    if (
        "cannot make_shape_compatible" in text
        or "incompatible dimensions" in text
        or "equal ranks" in text
        or "block type" in text
    ):
        return FailureSymptom.INCOMPATIBLE_BLOCK_SHAPE
    if "arange" in text and "constexpr" in text:
        return FailureSymptom.NON_CONSTEXPR_ARANGE_BOUND
    if (
        text.startswith("nameerror")
        or " is not defined" in text
        or "out of scope" in text
        or text.startswith("unboundlocalerror")
    ):
        return FailureSymptom.UNDEFINED_SYMBOL
    if "unsupportedlanguageconstruct" in text or "simultaneous multiple comparison" in text:
        return FailureSymptom.UNSUPPORTED_LANGUAGE_CONSTRUCT
    if (
        "unrecognised" in text
        or "unexpected keyword" in text
        or "missing" in text and "argument" in text
        or "grid" in text and "tuple" in text
    ):
        return FailureSymptom.LAUNCH_SIGNATURE_MISMATCH
    if (
        "module 'triton.language' has no attribute" in text
        or "cannot call @triton.jit" in text
        or "device option is deprecated" in text
        or "tl.store" in text and "unsupported" in text
    ):
        return FailureSymptom.INVALID_TRITON_API
    if "unexpected output type" in text or "return" in text and "tensor" in text:
        return FailureSymptom.INVALID_OUTPUT_TYPE
    if (
        "dimension out of range" in text
        or "tuple index out of range" in text
        or "not enough values to unpack" in text
        or "invalid for input of size" in text
        or re.search(r"size of tensor .* must match", text)
        or "incompatible shapes for matmul" in text
    ):
        return FailureSymptom.HOST_SHAPE_ERROR
    if text.startswith("compilationerror"):
        return FailureSymptom.TRITON_COMPILATION_ERROR
    return FailureSymptom.PYTHON_RUNTIME_ERROR
