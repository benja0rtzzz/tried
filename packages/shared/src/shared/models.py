"""
Pydantic v2 models for all record types.
PreflightSafeRecord maps to schema/dataset/preflight_safe_record.json (Step 5 input).
DatasetRow maps to schema/dataset/dataset_record.json (what the orchestrator writes).
EvalSpec / EvalCorpusRecord / EvalRecord map to schema/eval/{eval_spec,eval_corpus_record,eval_result}.json.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    DatasetOutcome,
    Difficulty,
    Dtype,
    EvalFinalOutcome,
    FailureSymptom,
    JudgeClassification,
    JudgeRepairAction,
    JudgeRootCause,
    OpCategory,
    Split,
    TolerancePolicy,
)

__all__ = [
    "CorpusRecord",
    "derive_dataset_id",
    "PreflightSafeRecord",
    "Source",
    "CorrectnessStats",
    "CorrectnessCheck",
    "DatasetCorrectnessCheck",
    "Attempt",
    "DatasetRow",
    "FailureSymptom",
    "JudgeRepairAction",
    "JudgeRootCause",
    "EvalSpec",
    "EvalCorpusRecord",
    "EvalRecord",
    "EvalAttempt",
    "EvalBenchmark",
    "EvalLatency",
    "EvalTokens",
    "AcceptanceMeta",
    "FormName",
]


# Locked closed vocabulary of the 11 fusion-form names. Mirrors the enum
# in schema/eval/eval_spec.json. Source of metadata for each form lives in
# shared.eval.forms.FORMS.
FormName = Literal[
    "chain_2_unary",
    "unary_then_residual",
    "chain_3_unary",
    "unary_then_reduction",
    "softmax_then_unary",
    "unary_then_norm",
    "attention_qkv",
    "fused_linear_norm_activation",
    "gated_mlp_swiglu",
    "chain_4_unary",
    "embedding_then_norm",
]


# ---------------------------------------------------------------------------
# Training corpus/preflight-safe records
# ---------------------------------------------------------------------------

_DATASET_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000da7")


def derive_dataset_id(
    *,
    source_id: str,
    op_category: OpCategory,
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[Dtype],
    rng_seed: int,
    tolerance_policy: TolerancePolicy,
) -> str:
    """Stable UUID for one exact dataset task.

    source_id/example_id identifies the PyTorch code family. dataset_id includes
    the concrete runtime fixtures so shape/seed variants do not collapse during
    dataset resume.
    """
    canonical = json.dumps(
        {
            "source_id": source_id,
            "op_category": op_category.value,
            "pytorch_code": pytorch_code,
            "input_shapes": input_shapes,
            "input_dtypes": [dtype.value for dtype in input_dtypes],
            "rng_seed": rng_seed,
            "tolerance_policy": tolerance_policy.value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return str(uuid.uuid5(_DATASET_NAMESPACE, digest))


class CorpusRecord(BaseModel):
    example_id:   str
    dataset_id:   Optional[str] = None
    split:        Split
    origin:       str
    op_category:  OpCategory
    difficulty:   Optional[Difficulty]
    pytorch_code: str
    input_shapes: list[list[int]]
    input_dtypes: list[Dtype]
    rng_seed:     int = Field(ge=0)
    tolerance_policy: TolerancePolicy

    @model_validator(mode="after")
    def _shapes_dtypes_length_match(self) -> CorpusRecord:
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                f"input_shapes length ({len(self.input_shapes)}) must equal "
                f"input_dtypes length ({len(self.input_dtypes)})"
            )
        return self

    @model_validator(mode="after")
    def _eval_requires_difficulty(self) -> CorpusRecord:
        if self.split == Split.EVAL and self.difficulty is None:
            raise ValueError("eval examples must have a non-null difficulty")
        if self.split == Split.TRAIN and self.difficulty is not None:
            raise ValueError("train examples must have difficulty=null")
        return self

    @model_validator(mode="after")
    def _dataset_id_matches_source(self) -> CorpusRecord:
        expected = derive_dataset_id(
            source_id=self.example_id,
            op_category=self.op_category,
            pytorch_code=self.pytorch_code,
            input_shapes=self.input_shapes,
            input_dtypes=self.input_dtypes,
            rng_seed=self.rng_seed,
            tolerance_policy=self.tolerance_policy,
        )
        if self.dataset_id is None:
            self.dataset_id = expected
            return self
        if self.dataset_id != expected:
            raise ValueError("dataset_id must be derived from the full corpus row")
        return self


class PreflightSafeRecord(BaseModel):
    """Slim persisted shape for generated training examples."""
    model_config = ConfigDict(extra="forbid")

    example_id:       str
    op_category:      OpCategory
    pytorch_code:     str
    input_shapes:     list[list[int]]
    input_dtypes:     list[Dtype]
    rng_seed:         int = Field(ge=0)
    tolerance_policy: TolerancePolicy

    @model_validator(mode="after")
    def _shapes_dtypes_length_match(self) -> PreflightSafeRecord:
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                f"input_shapes length ({len(self.input_shapes)}) must equal "
                f"input_dtypes length ({len(self.input_dtypes)})"
            )
        return self

    @classmethod
    def from_corpus_record(
        cls, record: CorpusRecord, tolerance_policy: TolerancePolicy | None = None
    ) -> PreflightSafeRecord:
        policy = tolerance_policy or record.tolerance_policy
        if tolerance_policy is not None and record.tolerance_policy != tolerance_policy:
            raise ValueError(
                "corpus record tolerance_policy must match the supplied "
                f"preflight policy ({record.tolerance_policy} != {tolerance_policy})"
            )
        if record.split != Split.TRAIN:
            raise ValueError(
                f"preflight-safe rows must have split=train, got {record.split}"
            )
        if record.origin != "synthetic/skeleton":
            raise ValueError(
                "preflight-safe rows must have "
                f"origin='synthetic/skeleton', got {record.origin!r}"
            )
        if record.difficulty is not None:
            raise ValueError(
                f"preflight-safe rows must have difficulty=None, got {record.difficulty}"
            )
        return cls(
            example_id=record.example_id,
            op_category=record.op_category,
            pytorch_code=record.pytorch_code,
            input_shapes=record.input_shapes,
            input_dtypes=record.input_dtypes,
            rng_seed=record.rng_seed,
            tolerance_policy=policy,
        )

    def to_corpus_record(self) -> CorpusRecord:
        dataset_id = derive_dataset_id(
            source_id=self.example_id,
            op_category=self.op_category,
            pytorch_code=self.pytorch_code,
            input_shapes=self.input_shapes,
            input_dtypes=self.input_dtypes,
            rng_seed=self.rng_seed,
            tolerance_policy=self.tolerance_policy,
        )
        return CorpusRecord(
            example_id=self.example_id,
            dataset_id=dataset_id,
            split=Split.TRAIN,
            origin="synthetic/skeleton",
            op_category=self.op_category,
            difficulty=None,
            pytorch_code=self.pytorch_code,
            input_shapes=self.input_shapes,
            input_dtypes=self.input_dtypes,
            rng_seed=self.rng_seed,
            tolerance_policy=self.tolerance_policy,
        )


# ---------------------------------------------------------------------------
# Sub-models for DatasetRow
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """Immutable inputs copied verbatim from the corpus record when the agent loop starts."""
    example_id:   str
    pytorch_code: str
    origin:       str
    input_shapes: list[list[int]]
    input_dtypes: list[Dtype]
    rng_seed:     int = Field(ge=0)
    op_category:  OpCategory
    tolerance_policy: TolerancePolicy

    @model_validator(mode="after")
    def _shapes_dtypes_length_match(self) -> Source:
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                f"input_shapes length ({len(self.input_shapes)}) must equal "
                f"input_dtypes length ({len(self.input_dtypes)})"
            )
        return self


class CorrectnessStats(BaseModel):
    max_abs_diff:              float = Field(ge=0)
    max_rel_diff:              float = Field(ge=0)
    mean_abs_diff:             float = Field(ge=0)
    n_elements_exceeding_tol:  int   = Field(ge=0)
    pct_elements_exceeding_tol: float = Field(ge=0, le=100)


class CorrectnessCheck(BaseModel):
    status:                CorrectnessStatus
    tolerance_policy_used: TolerancePolicy
    vs_eager:              CorrectnessStats
    vs_inductor:           CorrectnessStats


class DatasetCorrectnessCheck(BaseModel):
    status:                CorrectnessStatus
    tolerance_policy_used: TolerancePolicy


class CompileResult(BaseModel):
    status: CompileStatus
    error:  Optional[str]

    @model_validator(mode="after")
    def _error_iff_failed(self) -> CompileResult:
        if self.status == CompileStatus.SUCCESS and self.error is not None:
            raise ValueError("error must be null when compile status is success")
        if self.status == CompileStatus.FAILED and self.error is None:
            raise ValueError("error must be non-null when compile status is failed")
        return self


class Attempt(BaseModel):
    attempt_n:            int = Field(ge=0)
    prior_advice_applied: Optional[str]
    patch_from_previous:  Optional[str] = None
    triton_code:          str
    compile:              CompileResult
    run_error:            Optional[str] = None
    correctness:          Optional[DatasetCorrectnessCheck]
    failure_symptom:      Optional[FailureSymptom] = None
    judge_classification: Optional[JudgeClassification] = None
    judge_root_cause:     Optional[JudgeRootCause] = None
    judge_repair_action:  Optional[JudgeRepairAction] = None
    judge_fix_suggestion: Optional[str]
    timestamp:            datetime

    @model_validator(mode="after")
    def _run_and_correctness_consistency(self) -> Attempt:
        if self.compile.status == CompileStatus.FAILED and self.correctness is not None:
            raise ValueError("correctness must be null when compile failed")
        if self.compile.status == CompileStatus.FAILED and self.run_error is not None:
            raise ValueError("run_error must be null when compile failed")
        if self.compile.status == CompileStatus.SUCCESS:
            if self.correctness is None and self.run_error is None:
                raise ValueError(
                    "compile success requires either correctness or run_error"
                )
            if self.correctness is not None and self.run_error is not None:
                raise ValueError("run_error must be null when correctness is present")
        return self


# ---------------------------------------------------------------------------
# Top-level dataset row (schema/dataset/dataset_record.json shape)
# ---------------------------------------------------------------------------

class DatasetRow(BaseModel):
    dataset_id:    str
    example_id:    str
    source:        Source
    attempts:      list[Attempt] = Field(min_length=1)
    final_outcome: DatasetOutcome

    @model_validator(mode="after")
    def _attempt_indices_are_sequential(self) -> DatasetRow:
        for i, attempt in enumerate(self.attempts):
            if attempt.attempt_n != i:
                raise ValueError(
                    f"attempt_n={attempt.attempt_n} at array index {i}; must match index"
                )
        return self

    @model_validator(mode="after")
    def _source_identity_matches_row(self) -> DatasetRow:
        if self.source.example_id != self.example_id:
            raise ValueError("source.example_id must match row example_id")
        expected = derive_dataset_id(
            source_id=self.example_id,
            op_category=self.source.op_category,
            pytorch_code=self.source.pytorch_code,
            input_shapes=self.source.input_shapes,
            input_dtypes=self.source.input_dtypes,
            rng_seed=self.source.rng_seed,
            tolerance_policy=self.source.tolerance_policy,
        )
        if self.dataset_id != expected:
            raise ValueError("dataset_id must be derived from the full source row")
        return self

    @model_validator(mode="after")
    def _final_outcome_matches_verified_correctness(self) -> DatasetRow:
        any_passed = any(
            attempt.correctness is not None
            and attempt.correctness.status == CorrectnessStatus.PASSED
            for attempt in self.attempts
        )
        any_numeric_failed = any(
            attempt.correctness is not None
            and attempt.correctness.status == CorrectnessStatus.FAILED
            for attempt in self.attempts
        )
        any_runtime_failed = any(attempt.run_error is not None for attempt in self.attempts)

        if self.final_outcome == DatasetOutcome.COMPILED_CORRECT and not any_passed:
            raise ValueError("compiled_correct requires a passed correctness check")
        if self.final_outcome == DatasetOutcome.NUMERIC_FAIL:
            if any_passed or not any_numeric_failed:
                raise ValueError(
                    "numeric_fail requires at least one failed correctness check "
                    "and no passed correctness checks"
                )
        if self.final_outcome == DatasetOutcome.RUNTIME_FAIL:
            if any_passed or any_numeric_failed or not any_runtime_failed:
                raise ValueError(
                    "runtime_fail requires at least one run_error and no "
                    "correctness checks"
                )
        if self.final_outcome == DatasetOutcome.COMPILE_FAIL:
            if any_passed or any_numeric_failed or any_runtime_failed:
                raise ValueError(
                    "compile_fail requires no correctness checks or run errors"
                )
        return self

    @model_validator(mode="after")
    def _attempt_tolerance_matches_source(self) -> DatasetRow:
        for attempt in self.attempts:
            if (
                attempt.correctness is not None
                and attempt.correctness.tolerance_policy_used
                != self.source.tolerance_policy
            ):
                raise ValueError(
                    "attempt correctness tolerance_policy_used must match "
                    "source.tolerance_policy"
                )
        return self


# ===========================================================================
# Eval models — schema/eval/{eval_spec,eval_corpus_record,eval_result}.json
# ===========================================================================
#
# Eval pipeline is single-attempt, no judge: EvalLatency / EvalTokens track
# per-attempt timing and token counts. EvalBenchmark adds the raw 100-iter
# sample arrays needed for Wilcoxon / bootstrap CIs.


class EvalSpec(BaseModel):
    """Stage-1 sampler output — fully-determined fusion spec.
    Matches schema/eval/eval_spec.json."""
    spec_id:                str
    tier:                   Difficulty
    form:                   FormName
    ops:                    list[str] = Field(min_length=2)
    input_shapes:           list[list[int]] = Field(min_length=1)
    input_dtypes:           list[Dtype] = Field(min_length=1)
    expected_output_shape:  list[int] = Field(min_length=1)
    expected_output_dtype:  Dtype
    tolerance_policy:       TolerancePolicy
    rng_seed:               int = Field(ge=0)
    form_metadata:          dict[str, Any]

    @model_validator(mode="after")
    def _shapes_dtypes_length_match(self) -> EvalSpec:
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                f"input_shapes length ({len(self.input_shapes)}) must equal "
                f"input_dtypes length ({len(self.input_dtypes)})"
            )
        return self


class AcceptanceMeta(BaseModel):
    """Provenance for one accepted eval-corpus row at /preflight time."""
    accepted_at:                       datetime
    torch_version:                     str
    triton_version:                    str
    preflight_eager_first_call_ms:     float = Field(ge=0)
    preflight_inductor_first_call_ms:  float = Field(ge=0)


class EvalCorpusRecord(BaseModel):
    """Accepted held-out eval row. Matches schema/eval/eval_corpus_record.json.
    Lives in eval/holdout/synthetic_fusions.jsonl after the stage-1→5 pipeline."""
    example_id:   str
    spec:         EvalSpec
    pytorch_code: str
    origin:       Literal["synthetic/fusion"] = "synthetic/fusion"
    op_category:  OpCategory
    difficulty:   Difficulty
    acceptance:   AcceptanceMeta

    @model_validator(mode="after")
    def _difficulty_matches_spec_tier(self) -> EvalCorpusRecord:
        if self.difficulty != self.spec.tier:
            raise ValueError(
                f"difficulty ({self.difficulty}) must equal spec.tier ({self.spec.tier})"
            )
        return self


class EvalLatency(BaseModel):
    """Per-stage wall times for an eval attempt. No judge_ms — eval is judged-free."""
    generator_ms: int          = Field(ge=0)
    compile_ms:   int          = Field(ge=0)
    run_ms:       Optional[int] = Field(default=None, ge=0)


class EvalTokens(BaseModel):
    """Token counts for an eval attempt. No judge fields — eval is judged-free."""
    generator_prompt:     int = Field(ge=0)
    generator_completion: int = Field(ge=0)


class EvalBenchmark(BaseModel):
    """Timing results with raw 100-iter sample arrays required for
    paired non-parametric tests (Wilcoxon, bootstrap CIs, IQR)."""
    triton_ms:           float = Field(gt=0)
    eager_ms:            float = Field(gt=0)
    inductor_ms:         float = Field(gt=0)
    speedup_vs_eager:    float = Field(gt=0)
    speedup_vs_inductor: float = Field(gt=0)
    triton_std_ms:       float = Field(ge=0)
    eager_std_ms:        float = Field(ge=0)
    inductor_std_ms:     float = Field(ge=0)
    triton_samples_ms:   list[float] = Field(min_length=100, max_length=100)
    eager_samples_ms:    list[float] = Field(min_length=100, max_length=100)
    inductor_samples_ms: list[float] = Field(min_length=100, max_length=100)


class EvalAttempt(BaseModel):
    """Single eval attempt. No judge calls — uses EvalLatency / EvalTokens.
    Benchmark embeds raw 100-iter sample arrays (EvalBenchmark)."""
    attempt_n:            int = Field(ge=0)
    prior_advice_applied: Optional[str]
    triton_code:          str
    compile:              CompileResult
    run_error:            Optional[str] = None
    correctness:          Optional[CorrectnessCheck]
    benchmark:            Optional[EvalBenchmark]
    latency:              EvalLatency
    tokens:               EvalTokens
    timestamp:            datetime

    @model_validator(mode="after")
    def _run_and_correctness_consistency(self) -> EvalAttempt:
        if self.compile.status == CompileStatus.FAILED and self.correctness is not None:
            raise ValueError("correctness must be null when compile failed")
        if self.compile.status == CompileStatus.FAILED and self.run_error is not None:
            raise ValueError("run_error must be null when compile failed")
        if self.compile.status == CompileStatus.SUCCESS:
            if self.correctness is None and self.run_error is None:
                raise ValueError("compile success requires either correctness or run_error")
            if self.correctness is not None and self.run_error is not None:
                raise ValueError("run_error must be null when correctness is present")
        return self

    @model_validator(mode="after")
    def _benchmark_requires_correctness_passed(self) -> EvalAttempt:
        if self.benchmark is not None:
            if self.correctness is None:
                raise ValueError("benchmark requires correctness to be non-null")
            if self.correctness.status == CorrectnessStatus.FAILED:
                raise ValueError("benchmark must be null when correctness failed")
        return self


class EvalRecord(BaseModel):
    """Per-example result of one eval run. Matches schema/eval/eval_result.json.
    Written to eval/results/<model_label>/eval_rows.jsonl. The model_label
    is the parent directory; it is not stored in the record itself."""
    example_id:              str
    run_id:                  str
    spec:                    EvalSpec
    attempts:                list[EvalAttempt] = Field(min_length=1)
    final_outcome:           EvalFinalOutcome
    final_winning_attempt_n: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _attempt_indices_are_sequential(self) -> EvalRecord:
        for i, attempt in enumerate(self.attempts):
            if attempt.attempt_n != i:
                raise ValueError(
                    f"attempt_n={attempt.attempt_n} at array index {i}; must match index"
                )
        return self

    @model_validator(mode="after")
    def _winning_attempt_consistent_with_outcome(self) -> EvalRecord:
        terminal_without_winner = {
            EvalFinalOutcome.CORRECTNESS_FAILED,
            EvalFinalOutcome.RUNTIME_FAIL,
            EvalFinalOutcome.ALL_ATTEMPTS_FAILED,
        }
        if self.final_outcome in terminal_without_winner:
            if self.final_winning_attempt_n is not None:
                raise ValueError(
                    f"final_winning_attempt_n must be null for outcome '{self.final_outcome}'"
                )
        else:
            if self.final_winning_attempt_n is None:
                raise ValueError(
                    f"final_winning_attempt_n must be non-null for outcome '{self.final_outcome}'"
                )
        return self
