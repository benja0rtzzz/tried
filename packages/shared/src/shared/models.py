"""
Pydantic v2 models for all record types.
CorpusRecord maps to eval_and_training.json (what the agent loop reads).
DatasetRow maps to dataset_record.json (what the orchestrator writes).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from shared.enums import (
    CompileStatus,
    CorrectnessStatus,
    Difficulty,
    Dtype,
    FinalOutcome,
    JudgeClassification,
    OpCategory,
    Split,
    TolerancePolicy,
)

__all__ = [
    "CorpusRecord",
    "Source",
    "CorrectnessStats",
    "CorrectnessCheck",
    "Benchmark",
    "Latency",
    "Tokens",
    "Attempt",
    "DatasetRow",
]


# ---------------------------------------------------------------------------
# Corpus record (eval_and_training.json shape)
# ---------------------------------------------------------------------------

class CorpusRecord(BaseModel):
    example_id:   str
    split:        Split
    origin:       str
    op_category:  OpCategory
    difficulty:   Optional[Difficulty]
    pytorch_code: str
    input_shapes: list[list[int]]
    input_dtypes: list[Dtype]
    rng_seed:     int = Field(ge=0)

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


# ---------------------------------------------------------------------------
# Sub-models for DatasetRow
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """Immutable inputs copied verbatim from the corpus record when the agent loop starts."""
    pytorch_code: str
    origin:       str
    input_shapes: list[list[int]]
    input_dtypes: list[Dtype]
    rng_seed:     int = Field(ge=0)
    op_category:  OpCategory

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


class Benchmark(BaseModel):
    triton_ms:           float = Field(gt=0)
    eager_ms:            float = Field(gt=0)
    inductor_ms:         float = Field(gt=0)
    speedup_vs_eager:    float = Field(gt=0)
    speedup_vs_inductor: float = Field(gt=0)
    triton_std_ms:       float = Field(ge=0)
    eager_std_ms:        float = Field(ge=0)
    inductor_std_ms:     float = Field(ge=0)


class Latency(BaseModel):
    generator_ms: int          = Field(ge=0)
    judge_ms:     int          = Field(ge=0)
    compile_ms:   int          = Field(ge=0)
    run_ms:       Optional[int] = Field(default=None, ge=0)


class Tokens(BaseModel):
    generator_prompt:     int = Field(ge=0)
    generator_completion: int = Field(ge=0)
    judge_prompt:         int = Field(ge=0)
    judge_completion:     int = Field(ge=0)


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
    attempt_n:           int = Field(ge=0)
    prior_advice_applied: Optional[str]
    triton_code:         str
    compile:             CompileResult
    correctness:         Optional[CorrectnessCheck]
    benchmark:           Optional[Benchmark]
    judge_classification: JudgeClassification
    judge_fix_suggestion: Optional[str]
    latency:             Latency
    tokens:              Tokens
    timestamp:           datetime

    @model_validator(mode="after")
    def _correctness_requires_compile_success(self) -> Attempt:
        if self.compile.status == CompileStatus.FAILED and self.correctness is not None:
            raise ValueError("correctness must be null when compile failed")
        return self

    @model_validator(mode="after")
    def _benchmark_requires_correctness_passed(self) -> Attempt:
        if self.benchmark is not None:
            if self.correctness is None:
                raise ValueError("benchmark requires correctness to be non-null")
            if self.correctness.status == CorrectnessStatus.FAILED:
                raise ValueError("benchmark must be null when correctness failed")
        return self


# ---------------------------------------------------------------------------
# Top-level dataset row (dataset_record.json shape)
# ---------------------------------------------------------------------------

class DatasetRow(BaseModel):
    example_id:              str
    source:                  Source
    attempts:                list[Attempt] = Field(min_length=1)
    final_outcome:           FinalOutcome
    final_winning_attempt_n: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _attempt_indices_are_sequential(self) -> DatasetRow:
        for i, attempt in enumerate(self.attempts):
            if attempt.attempt_n != i:
                raise ValueError(
                    f"attempt_n={attempt.attempt_n} at array index {i}; must match index"
                )
        return self

    @model_validator(mode="after")
    def _winning_attempt_consistent_with_outcome(self) -> DatasetRow:
        terminal_without_winner = {
            FinalOutcome.COMPILE_FAILED,
            FinalOutcome.CORRECTNESS_FAILED,
            FinalOutcome.ALL_ATTEMPTS_FAILED,
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
