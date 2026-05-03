"""
HTTP contract models for the MacBook ↔ Lenovo verification API.
Wire format is defined in packages/shared/src/shared/schema/verification_api.json.
Both the FastAPI server (Lenovo) and the orchestrator HTTP client (MacBook) import from here.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from shared.enums import CompileStatus, CorrectnessStatus, Dtype, TolerancePolicy
from shared.models import CorrectnessStats

__all__ = [
    "PreflightRequest",
    "CompileRequest",
    "RunRequest",
    "BenchmarkRequest",
    "PreflightResponse",
    "CompileResponse",
    "RunResponse",
    "BenchmarkResponse",
    "JobAccepted",
    "JobStatus",
    "JobStatusValue",
]


class JobStatusValue(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class PreflightRequest(BaseModel):
    pytorch_code:     str
    input_shapes:     list[list[int]] = Field(min_length=1)
    input_dtypes:     list[Dtype]
    rng_seed:         int             = Field(ge=0)
    tolerance_policy: TolerancePolicy


class CompileRequest(BaseModel):
    triton_code: str


class RunRequest(BaseModel):
    triton_code:      str
    pytorch_code:     str
    input_shapes:     list[list[int]] = Field(min_length=1)
    input_dtypes:     list[Dtype]
    rng_seed:         int             = Field(ge=0)
    tolerance_policy: TolerancePolicy


class BenchmarkRequest(BaseModel):
    triton_code:  str
    pytorch_code: str
    input_shapes: list[list[int]] = Field(min_length=1)
    input_dtypes: list[Dtype]
    rng_seed:     int             = Field(ge=0)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class PreflightResponse(BaseModel):
    passed:            bool
    vs_eager_inductor: Optional[CorrectnessStats] = None
    error_message:     Optional[str] = None


class CompileResponse(BaseModel):
    status:        CompileStatus
    error_message: Optional[str] = None


class RunResponse(BaseModel):
    correctness_status:    Optional[CorrectnessStatus] = None
    tolerance_policy_used: Optional[TolerancePolicy] = None
    vs_eager:              Optional[CorrectnessStats] = None
    vs_inductor:           Optional[CorrectnessStats] = None
    error_message:         Optional[str] = None


class BenchmarkResponse(BaseModel):
    triton_ms:           float = Field(gt=0)
    eager_ms:            float = Field(gt=0)
    inductor_ms:         float = Field(gt=0)
    speedup_vs_eager:    float = Field(gt=0)
    speedup_vs_inductor: float = Field(gt=0)
    triton_std_ms:       float = Field(ge=0)
    eager_std_ms:        float = Field(ge=0)
    inductor_std_ms:     float = Field(ge=0)


class JobAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status:        JobStatusValue
    result:        Optional[BenchmarkResponse] = None
    error_message: Optional[str] = None
