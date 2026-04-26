"""
Single source of truth for all correctness tolerance values.
Fixed before data collection begins. Do not change during the experiment.
See docs/tolerance-policy.md.
"""
from enum import Enum
from dataclasses import dataclass


class TolerancePolicy(str, Enum):
    DEFAULT_FP32   = "default_fp32"
    DEFAULT_FP16   = "default_fp16"
    REDUCTION_FP32 = "reduction_fp32"
    REDUCTION_FP16 = "reduction_fp16"


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


_POLICIES: dict[TolerancePolicy, Tolerance] = {
    TolerancePolicy.DEFAULT_FP32:   Tolerance(atol=1e-5, rtol=1e-5),
    TolerancePolicy.DEFAULT_FP16:   Tolerance(atol=1e-3, rtol=1e-3),
    TolerancePolicy.REDUCTION_FP32: Tolerance(atol=1e-4, rtol=1e-4),
    TolerancePolicy.REDUCTION_FP16: Tolerance(atol=5e-3, rtol=5e-3),
}


def get(policy: TolerancePolicy) -> Tolerance:
    return _POLICIES[policy]
