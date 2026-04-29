"""
Single source of truth for all correctness tolerance values.
Fixed before data collection begins. Do not change during the experiment.
See docs/tolerance-policy.md.
"""
from enum import Enum
from dataclasses import dataclass


class ComparisonMode(str, Enum):
    NUMERIC = "numeric"
    EXACT = "exact"
    INF_AWARE_NUMERIC = "inf_aware_numeric"


class TolerancePolicy(str, Enum):
    DEFAULT_FP32           = "default_fp32"
    DEFAULT_FP16           = "default_fp16"
    REDUCTION_FP32         = "reduction_fp32"
    REDUCTION_FP16         = "reduction_fp16"
    EXACT_INTEGER          = "exact_integer"
    MASKED_LOGITS          = "masked_logits"
    ATTENTION_SOFTMAX_FP16 = "attention_softmax_fp16"
    RECURRENT_SCAN_FP16    = "recurrent_scan_fp16"
    LOW_PRECISION_DEQUANT  = "low_precision_dequant"
    FP8_CAST               = "fp8_cast"


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    comparison: ComparisonMode = ComparisonMode.NUMERIC


_POLICIES: dict[TolerancePolicy, Tolerance] = {
    TolerancePolicy.DEFAULT_FP32:           Tolerance(atol=1e-5, rtol=1e-5),
    TolerancePolicy.DEFAULT_FP16:           Tolerance(atol=1e-3, rtol=1e-3),
    TolerancePolicy.REDUCTION_FP32:         Tolerance(atol=1e-4, rtol=1e-4),
    TolerancePolicy.REDUCTION_FP16:         Tolerance(atol=5e-3, rtol=5e-3),
    TolerancePolicy.EXACT_INTEGER:          Tolerance(
        atol=0.0,
        rtol=0.0,
        comparison=ComparisonMode.EXACT,
    ),
    TolerancePolicy.MASKED_LOGITS:          Tolerance(
        atol=1e-5,
        rtol=1e-5,
        comparison=ComparisonMode.INF_AWARE_NUMERIC,
    ),
    TolerancePolicy.ATTENTION_SOFTMAX_FP16: Tolerance(atol=5e-3, rtol=5e-3),
    TolerancePolicy.RECURRENT_SCAN_FP16:    Tolerance(atol=1e-2, rtol=1e-2),
    TolerancePolicy.LOW_PRECISION_DEQUANT:  Tolerance(atol=2e-2, rtol=2e-2),
    TolerancePolicy.FP8_CAST:               Tolerance(atol=1e-2, rtol=1e-2),
}


def get(policy: TolerancePolicy) -> Tolerance:
    return _POLICIES[policy]
