"""
Closed-vocabulary enums for all categorical fields in the dataset.
"""
from enum import Enum

from shared.verification.tolerance import TolerancePolicy

__all__ = [
    "OpCategory",
    "Dtype",
    "Split",
    "Difficulty",
    "FinalOutcome",
    "JudgeClassification",
    "CompileStatus",
    "CorrectnessStatus",
    "TolerancePolicy",
]


class OpCategory(str, Enum):
    ELEMENTWISE_CHAIN = "elementwise_chain"
    REDUCTION         = "reduction"
    MATMUL            = "matmul"
    CONVOLUTION       = "convolution"
    FUSED_ATTENTION   = "fused_attention"
    NORMALIZATION     = "normalization"
    ACTIVATION        = "activation"
    LOSS              = "loss"
    EMBEDDING         = "embedding"
    QUANTIZATION      = "quantization"
    OTHER             = "other"


class Dtype(str, Enum):
    FLOAT16  = "float16"
    FLOAT32  = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT64  = "float64"
    INT8     = "int8"
    INT16    = "int16"
    INT32    = "int32"
    INT64    = "int64"
    BOOL     = "bool"


class Split(str, Enum):
    TRAIN = "train"
    EVAL  = "eval"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class FinalOutcome(str, Enum):
    CORRECTNESS_FAILED                    = "correctness_failed"
    COMPILED_CORRECT_SLOW                 = "compiled_correct_slow"
    COMPILED_CORRECT_PARITY               = "compiled_correct_parity"
    COMPILED_CORRECT_FASTER_THAN_INDUCTOR = "compiled_correct_faster_than_inductor"
    ALL_ATTEMPTS_FAILED                   = "all_attempts_failed"


class JudgeClassification(str, Enum):
    SHAPE_MISMATCH                   = "shape_mismatch"
    DTYPE_MISMATCH                   = "dtype_mismatch"
    INDEXING_ERROR                   = "indexing_error"
    CORRECTNESS_FAILED_NUMERIC       = "correctness_failed_numeric"
    CORRECT_BUT_SLOWER_THAN_INDUCTOR = "correct_but_slower_than_inductor"
    CORRECT_AND_COMPETITIVE          = "correct_and_competitive"
    CORRECT_AND_FASTER               = "correct_and_faster"
    AMBIGUOUS                        = "ambiguous"
    OTHER                            = "other"


class CompileStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"


class CorrectnessStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
