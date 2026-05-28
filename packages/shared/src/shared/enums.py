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
    "DatasetOutcome",
    "EvalFinalOutcome",
    "FailureSymptom",
    "JudgeClassification",
    "JudgeRootCause",
    "JudgeRepairAction",
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


class DatasetOutcome(str, Enum):
    """Top-level outcome for a training dataset row.
    compiled_correct = at least one attempt compiled and passed numerics.
    numeric_fail     = at least one attempt produced correctness stats but none passed.
    runtime_fail     = at least one attempt compiled but failed before correctness stats.
    compile_fail     = no attempt passed the compile endpoint.
    """
    COMPILED_CORRECT = "compiled_correct"
    NUMERIC_FAIL     = "numeric_fail"
    RUNTIME_FAIL     = "runtime_fail"
    COMPILE_FAIL     = "compile_fail"


class EvalFinalOutcome(str, Enum):
    CORRECTNESS_FAILED                    = "correctness_failed"
    RUNTIME_FAIL                          = "runtime_fail"
    COMPILE_FAIL                          = "compile_fail"
    COMPILED_CORRECT_SLOW                 = "compiled_correct_slow"
    COMPILED_CORRECT_PARITY               = "compiled_correct_parity"
    COMPILED_CORRECT_FASTER_THAN_INDUCTOR = "compiled_correct_faster_than_inductor"


class FailureSymptom(str, Enum):
    NUMERIC_MISMATCH              = "numeric_mismatch"
    UNSUPPORTED_PTR_TYPE          = "unsupported_ptr_type"
    INCOMPATIBLE_BLOCK_SHAPE      = "incompatible_block_shape"
    NON_CONSTEXPR_ARANGE_BOUND    = "non_constexpr_arange_bound"
    UNDEFINED_SYMBOL              = "undefined_symbol"
    UNSUPPORTED_LANGUAGE_CONSTRUCT = "unsupported_language_construct"
    LAUNCH_SIGNATURE_MISMATCH     = "launch_signature_mismatch"
    INVALID_TRITON_API            = "invalid_triton_api"
    INVALID_OUTPUT_TYPE           = "invalid_output_type"
    HOST_SHAPE_ERROR              = "host_shape_error"
    CUDA_ILLEGAL_MEMORY_ACCESS    = "cuda_illegal_memory_access"
    TIMEOUT                       = "timeout"
    TRITON_COMPILATION_ERROR      = "triton_compilation_error"
    PYTHON_RUNTIME_ERROR          = "python_runtime_error"


class JudgeClassification(str, Enum):
    SHAPE_MISMATCH             = "shape_mismatch"
    DTYPE_MISMATCH             = "dtype_mismatch"
    INDEXING_ERROR             = "indexing_error"
    TRITON_API_ERROR           = "triton_api_error"
    CORRECTNESS_FAILED_NUMERIC = "correctness_failed_numeric"
    COMPILED_CORRECT           = "compiled_correct"
    OTHER                      = "other"


class JudgeRootCause(str, Enum):
    RAW_DATA_PTR_PASSED_AS_POINTER = "raw_data_ptr_passed_as_pointer"
    INVALID_POINTER_ARITHMETIC_OR_OFFSET_DTYPE = (
        "invalid_pointer_arithmetic_or_offset_dtype"
    )
    RUNTIME_VALUE_USED_AS_CONSTEXPR = "runtime_value_used_as_constexpr"
    BROADCAST_OR_MASK_RANK_MISMATCH = "broadcast_or_mask_rank_mismatch"
    WRONG_LAUNCH_GRID_OR_META_SIGNATURE = "wrong_launch_grid_or_meta_signature"
    WRONG_PROGRAM_ID_MAPPING = "wrong_program_id_mapping"
    WRONG_LINEARIZATION_OR_STRIDE = "wrong_linearization_or_stride"
    WRONG_REDUCTION_AXIS_OR_EXTENT = "wrong_reduction_axis_or_extent"
    WRONG_MATMUL_TILE_OR_ACCUMULATION = "wrong_matmul_tile_or_accumulation"
    DTYPE_CAST_OR_DOT_OPERAND_MISMATCH = "dtype_cast_or_dot_operand_mismatch"
    WRONG_OUTPUT_SHAPE_OR_RETURN_TYPE = "wrong_output_shape_or_return_type"
    INVALID_TRITON_API_USAGE = "invalid_triton_api_usage"
    HOST_WRAPPER_SHAPE_LOGIC_ERROR = "host_wrapper_shape_logic_error"
    MEMORY_SAFETY_OR_OUT_OF_BOUNDS = "memory_safety_or_out_of_bounds"
    UNSUPPORTED_PYTHON_CONSTRUCT_IN_KERNEL = (
        "unsupported_python_construct_in_kernel"
    )
    UNKNOWN = "unknown"


class JudgeRepairAction(str, Enum):
    REPLACE_DATA_PTR_WITH_TENSOR_ARGS = "replace_data_ptr_with_tensor_args"
    CAST_OFFSETS_OR_STRIDES_TO_INT32 = "cast_offsets_or_strides_to_int32"
    MAKE_DIMENSION_CONSTEXPR_OR_TILE = "make_dimension_constexpr_or_tile"
    FIX_MASK_BROADCAST_RANK = "fix_mask_broadcast_rank"
    FIX_LAUNCH_GRID_OR_META_ARGS = "fix_launch_grid_or_meta_args"
    FIX_PROGRAM_ID_MAPPING = "fix_program_id_mapping"
    FIX_LINEAR_INDEXING_OR_STRIDES = "fix_linear_indexing_or_strides"
    FIX_REDUCTION_AXIS_OR_LOOP_EXTENT = "fix_reduction_axis_or_loop_extent"
    FIX_MATMUL_TILING_OR_ACCUMULATOR = "fix_matmul_tiling_or_accumulator"
    ADD_OR_CHANGE_DTYPE_CAST = "add_or_change_dtype_cast"
    FIX_OUTPUT_ALLOCATION_SHAPE_OR_RETURN = (
        "fix_output_allocation_shape_or_return"
    )
    REPLACE_INVALID_TRITON_API = "replace_invalid_triton_api"
    FIX_HOST_WRAPPER_SHAPE_LOGIC = "fix_host_wrapper_shape_logic"
    ADD_BOUNDS_MASKING = "add_bounds_masking"
    REMOVE_UNSUPPORTED_KERNEL_CONSTRUCT = "remove_unsupported_kernel_construct"
    UNKNOWN = "unknown"


class CompileStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"


class CorrectnessStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
