"""Structural pattern enums + SkeletonSpec.

Initial enum values are educated guesses informed by the existing corpus_train
data; discovery output may suggest additions or flag values with zero observed
frequency. Adding or removing a value requires a decision-log entry, same rule
as `OpCategory`.

A SkeletonSpec is the fully-determined input to the codex prompt: one row of
the cross-product over (op_category, shape_rank, dtype_mix, broadcast,
reduction, fusion, memory) plus suggested input shapes/dtypes. spec_id is a
UUIDv5 over the canonical key so re-running the sampler is idempotent.
"""
from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field
from shared.enums import Dtype, OpCategory


class ShapeRank(str, Enum):
    D1 = "1D"
    D2 = "2D"
    D3 = "3D"
    D4 = "4D"


class DtypeMix(str, Enum):
    FP32_ONLY        = "fp32_only"
    FP16_ONLY        = "fp16_only"
    BF16_ONLY        = "bf16_only"
    MIXED_FP32_FP16  = "mixed_fp32_fp16"
    MIXED_FP32_BF16  = "mixed_fp32_bf16"
    WITH_INT8        = "with_int8"


class BroadcastPattern(str, Enum):
    NONE     = "none"
    ROW      = "row"
    COL      = "col"
    CHANNEL  = "channel"
    BATCH    = "batch"


class ReductionAxis(str, Enum):
    NONE     = "none"
    LAST     = "last"
    ALL      = "all"
    CHANNEL  = "channel"
    BATCH    = "batch"


class FusionShape(str, Enum):
    SINGLE_OP        = "single_op"
    PAIR             = "pair"
    TRIPLET          = "triplet"
    REDUCE_THEN_OP   = "reduce_then_op"
    OP_THEN_REDUCE   = "op_then_reduce"


class MemoryPattern(str, Enum):
    CONTIGUOUS  = "contiguous"
    STRIDED     = "strided"
    JAGGED      = "jagged"
    MASKED      = "masked"


# Stable namespace for spec_id UUIDv5 derivation (TRIED corpus_gen).
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000c01")


class SkeletonSpec(BaseModel):
    """Fully-determined input to the codex skeleton prompt.

    Cross-product axes are categorical; suggested shapes/dtypes are concrete
    so the prompt and the downstream preflight see the same values. spec_id
    is derived deterministically from the canonical fields.
    """
    spec_id: str

    op_category:       OpCategory
    shape_rank:        ShapeRank
    dtype_mix:         DtypeMix
    broadcast_pattern: BroadcastPattern
    reduction_axis:    ReductionAxis
    fusion_shape:      FusionShape
    memory_pattern:    MemoryPattern

    suggested_input_shapes: list[list[int]] = Field(min_length=1, max_length=4)
    suggested_input_dtypes: list[Dtype]     = Field(min_length=1, max_length=4)

    rng_seed: int = Field(ge=0)


def derive_spec_id(
    op_category: OpCategory,
    shape_rank: ShapeRank,
    dtype_mix: DtypeMix,
    broadcast_pattern: BroadcastPattern,
    reduction_axis: ReductionAxis,
    fusion_shape: FusionShape,
    memory_pattern: MemoryPattern,
    suggested_input_shapes: list[list[int]],
    suggested_input_dtypes: list[Dtype],
) -> str:
    """Canonical UUIDv5 over the spec's content. Same content → same spec_id."""
    key = "|".join([
        op_category.value,
        shape_rank.value,
        dtype_mix.value,
        broadcast_pattern.value,
        reduction_axis.value,
        fusion_shape.value,
        memory_pattern.value,
        ",".join("x".join(str(d) for d in s) for s in suggested_input_shapes),
        ",".join(d.value for d in suggested_input_dtypes),
    ])
    return str(uuid.uuid5(_NAMESPACE, key))
