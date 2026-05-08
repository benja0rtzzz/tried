"""Locked registry of fusion forms for the eval-set generation pipeline.

Each Form bundles every load-bearing decision for one fusion shape:
  - tier (easy / medium / hard) and op_category for the dataset schema
  - op_count and op_pool — what the stage-1 sampler may pick from
  - required_metadata_keys — the form_metadata keys stage-1 must populate
  - dtypes and shape_grid — the design grid the sampler iterates over
  - output_shape, output_dtype, tolerance_policy — pure functions called by
    stage-1 to fill the EvalSpec; same functions are used by stage-3/4 to
    cross-check the candidate's actual output
  - prompt_block — the form-specific section appended to the locked stage-2
    prompt template, rendered with .format(**spec_kwargs)

LOCKED. Edits to op_pool, op_count, dtypes, shape_grid, or any of the four
output / tolerance / prompt callables change the eval-set content. See
docs/decision-log.md and docs/corpus.md.

Total: 11 forms — 2 Easy, 4 Medium, 5 Hard.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shared.enums import Difficulty, Dtype, OpCategory, TolerancePolicy

from .base_ops import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    MATMUL,
    NORMALIZATION,
    REDUCTION,
)


# ---------------------------------------------------------------------------
# Shape / dtype / tolerance helpers — shared by multiple forms
# ---------------------------------------------------------------------------

ShapeFn = Callable[[dict, list[list[int]], list[str]], list[int]]
DtypeFn = Callable[[dict, list[str], list[str]], str]
ToleranceFn = Callable[[str, list[str]], str]


def _shape_preserving(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    return list(ishapes[0])


def _drop_reduce_dim(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    s = list(ishapes[0])
    dim = fm["reduce_dim"]
    if dim < 0:
        dim += len(s)
    if fm.get("keepdim", False):
        s[dim] = 1
        return s
    s.pop(dim)
    return s


def _linear_output(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    s = list(ishapes[0])
    s[-1] = fm["weight_shape"][0]  # weight is [D_out, D_in]
    return s


def _gated_mlp_output(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    s = list(ishapes[0])
    s[-1] = fm["weight_shapes"][0][0]
    return s


def _embedding_output(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    # input_shapes[0] is the index tensor [B, T]; output is [B, T, embedding_dim]
    return list(ishapes[0]) + [fm["embedding_dim"]]


def _attention_output(fm: dict, ishapes: list[list[int]], ops: list[str]) -> list[int]:
    # Q, K, V share the same shape; output matches Q (input_shapes[0]).
    return list(ishapes[0])


def _input_dtype(fm: dict, idtypes: list[str], ops: list[str]) -> str:
    return idtypes[0]


def _maybe_argmax_dtype(fm: dict, idtypes: list[str], ops: list[str]) -> str:
    return Dtype.INT64.value if "torch.argmax" in ops else idtypes[0]


def _embedding_table_dtype(fm: dict, idtypes: list[str], ops: list[str]) -> str:
    # idtypes[0] is the index tensor (int64); idtypes[1] is the weight (float).
    return idtypes[1]


def _default_tol(output_dtype: str, ops: list[str]) -> str:
    if output_dtype == Dtype.INT64.value:
        return TolerancePolicy.EXACT_INTEGER.value
    if output_dtype == Dtype.FLOAT16.value:
        return TolerancePolicy.DEFAULT_FP16.value
    return TolerancePolicy.DEFAULT_FP32.value


def _reduction_tol(output_dtype: str, ops: list[str]) -> str:
    if output_dtype == Dtype.INT64.value:  # argmax
        return TolerancePolicy.EXACT_INTEGER.value
    if output_dtype == Dtype.FLOAT16.value:
        return TolerancePolicy.REDUCTION_FP16.value
    return TolerancePolicy.REDUCTION_FP32.value


def _attention_tol(output_dtype: str, ops: list[str]) -> str:
    if output_dtype == Dtype.FLOAT16.value:
        return TolerancePolicy.ATTENTION_SOFTMAX_FP16.value
    return TolerancePolicy.DEFAULT_FP32.value


# ---------------------------------------------------------------------------
# Form dataclass and registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Form:
    name: str
    tier: Difficulty
    op_category: OpCategory
    op_count: int
    op_pool: tuple[frozenset[str], ...]
    required_metadata_keys: tuple[str, ...]
    dtypes: tuple[Dtype, ...]
    shape_grid: tuple[tuple[tuple[int, ...], ...], ...]  # outer: variants; mid: per-input shapes; inner: dims
    output_shape: ShapeFn
    output_dtype: DtypeFn
    tolerance_policy: ToleranceFn
    prompt_block: str  # rendered via .format(**spec_kwargs) by stage-2


# Helper: a single element of shape_grid is one (sampler-pickable) value of
# input_shapes — a list of per-input tensor shapes. We store as nested tuples
# for hashability; the sampler converts to list[list[int]] when filling EvalSpec.

# ---------------------------------------------------------------------------
# Form 1 — chain_2_unary (Easy)
# ---------------------------------------------------------------------------
chain_2_unary = Form(
    name="chain_2_unary",
    tier=Difficulty.EASY,
    op_category=OpCategory.ELEMENTWISE_CHAIN,
    op_count=2,
    op_pool=(ELEMENTWISE_UNARY, ELEMENTWISE_UNARY),
    required_metadata_keys=(),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((1024,),),
        ((4096,),),
        ((16384,),),
        ((32, 1024),),
        ((64, 2048),),
        ((128, 4096),),
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: chain_2_unary. Apply two unary ops in this exact order: "
        "first {ops[0]}, then {ops[1]} on the result. The input is a single "
        "tensor of shape {input_shapes[0]} and dtype {input_dtypes[0]}. "
        "Output preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 2 — unary_then_residual (Easy)
# ---------------------------------------------------------------------------
unary_then_residual = Form(
    name="unary_then_residual",
    tier=Difficulty.EASY,
    op_category=OpCategory.ELEMENTWISE_CHAIN,
    op_count=2,
    op_pool=(ELEMENTWISE_UNARY, frozenset({"torch.add"})),
    required_metadata_keys=(),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((32, 1024),),
        ((64, 2048),),
        ((128, 4096),),
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: unary_then_residual. Compute torch.add({ops[0]}(x), x) — apply "
        "the unary op {ops[0]} to x, then add the original x as a residual. "
        "Input is a single tensor of shape {input_shapes[0]} and dtype "
        "{input_dtypes[0]}. Output preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 3 — chain_3_unary (Medium)
# ---------------------------------------------------------------------------
chain_3_unary = Form(
    name="chain_3_unary",
    tier=Difficulty.MEDIUM,
    op_category=OpCategory.ELEMENTWISE_CHAIN,
    op_count=3,
    op_pool=(ELEMENTWISE_UNARY, ELEMENTWISE_UNARY, ELEMENTWISE_UNARY),
    required_metadata_keys=(),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((32, 1024),),
        ((64, 2048),),
        ((128, 4096),),
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: chain_3_unary. Apply three unary ops in this exact order: "
        "first {ops[0]}, then {ops[1]}, then {ops[2]}. Input is a single "
        "tensor of shape {input_shapes[0]} and dtype {input_dtypes[0]}. "
        "Output preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 4 — unary_then_reduction (Medium)
# ---------------------------------------------------------------------------
unary_then_reduction = Form(
    name="unary_then_reduction",
    tier=Difficulty.MEDIUM,
    op_category=OpCategory.REDUCTION,
    op_count=2,
    op_pool=(ELEMENTWISE_UNARY, REDUCTION),
    required_metadata_keys=("reduce_dim", "keepdim"),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((32, 1024),),
        ((128, 4096),),
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_drop_reduce_dim,
    output_dtype=_maybe_argmax_dtype,
    tolerance_policy=_reduction_tol,
    prompt_block=(
        "Form: unary_then_reduction. Compute {ops[1]}({ops[0]}(x), "
        "dim={form_metadata[reduce_dim]}, keepdim={form_metadata[keepdim]}). "
        "Input is a single tensor of shape {input_shapes[0]} and dtype "
        "{input_dtypes[0]}. Expected output shape: {expected_output_shape}, "
        "dtype: {expected_output_dtype}."
    ),
)


# ---------------------------------------------------------------------------
# Form 5 — softmax_then_unary (Medium)
# ---------------------------------------------------------------------------
softmax_then_unary = Form(
    name="softmax_then_unary",
    tier=Difficulty.MEDIUM,
    op_category=OpCategory.NORMALIZATION,
    op_count=2,
    op_pool=(frozenset({"torch.nn.functional.softmax"}), ELEMENTWISE_UNARY),
    required_metadata_keys=("softmax_dim",),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((32, 1024),),
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((4, 8, 128, 128),),  # attention-shaped scores
        ((4, 16, 256, 256),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_attention_tol,
    prompt_block=(
        "Form: softmax_then_unary. Compute {ops[1]}({ops[0]}(x, "
        "dim={form_metadata[softmax_dim]})). Input is a single tensor of "
        "shape {input_shapes[0]} and dtype {input_dtypes[0]}. Output "
        "preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 6 — unary_then_norm (Medium)
# ---------------------------------------------------------------------------
unary_then_norm = Form(
    name="unary_then_norm",
    tier=Difficulty.MEDIUM,
    op_category=OpCategory.NORMALIZATION,
    op_count=2,
    op_pool=(ELEMENTWISE_UNARY, frozenset({"torch.nn.functional.layer_norm"})),
    required_metadata_keys=("normalized_shape", "norm_eps"),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: unary_then_norm. Compute torch.nn.functional.layer_norm("
        "{ops[0]}(x), normalized_shape={form_metadata[normalized_shape]}, "
        "eps={form_metadata[norm_eps]}). Pass weight=None and bias=None to "
        "layer_norm (no learnable parameters). Input is a single tensor of "
        "shape {input_shapes[0]} and dtype {input_dtypes[0]}. Output "
        "preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 7 — attention_qkv (Hard)
# ---------------------------------------------------------------------------
attention_qkv = Form(
    name="attention_qkv",
    tier=Difficulty.HARD,
    op_category=OpCategory.FUSED_ATTENTION,
    op_count=4,
    op_pool=(
        frozenset({"torch.matmul"}),
        frozenset({"torch.mul"}),
        frozenset({"torch.nn.functional.softmax"}),
        frozenset({"torch.matmul"}),
    ),
    required_metadata_keys=("num_heads", "causal", "scale"),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        # Q, K, V all share shape [B, H, T, D_h].
        (((2, 4, 64, 64), (2, 4, 64, 64), (2, 4, 64, 64))),
        (((2, 8, 128, 64), (2, 8, 128, 64), (2, 8, 128, 64))),
        (((4, 8, 256, 64), (4, 8, 256, 64), (4, 8, 256, 64))),
        (((4, 16, 128, 128), (4, 16, 128, 128), (4, 16, 128, 128))),
    ),
    output_shape=_attention_output,
    output_dtype=_input_dtype,
    tolerance_policy=_attention_tol,
    prompt_block=(
        "Form: attention_qkv. Compute scaled dot-product attention. Inputs "
        "are Q, K, V each of shape {input_shapes[0]} and dtype "
        "{input_dtypes[0]}. Compute scores = torch.matmul(Q, K.transpose(-2, -1)); "
        "scaled = torch.mul(scores, {form_metadata[scale]}); "
        "weights = torch.nn.functional.softmax(scaled, dim=-1); "
        "out = torch.matmul(weights, V). num_heads={form_metadata[num_heads]} "
        "is implicit in the shape (dim 1). causal={form_metadata[causal]} "
        "(if true, mask the upper triangle to -inf BEFORE softmax — but "
        "do this via tensor indexing assignment, not a new op). Output "
        "shape matches Q."
    ),
)


# ---------------------------------------------------------------------------
# Form 8 — fused_linear_norm_activation (Hard)
# ---------------------------------------------------------------------------
fused_linear_norm_activation = Form(
    name="fused_linear_norm_activation",
    tier=Difficulty.HARD,
    op_category=OpCategory.FUSED_ATTENTION,  # subblock of a transformer block
    op_count=3,
    op_pool=(
        frozenset({"torch.nn.functional.linear"}),
        frozenset({"torch.nn.functional.layer_norm"}),
        ELEMENTWISE_UNARY,
    ),
    required_metadata_keys=("weight_shape", "norm_eps"),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        # input shape [B, T, D_in]; weight is form_metadata.weight_shape = [D_out, D_in]
        (((4, 128, 256),),),
        (((4, 256, 512),),),
        (((8, 128, 768),),),
        (((8, 256, 1024),),),
    ),
    output_shape=_linear_output,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: fused_linear_norm_activation. Compute {ops[2]}("
        "torch.nn.functional.layer_norm(torch.nn.functional.linear(x, weight), "
        "normalized_shape=({form_metadata[weight_shape][0]},), "
        "eps={form_metadata[norm_eps]})). Inputs: x of shape "
        "{input_shapes[0]} and weight of shape {form_metadata[weight_shape]} "
        "(passed as a second positional argument to your function), both "
        "dtype {input_dtypes[0]}. Pass bias=None to linear. Output shape: "
        "{expected_output_shape}."
    ),
)


# ---------------------------------------------------------------------------
# Form 9 — gated_mlp_swiglu (Hard)
# ---------------------------------------------------------------------------
gated_mlp_swiglu = Form(
    name="gated_mlp_swiglu",
    tier=Difficulty.HARD,
    op_category=OpCategory.ACTIVATION,
    op_count=4,
    op_pool=(
        frozenset({"torch.nn.functional.linear"}),
        frozenset({"torch.nn.functional.linear"}),
        frozenset({"torch.nn.functional.silu"}),
        frozenset({"torch.mul"}),
    ),
    required_metadata_keys=("weight_shapes",),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        (((4, 128, 256),),),
        (((4, 256, 512),),),
        (((8, 128, 768),),),
        (((8, 256, 1024),),),
    ),
    output_shape=_gated_mlp_output,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: gated_mlp_swiglu. Compute torch.mul("
        "torch.nn.functional.silu(torch.nn.functional.linear(x, w_gate)), "
        "torch.nn.functional.linear(x, w_up)). Inputs: x of shape "
        "{input_shapes[0]}, plus two weights w_gate and w_up each of shape "
        "{form_metadata[weight_shapes][0]} (passed as second and third "
        "positional args). All dtype {input_dtypes[0]}. Pass bias=None to "
        "both linears. Output shape: {expected_output_shape}."
    ),
)


# ---------------------------------------------------------------------------
# Form 10 — chain_4_unary (Hard)
# ---------------------------------------------------------------------------
chain_4_unary = Form(
    name="chain_4_unary",
    tier=Difficulty.HARD,
    op_category=OpCategory.ELEMENTWISE_CHAIN,
    op_count=4,
    op_pool=(ELEMENTWISE_UNARY,) * 4,
    required_metadata_keys=(),
    dtypes=(Dtype.FLOAT32, Dtype.FLOAT16),
    shape_grid=(
        ((8, 128, 1024),),
        ((16, 256, 2048),),
        ((32, 512, 1024),),
    ),
    output_shape=_shape_preserving,
    output_dtype=_input_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: chain_4_unary. Apply four unary ops in this exact order: "
        "first {ops[0]}, then {ops[1]}, then {ops[2]}, then {ops[3]}. "
        "Input is a single tensor of shape {input_shapes[0]} and dtype "
        "{input_dtypes[0]}. Output preserves shape and dtype."
    ),
)


# ---------------------------------------------------------------------------
# Form 11 — embedding_then_norm (Hard)
# ---------------------------------------------------------------------------
embedding_then_norm = Form(
    name="embedding_then_norm",
    tier=Difficulty.HARD,
    op_category=OpCategory.EMBEDDING,
    op_count=3,
    op_pool=(
        frozenset({"torch.nn.functional.embedding"}),
        frozenset({"torch.nn.functional.layer_norm"}),
        ELEMENTWISE_UNARY,
    ),
    required_metadata_keys=("num_embeddings", "embedding_dim", "norm_eps"),
    dtypes=(Dtype.INT64, Dtype.FLOAT32),  # idx int64, weight float32; second variant uses fp16 weight
    shape_grid=(
        # First input: indices [B, T] int64. Second input: weight [V, D] float.
        (((4, 128), (4096, 256))),
        (((8, 256), (8192, 512))),
        (((4, 512), (16384, 1024))),
    ),
    output_shape=_embedding_output,
    output_dtype=_embedding_table_dtype,
    tolerance_policy=_default_tol,
    prompt_block=(
        "Form: embedding_then_norm. Compute {ops[2]}("
        "torch.nn.functional.layer_norm(torch.nn.functional.embedding(idx, weight), "
        "normalized_shape=({form_metadata[embedding_dim]},), "
        "eps={form_metadata[norm_eps]})). Inputs: idx of shape "
        "{input_shapes[0]} (int64 indices in [0, {form_metadata[num_embeddings]})), "
        "and weight of shape {input_shapes[1]} (float embedding table). "
        "Pass weight=None and bias=None to layer_norm. Expected output "
        "shape: {expected_output_shape}."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORMS: dict[str, Form] = {
    f.name: f for f in (
        chain_2_unary,
        unary_then_residual,
        chain_3_unary,
        unary_then_reduction,
        softmax_then_unary,
        unary_then_norm,
        attention_qkv,
        fused_linear_norm_activation,
        gated_mlp_swiglu,
        chain_4_unary,
        embedding_then_norm,
    )
}


# Sanity checks at import time.

assert len(FORMS) == 11, f"FORMS size drifted: {len(FORMS)} != 11"

for _form in FORMS.values():
    assert _form.op_count == len(_form.op_pool), (
        f"{_form.name}: op_count={_form.op_count} != len(op_pool)={len(_form.op_pool)}"
    )
    assert _form.op_count >= 2, f"{_form.name}: op_count must be >= 2"
    if _form.tier == Difficulty.EASY:
        assert _form.op_count == 2, f"{_form.name}: Easy must be 2 ops"
    if _form.tier == Difficulty.MEDIUM:
        assert _form.op_count in (2, 3), f"{_form.name}: Medium must be 2 or 3 ops"
    if _form.tier == Difficulty.HARD:
        assert _form.op_count >= 3, f"{_form.name}: Hard must be >= 3 ops"
