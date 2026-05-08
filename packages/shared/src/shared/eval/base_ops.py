"""Locked allow-list of PyTorch ops that may appear in eval-set fusions.

The stage-3 AST validator rejects any pytorch_code that calls a name outside
BASE_OPS, except for names in FREE_MOVEMENT (pure shape/layout operations
that don't count as ops). The stage-1 sampler picks ops from this list
when materializing specs.

LOCKED. Editing this file once data collection begins changes the search
space the LLM is asked to compose over and is a different experiment.

See docs/corpus.md for the rationale on which ops are present (tritonbench
operators set + the four plain PyTorch built-ins not in training:
relu / sigmoid / tanh / exp).
"""
from __future__ import annotations


# 25 canonical callable names. The validator extracts these as dotted
# paths from ast.Attribute / ast.Call nodes. Each frozenset documents the
# pool from which the corresponding form's op slot may draw.

ELEMENTWISE_UNARY: frozenset[str] = frozenset({
    "torch.nn.functional.gelu",
    "torch.nn.functional.silu",
    "torch.relu",
    "torch.sigmoid",
    "torch.tanh",
    "torch.exp",
    "torch.abs",
    "torch.sqrt",
})

ELEMENTWISE_BINARY: frozenset[str] = frozenset({
    "torch.add",
    "torch.mul",
    "torch.sub",
    "torch.div",
})

REDUCTION: frozenset[str] = frozenset({
    "torch.sum",
    "torch.mean",
    "torch.amax",
    "torch.var",
    "torch.argmax",
})

NORMALIZATION: frozenset[str] = frozenset({
    "torch.nn.functional.layer_norm",
    "torch.nn.functional.softmax",
    "torch.nn.functional.log_softmax",
})

MATMUL: frozenset[str] = frozenset({
    "torch.matmul",
    "torch.bmm",
    "torch.nn.functional.linear",
})

INDEXING: frozenset[str] = frozenset({
    "torch.nn.functional.embedding",
    "torch.gather",
})

BASE_OPS: frozenset[str] = (
    ELEMENTWISE_UNARY
    | ELEMENTWISE_BINARY
    | REDUCTION
    | NORMALIZATION
    | MATMUL
    | INDEXING
)

# Tensor methods and torch.X functions the AST validator IGNORES.
# Pure shape/layout manipulation — does not count toward the op-count rule.
# Listed here so stage 3 doesn't reject candidates that need shape arithmetic.
FREE_MOVEMENT: frozenset[str] = frozenset({
    # Tensor methods (called as x.<name>(...))
    "view", "reshape", "transpose", "permute", "contiguous",
    "unsqueeze", "squeeze", "flatten", "expand", "expand_as",
    "size",
    # torch.X function-form equivalents
    "torch.transpose", "torch.permute", "torch.reshape",
    "torch.unsqueeze", "torch.squeeze", "torch.flatten",
})

# Ops that produce non-float dtypes (or are otherwise terminal in a chain).
# Stage-1 sampler enforces these are last in the op slot order so a
# follow-on float-typed unary doesn't try to consume an int output.
TERMINAL_OPS: frozenset[str] = frozenset({
    "torch.argmax",
})

# Ops whose input is a long/integer tensor (indices), not a float.
# Stage-1 sampler threads the right dtype through the input fixture.
INTEGER_INPUT_OPS: frozenset[str] = frozenset({
    "torch.nn.functional.embedding",
    "torch.gather",
})


# Sanity checks executed at import time. These pin the op-set size at 25.
assert len(BASE_OPS) == 25, f"BASE_OPS size drifted: {len(BASE_OPS)} != 25"
assert TERMINAL_OPS <= BASE_OPS
assert INTEGER_INPUT_OPS <= BASE_OPS
assert FREE_MOVEMENT.isdisjoint(BASE_OPS), \
    "FREE_MOVEMENT must not overlap BASE_OPS"
