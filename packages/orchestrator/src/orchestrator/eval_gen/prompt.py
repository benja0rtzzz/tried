"""Stage 2 — Locked prompt template + rendering for the Codex wrapper.

The prompt has two layers:
  - GLOBAL_TEMPLATE: invariant across every form. Encodes all of the
    AST-validator's hard rules in natural language so the LLM doesn't
    waste tokens producing code that stage 3 would reject.
  - per-form prompt_block from shared.eval.forms.FORMS, formatted with
    spec contents (ops, input_shapes, input_dtypes, form_metadata,
    expected_output_shape, expected_output_dtype).

LOCKED. Editing the prompt mid-experiment changes the search space the
LLM is sampling from and is a different experiment.
"""
from __future__ import annotations

from shared.eval.base_ops import (
    BASE_OPS,
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FREE_MOVEMENT,
    INDEXING,
    MATMUL,
    NORMALIZATION,
    REDUCTION,
)
from shared.eval.forms import FORMS
from shared.models import EvalSpec


def _format_op_section(name: str, ops: frozenset[str]) -> str:
    return f"- {name}: " + ", ".join(sorted(ops))


_ALLOWED_OPS_BLOCK = "\n".join([
    _format_op_section("elementwise unary", ELEMENTWISE_UNARY),
    _format_op_section("elementwise binary (function-form only)", ELEMENTWISE_BINARY),
    _format_op_section("reduction", REDUCTION),
    _format_op_section("normalization / softmax", NORMALIZATION),
    _format_op_section("matmul / linear", MATMUL),
    _format_op_section("indexing / embedding", INDEXING),
])

_FREE_MOVEMENT_BLOCK = "Allowed shape / layout (do not count toward op count): " + ", ".join(
    sorted(FREE_MOVEMENT)
)


GLOBAL_TEMPLATE = """\
You are translating a fully-determined fusion spec into a single PyTorch function.

# Hard rules

- No imports beyond `import torch`.
- No control flow (if / for / while), no comprehensions, no lambdas, no nested function definitions.
- Function-form binary and unary ops only: write `torch.add(x, y)`, never `x + y` or `-x`.
- No tensor methods that change device or dtype: no `.to()`, `.cuda()`, `.cpu()`, `.type()`, `.type_as()`.
- No tensor allocation inside the function body: no `torch.rand*`, `torch.empty*`, `torch.zeros*`, `torch.ones*`, `torch.full`, `torch.tensor`.
- The function must be deterministic given fixed inputs.
- Apply the required ops in the EXACT order listed in "Required ops" below — this is checked by an AST validator.

# Allowed ops

{allowed_ops_block}

{free_movement_block}

# Function signature

Write exactly one function definition. The function takes {n_inputs} positional argument(s) and returns a single torch.Tensor.

# Form-specific instructions

{form_block}

# Required ops, in order

{ops_block}

# Input tensors

{input_tensors_block}

# Expected output

- shape: {expected_output_shape}
- dtype: {expected_output_dtype}

# Response format

Respond with EXACTLY ONE fenced ```python code block containing the function definition. No prose before or after the block. The function name does not matter — pick whatever you like.

If the spec is unrealizable as stated, respond with EXACTLY:

```json
{{"error": "<one-line explanation>"}}
```

instead of a code block.
"""


def render(spec: EvalSpec) -> str:
    """Render the full stage-2 prompt for one spec."""
    form = FORMS[spec.form]

    form_block = form.prompt_block.format(
        ops=spec.ops,
        input_shapes=spec.input_shapes,
        input_dtypes=[d.value for d in spec.input_dtypes],
        form_metadata=spec.form_metadata,
        expected_output_shape=spec.expected_output_shape,
        expected_output_dtype=spec.expected_output_dtype.value,
    )

    ops_block = "\n".join(f"  {i + 1}. {op}" for i, op in enumerate(spec.ops))

    input_tensors_block = "\n".join(
        f"  - arg {i}: shape {list(shape)}, dtype {dtype.value}"
        for i, (shape, dtype) in enumerate(zip(spec.input_shapes, spec.input_dtypes))
    )

    return GLOBAL_TEMPLATE.format(
        allowed_ops_block=_ALLOWED_OPS_BLOCK,
        free_movement_block=_FREE_MOVEMENT_BLOCK,
        n_inputs=len(spec.input_shapes),
        form_block=form_block,
        ops_block=ops_block,
        input_tensors_block=input_tensors_block,
        expected_output_shape=spec.expected_output_shape,
        expected_output_dtype=spec.expected_output_dtype.value,
    )
