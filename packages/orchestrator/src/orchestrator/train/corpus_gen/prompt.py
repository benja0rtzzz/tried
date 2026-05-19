"""LOCKED skeleton prompt for the codex CLI.

Produces ONE standalone PyTorch function per spec — a training-corpus input
that the agent loop will later try to translate into Triton. The prompt is
deliberately spec-driven (categorical axes, suggested shapes/dtypes) and
contains no kernel examples and no Triton.

LOCKED. Editing this prompt mid-experiment changes the distribution the
sampler is producing and is a different experiment.
"""
from __future__ import annotations

from .patterns import SkeletonSpec


SYSTEM = """\
You generate ONE standalone PyTorch reference function from a structural spec.

# Hard rules

- The module must begin with exactly `import torch` on line 1, then a blank line, then exactly one function definition. No other imports, no aliases, no helper functions.
- The function takes the listed positional tensor arguments and returns a single torch.Tensor.
- No tensor allocation inside the function body: no `torch.rand*`, `torch.empty*`, `torch.zeros*`, `torch.ones*`, `torch.full`, `torch.tensor`, `torch.arange` — only operate on the inputs.
- No tensor methods that change device or dtype: no `.to()`, `.cuda()`, `.cpu()`, `.type()`, `.type_as()`, `.float()`, `.half()`, `.double()`, `.bfloat16()`.
- The function must be deterministic given fixed inputs. No control flow that depends on tensor values, no randomness, no in-place mutation of inputs.
- The op_category and structural pattern in the spec are binding: the function MUST realise the indicated category and respect the listed reduction axis, broadcast, and fusion shape.
- Do NOT write or reference any Triton code.

# Response

The agent CLI is invoked with --output-schema; respond with a single JSON object matching:

  {
    "pytorch_code": "<the full python module text, exactly one function>",
    "input_shapes": [[d, d, ...], ...],
    "input_dtypes": ["float32" | "float16" | "bfloat16" | "float64" | "int32" | "int16" | "int8" | "int64" | "bool", ...],
    "rationale": "<one short sentence on why this realises the spec>"
  }

`input_shapes` and `input_dtypes` MUST be the concrete values the function expects. They may differ from the suggested values in the spec only if the suggestion is genuinely unrealizable; in that case keep them as close as possible and explain the deviation in `rationale`.
"""


_USER_TEMPLATE = """\
# Spec

- op_category:       {op_category}
- shape_rank:        {shape_rank}
- dtype_mix:         {dtype_mix}
- broadcast_pattern: {broadcast_pattern}
- reduction_axis:    {reduction_axis}
- fusion_shape:      {fusion_shape}
- memory_pattern:    {memory_pattern}

# Suggested inputs

{suggested_inputs_block}

# Reminder

- Realise op_category={op_category} as a {fusion_shape} over the inputs.
- If reduction_axis != none, the function must include a reduction along that axis.
- If broadcast_pattern != none, the function must rely on PyTorch broadcasting along that axis.
- Output exactly one JSON object matching the response schema. No prose, no code fences.
"""


def render(spec: SkeletonSpec) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for codex.exec.

    The codex CLI takes a single string; callers concatenate the two with the
    same <system>/<user> wrapper used by the judge client.
    """
    suggested_inputs_block = "\n".join(
        f"  - arg {i}: shape {shape}, dtype {dtype.value}"
        for i, (shape, dtype) in enumerate(
            zip(spec.suggested_input_shapes, spec.suggested_input_dtypes)
        )
    )
    user = _USER_TEMPLATE.format(
        op_category=spec.op_category.value,
        shape_rank=spec.shape_rank.value,
        dtype_mix=spec.dtype_mix.value,
        broadcast_pattern=spec.broadcast_pattern.value,
        reduction_axis=spec.reduction_axis.value,
        fusion_shape=spec.fusion_shape.value,
        memory_pattern=spec.memory_pattern.value,
        suggested_inputs_block=suggested_inputs_block,
    )
    return SYSTEM, user
