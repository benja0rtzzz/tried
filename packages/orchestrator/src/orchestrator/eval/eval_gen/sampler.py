"""Stage 1 — Spec sampler.

Reads the locked FORMS registry from shared.eval.forms and emits EvalSpec rows
for the medium+hard supplement run. Per-form QUOTAS currently sum to 150; easy
forms are omitted because the earlier easy batch already had enough accepted
rows.

Output: data/eval_gen/specs.jsonl, one EvalSpec per line.

Determinism:
  - Each spec_id is UUIDv5 over its canonical key, so re-running the
    sampler on the same FORMS produces identical spec_ids.
  - rng_seed for input fixtures is independent of spec_id and is used
    by the verification server's /preflight to materialize inputs.
"""
from __future__ import annotations

import json
import random
import uuid
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Any

from shared.eval.forms import FORMS, Form
from shared.models import EvalSpec


# Per-form spec quotas for the medium+hard supplement run.
# Easy forms (chain_2_unary, unary_then_residual) are omitted — the
# 123 easy rows already in with_code.jsonl are sufficient.
# Quotas are weighted toward forms with large unexplored grids so that
# the driver's IntraEvalDedup doesn't eat most of the budget.
QUOTAS: dict[str, int] = {
    "chain_3_unary":                20,   # medium, grid=6144
    "unary_then_reduction":         20,   # medium, grid=2080
    "softmax_then_unary":           15,   # medium, grid=256
    "unary_then_norm":              15,   # medium, grid=48 (mostly covered)
    "attention_qkv":                10,   # hard,   grid=8  (exhausted)
    "fused_linear_norm_activation": 20,   # hard,   grid=64
    "gated_mlp_swiglu":             10,   # hard,   grid=8  (exhausted/low-pass)
    "chain_4_unary":                25,   # hard,   grid=24576
    "embedding_then_norm":          15,   # hard,   grid=48
}
assert sum(QUOTAS.values()) == 150
assert set(QUOTAS) == set(FORMS) - {"chain_2_unary", "unary_then_residual"}, \
    "QUOTAS keys must match non-easy FORMS"

# Stable namespace for spec_id UUIDv5 derivation (TRIED-eval).
_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000eaa")

# Default base seed; sampler increments per-spec deterministically.
DEFAULT_BASE_SEED = 1000


def _form_metadata_variants(form: Form, shape_variant: tuple) -> list[dict[str, Any]]:
    """Enumerate the form_metadata variants for a given (form, shape).
    Returns at least one dict; multiple if there are real choices to make."""
    name = form.name
    if name == "unary_then_reduction":
        ndim = len(shape_variant[0])
        return [
            {"reduce_dim": d, "keepdim": k}
            for d in range(-ndim, 0)
            for k in (False, True)
        ]
    if name == "softmax_then_unary":
        ndim = len(shape_variant[0])
        return [{"softmax_dim": d} for d in range(-ndim, 0)]
    if name == "unary_then_norm":
        return [{"normalized_shape": (shape_variant[0][-1],), "norm_eps": 1e-5}]
    if name == "attention_qkv":
        # num_heads from H dim of [B, H, T, D_h]; scale = 1/sqrt(D_h).
        # causal=True omitted: building a triangular mask requires tensor
        # allocation or control flow, both of which the AST validator rejects.
        head_dim = shape_variant[0][3]
        return [
            {"num_heads": shape_variant[0][1], "causal": False, "scale": 1.0 / sqrt(head_dim)}
        ]
    if name == "fused_linear_norm_activation":
        # normalized_shape comes from weight.shape[0] (D_out).
        return [{"normalized_shape": (shape_variant[1][0],), "norm_eps": 1e-5}]
    if name == "embedding_then_norm":
        return [{"num_embeddings": shape_variant[1][0], "norm_eps": 1e-5}]
    # Forms with no required_metadata_keys.
    return [{}]


def _enumerate_op_combos(form: Form) -> list[tuple[str, ...]]:
    """Cartesian product across op_pool slots — every legal ops tuple
    for this form."""
    return [tuple(combo) for combo in product(*(sorted(p) for p in form.op_pool))]


def _spec_canonical_key(
    form_name: str,
    ops: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[str, ...],
    form_metadata: dict[str, Any],
    rng_seed: int,
) -> str:
    """Stable JSON serialization for UUIDv5 derivation."""
    payload = {
        "form": form_name,
        "ops": list(ops),
        "shapes": [list(s) for s in shapes],
        "dtypes": list(dtypes),
        "form_metadata": form_metadata,
        "rng_seed": rng_seed,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _spec_id(canonical_key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, canonical_key))


def _build_spec(
    form: Form,
    ops: tuple[str, ...],
    shape_variant: tuple[tuple[int, ...], ...],
    dtype_variant: tuple,
    form_metadata: dict[str, Any],
    rng_seed: int,
) -> EvalSpec:
    """Construct one EvalSpec with derived fields."""
    input_shapes = [list(s) for s in shape_variant]
    input_dtypes = [d.value for d in dtype_variant]
    expected_output_shape = form.output_shape(form_metadata, input_shapes, list(ops))
    expected_output_dtype = form.output_dtype(form_metadata, input_dtypes, list(ops))
    tolerance_policy = form.tolerance_policy(expected_output_dtype, list(ops))
    canonical = _spec_canonical_key(
        form.name, ops, shape_variant, tuple(input_dtypes), form_metadata, rng_seed
    )
    return EvalSpec(
        spec_id=_spec_id(canonical),
        tier=form.tier,
        form=form.name,  # type: ignore[arg-type]
        ops=list(ops),
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,  # type: ignore[arg-type]
        expected_output_shape=expected_output_shape,
        expected_output_dtype=expected_output_dtype,  # type: ignore[arg-type]
        tolerance_policy=tolerance_policy,  # type: ignore[arg-type]
        rng_seed=rng_seed,
        form_metadata=form_metadata,
    )


def sample_specs_for_form(
    form: Form,
    target_count: int,
    rng: random.Random,
    base_seed: int = DEFAULT_BASE_SEED,
) -> list[EvalSpec]:
    """Sample `target_count` specs for one form by enumerating the design
    grid (shape * dtype * ops * fm) and picking cells. If the grid is
    smaller than target_count, cells repeat with distinct rng_seeds so
    every spec has a unique spec_id."""
    op_combos = _enumerate_op_combos(form)
    # Build the full design-cell list (without rng_seed; that's appended later).
    cells: list[tuple[tuple[str, ...], tuple, tuple, dict]] = []
    for shape_variant in form.shape_grid:
        for dtype_variant in form.dtype_grid:
            for fm in _form_metadata_variants(form, shape_variant):
                for ops in op_combos:
                    cells.append((ops, shape_variant, dtype_variant, fm))

    rng.shuffle(cells)
    grid_size = len(cells)

    specs: list[EvalSpec] = []
    seed = base_seed
    if grid_size >= target_count:
        for cell in cells[:target_count]:
            ops, shape_variant, dtype_variant, fm = cell
            specs.append(_build_spec(form, ops, shape_variant, dtype_variant, fm, seed))
            seed += 1
    else:
        # Grid is smaller than quota — cycle through cells, varying rng_seed
        # so every spec has a unique spec_id.
        i = 0
        while len(specs) < target_count:
            ops, shape_variant, dtype_variant, fm = cells[i % grid_size]
            specs.append(_build_spec(form, ops, shape_variant, dtype_variant, fm, seed))
            seed += 1
            i += 1

    return specs


def sample_all(rng_master_seed: int = 1) -> list[EvalSpec]:
    """Run the sampler across every form, respecting QUOTAS. Returns the
    full ~600 spec list."""
    rng = random.Random(rng_master_seed)
    out: list[EvalSpec] = []
    seed_cursor = DEFAULT_BASE_SEED
    for form_name, target in QUOTAS.items():
        form = FORMS[form_name]
        specs = sample_specs_for_form(form, target, rng, base_seed=seed_cursor)
        out.extend(specs)
        seed_cursor += target  # disjoint seed ranges per form
    return out


def write_specs(specs: list[EvalSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for spec in specs:
            f.write(spec.model_dump_json() + "\n")


DEFAULT_OUT = Path("data/eval_gen/specs.jsonl")


def main() -> None:
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="orchestrator.eval.eval_gen.sampler")
    parser.add_argument("--seed", type=int, default=1, help="master RNG seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = sample_all(rng_master_seed=args.seed)
    write_specs(specs, args.out)
    by_form: dict[str, int] = {}
    for s in specs:
        by_form[s.form] = by_form.get(s.form, 0) + 1
    print(f"Wrote {len(specs)} specs to {args.out}")
    for form_name, count in by_form.items():
        target = QUOTAS[form_name]
        print(f"  {form_name}: {count} (target {target})")


if __name__ == "__main__":
    main()
