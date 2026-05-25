"""Stratified SkeletonSpec sampler weighted by discovery observations."""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from shared.enums import Dtype, OpCategory
from shared.logging import get_logger

from .patterns import (
    BroadcastPattern,
    DtypeMix,
    FusionShape,
    MemoryPattern,
    ReductionAxis,
    ShapeRank,
    SkeletonSpec,
    derive_spec_id,
)

logger = get_logger(__name__)
DEFAULT_OBS = Path("data/corpus_gen/observations.jsonl")
DEFAULT_OUT = Path("data/corpus_gen/specs.jsonl")

QUOTAS: dict[OpCategory, int] = {
    OpCategory.MATMUL: 120,
    OpCategory.FUSED_ATTENTION: 100,
    OpCategory.NORMALIZATION: 100,
    OpCategory.REDUCTION: 100,
    OpCategory.ACTIVATION: 100,
    OpCategory.ELEMENTWISE_CHAIN: 100,
    OpCategory.EMBEDDING: 80,
    OpCategory.LOSS: 80,
    OpCategory.CONVOLUTION: 80,
    OpCategory.QUANTIZATION: 80,
    OpCategory.OTHER: 60,
}

AXES: tuple[tuple[str, type], ...] = (
    ("shape_rank", ShapeRank),
    ("dtype_mix", DtypeMix),
    ("broadcast_pattern", BroadcastPattern),
    ("reduction_axis", ReductionAxis),
    ("fusion_shape", FusionShape),
    ("memory_pattern", MemoryPattern),
)

STOCK_SHAPES: dict[ShapeRank, list[list[int]]] = {
    ShapeRank.D1: [
        [256], [512], [1024], [2048], [4096],
        [8192], [16384], [32768], [65536],
    ],
    ShapeRank.D2: [
        [8, 512], [16, 768], [32, 1024], [64, 1024], [64, 2048],
        [128, 1024], [128, 2048], [256, 1024], [256, 2048], [256, 4096],
        [512, 1024], [32, 8192], [64, 4096], [16, 16384],
    ],
    ShapeRank.D3: [
        [4, 64, 4096], [4, 128, 2048], [8, 128, 1024], [8, 256, 1024],
        [16, 128, 1024], [16, 256, 2048], [32, 128, 1024], [32, 256, 1024],
        [32, 512, 1024], [64, 128, 512], [4, 1024, 1024], [16, 512, 2048],
    ],
    ShapeRank.D4: [
        [2, 8, 128, 128], [2, 32, 128, 256], [4, 8, 128, 128], [4, 16, 64, 64],
        [4, 16, 128, 128], [8, 8, 64, 64], [8, 16, 64, 64], [8, 16, 128, 128],
        [16, 8, 64, 64], [16, 16, 64, 128], [16, 32, 64, 128], [32, 32, 32, 64],
    ],
}


def _parse_allowed_ops(raw_values: list[str] | None) -> set[OpCategory] | None:
    if raw_values is None:
        return None

    values = [
        value.strip()
        for raw in raw_values
        for value in raw.split(",")
        if value.strip()
    ]
    valid = {category.value for category in OpCategory}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise ValueError(
            "invalid op categories: "
            + ", ".join(invalid)
            + "; valid values: "
            + ", ".join(sorted(valid))
        )
    if not values:
        raise ValueError("--allowed-ops was provided but no categories were listed")
    return {OpCategory(value) for value in values}


def _load_observations(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        logger.warning("observations file missing: %s", path)
        return []
    out: list[dict[str, str]] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append({str(k): str(v) for k, v in row.items()})
    return out


def _load_existing_spec_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    seen: set[str] = set()
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            spec_id = row.get("spec_id")
            if isinstance(spec_id, str):
                seen.add(spec_id)
    return seen


def _default_n(allowed_ops: set[OpCategory] | None) -> int:
    if allowed_ops is None:
        return sum(QUOTAS.values())
    return sum(QUOTAS[category] for category in QUOTAS if category in allowed_ops)


def _scale_quotas(
    target_total: int,
    allowed_ops: set[OpCategory] | None = None,
) -> dict[OpCategory, int]:
    active_categories = [category for category in QUOTAS if allowed_ops is None or category in allowed_ops]
    scaled = {category: 0 for category in QUOTAS}
    if target_total <= 0:
        return scaled
    if not active_categories:
        return scaled

    base_total = sum(QUOTAS[category] for category in active_categories)
    raw = {c: QUOTAS[c] * target_total / base_total for c in active_categories}
    scaled.update({c: int(raw[c]) for c in raw})
    remaining = target_total - sum(scaled.values())
    if remaining > 0:
        by_fraction = sorted(raw, key=lambda c: raw[c] - int(raw[c]), reverse=True)
        for i in range(remaining):
            scaled[by_fraction[i % len(by_fraction)]] += 1
    return scaled


def _choice(rng: random.Random, pairs: list[tuple[object, float]]) -> object:
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return rng.choice([value for value, _ in pairs])
    pivot = rng.random() * total
    upto = 0.0
    for value, weight in pairs:
        upto += weight
        if pivot <= upto:
            return value
    return pairs[-1][0]


def _build_distributions(observations: list[dict[str, str]]) -> dict[OpCategory, dict[str, list[tuple[object, float]]]]:
    by_category: dict[OpCategory, list[dict[str, str]]] = {c: [] for c in OpCategory}
    for row in observations:
        raw = row.get("op_category")
        try:
            category = OpCategory(raw) if raw is not None else None
        except ValueError:
            category = None
        if category is not None:
            by_category[category].append(row)

    out: dict[OpCategory, dict[str, list[tuple[object, float]]]] = {}
    for category in OpCategory:
        cat_rows = by_category[category]
        axis_map: dict[str, list[tuple[object, float]]] = {}
        for axis_name, enum_cls in AXES:
            values = list(enum_cls)
            if not cat_rows:
                axis_map[axis_name] = [(v, 1.0 / len(values)) for v in values]
                continue
            counts = {v: 0 for v in values}
            total = 0
            for row in cat_rows:
                raw = row.get(axis_name)
                if raw is None:
                    continue
                try:
                    v = enum_cls(raw)
                except ValueError:
                    continue
                counts[v] += 1
                total += 1
            axis_map[axis_name] = (
                [(v, counts[v] / total) for v in values]
                if total > 0
                else [(v, 1.0 / len(values)) for v in values]
            )
        out[category] = axis_map
    return out


def _input_count(fusion_shape: FusionShape, rng: random.Random) -> int:
    if fusion_shape == FusionShape.TRIPLET:
        return 3
    if fusion_shape == FusionShape.PAIR:
        return 2
    return rng.randint(1, 2)


def _broadcast_shape(base: list[int], pattern: BroadcastPattern, rank: ShapeRank) -> list[int]:
    dims = len(base)
    if rank == ShapeRank.D1:
        return [1]
    if pattern == BroadcastPattern.ROW:
        return [1] * (dims - 1) + [base[-1]]
    if pattern == BroadcastPattern.COL:
        return [base[0]] + [1] * (dims - 1)
    if pattern == BroadcastPattern.BATCH:
        return [1] + base[1:]
    if pattern == BroadcastPattern.CHANNEL:
        out = [1] * dims
        if dims >= 2:
            out[1] = base[1]
        return out
    return base.copy()


def _sample_shapes(rng: random.Random, rank: ShapeRank, broadcast: BroadcastPattern, n_inputs: int) -> list[list[int]]:
    base = rng.choice(STOCK_SHAPES[rank]).copy()
    shapes = [base]
    for _ in range(1, n_inputs):
        shapes.append(
            rng.choice(STOCK_SHAPES[rank]).copy()
            if broadcast == BroadcastPattern.NONE
            else _broadcast_shape(base, broadcast, rank)
        )
    return shapes


def _mixed_dtypes(rng: random.Random, a: Dtype, b: Dtype, n_inputs: int) -> list[Dtype]:
    if n_inputs == 1:
        return [rng.choice([a, b])]
    out = [a, b]
    while len(out) < n_inputs:
        out.append(rng.choice([a, b]))
    rng.shuffle(out)
    return out[:n_inputs]


def _sample_dtypes(rng: random.Random, dtype_mix: DtypeMix, n_inputs: int) -> list[Dtype]:
    if dtype_mix == DtypeMix.FP32_ONLY:
        return [Dtype.FLOAT32] * n_inputs
    if dtype_mix == DtypeMix.FP16_ONLY:
        return [Dtype.FLOAT16] * n_inputs
    if dtype_mix == DtypeMix.BF16_ONLY:
        return [Dtype.BFLOAT16] * n_inputs
    if dtype_mix == DtypeMix.MIXED_FP32_FP16:
        return _mixed_dtypes(rng, Dtype.FLOAT32, Dtype.FLOAT16, n_inputs)
    if dtype_mix == DtypeMix.MIXED_FP32_BF16:
        return _mixed_dtypes(rng, Dtype.FLOAT32, Dtype.BFLOAT16, n_inputs)
    if dtype_mix == DtypeMix.WITH_INT8:
        return _mixed_dtypes(rng, Dtype.INT8, Dtype.FLOAT32, n_inputs)
    return [Dtype.FLOAT32] * n_inputs


def _seed_from_spec_id(spec_id: str) -> int:
    return uuid.UUID(spec_id).int % (2**31 - 1)


def sample_specs(
    n: int,
    seed: int,
    observations_path: Path,
    allowed_ops: set[OpCategory] | None = None,
    exclude_spec_ids: set[str] | None = None,
) -> list[SkeletonSpec]:
    rng = random.Random(seed)
    distributions = _build_distributions(_load_observations(observations_path))
    quotas = _scale_quotas(n, allowed_ops)

    specs: list[SkeletonSpec] = []
    seen: set[str] = set(exclude_spec_ids or ())
    for category, quota in quotas.items():
        created = 0
        attempts = 0
        max_attempts = max(quota * 20, 100)
        while created < quota and attempts < max_attempts:
            attempts += 1
            axis = distributions[category]
            rank = _choice(rng, axis["shape_rank"])
            dtype_mix = _choice(rng, axis["dtype_mix"])
            broadcast = _choice(rng, axis["broadcast_pattern"])
            reduction = _choice(rng, axis["reduction_axis"])
            fusion = _choice(rng, axis["fusion_shape"])
            memory = _choice(rng, axis["memory_pattern"])

            n_inputs = _input_count(fusion, rng)
            shapes = _sample_shapes(rng, rank, broadcast, n_inputs)
            dtypes = _sample_dtypes(rng, dtype_mix, n_inputs)

            spec_id = derive_spec_id(
                op_category=category,
                shape_rank=rank,
                dtype_mix=dtype_mix,
                broadcast_pattern=broadcast,
                reduction_axis=reduction,
                fusion_shape=fusion,
                memory_pattern=memory,
                suggested_input_shapes=shapes,
                suggested_input_dtypes=dtypes,
            )
            if spec_id in seen:
                continue

            seen.add(spec_id)
            specs.append(
                SkeletonSpec(
                    spec_id=spec_id,
                    op_category=category,
                    shape_rank=rank,
                    dtype_mix=dtype_mix,
                    broadcast_pattern=broadcast,
                    reduction_axis=reduction,
                    fusion_shape=fusion,
                    memory_pattern=memory,
                    suggested_input_shapes=shapes,
                    suggested_input_dtypes=dtypes,
                    rng_seed=_seed_from_spec_id(spec_id),
                )
            )
            created += 1

        if created < quota:
            logger.warning("category %s capped: requested=%d created=%d", category.value, quota, created)

    return specs


def write_specs(specs: list[SkeletonSpec], out_path: Path, append: bool = True) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_spec_ids(out_path) if append else set()
    written = 0
    with out_path.open("a" if append else "w") as f:
        for spec in specs:
            if spec.spec_id in existing:
                continue
            f.write(spec.model_dump_json() + "\n")
            existing.add(spec.spec_id)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.train.corpus_gen.sampler")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="total specs to sample; defaults to the base quota total for selected categories",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--allowed-ops",
        nargs="+",
        default=None,
        metavar="OP_CATEGORY",
        help="only sample specs from these op categories; accepts spaces or commas",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--append",
        dest="append",
        action="store_true",
        default=True,
        help="append only new spec_id rows to --out (default)",
    )
    mode.add_argument(
        "--overwrite",
        dest="append",
        action="store_false",
        help="replace --out instead of appending",
    )
    args = parser.parse_args()

    try:
        allowed_ops = _parse_allowed_ops(args.allowed_ops)
    except ValueError as exc:
        parser.error(str(exc))
    n = args.n if args.n is not None else _default_n(allowed_ops)
    exclude_spec_ids = _load_existing_spec_ids(args.out) if args.append else set()
    if exclude_spec_ids:
        logger.info("append mode: excluding %d existing spec_id(s) from %s", len(exclude_spec_ids), args.out)

    specs = sample_specs(
        n=n,
        seed=args.seed,
        observations_path=args.observations,
        allowed_ops=allowed_ops,
        exclude_spec_ids=exclude_spec_ids,
    )
    written = write_specs(specs, args.out, append=args.append)
    action = "appended" if args.append else "wrote"
    logger.info("%s %d specs to %s", action, written, args.out)


if __name__ == "__main__":
    main()
