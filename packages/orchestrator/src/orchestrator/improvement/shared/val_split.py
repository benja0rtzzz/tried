"""
Carve and persist the SFT/DPO validation split.

Carved ONCE, persisted to `data/improvement/val_split.json`, stratified
by `op_category`. The same split is used by SFT and DPO so val rows
never leak into either stage's training data.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path


def carve_val_split(
    dataset_path: Path,
    val_split_path: Path,
    seed: int,
    val_fraction: float,
    stratify_by: str = "op_category",
) -> set[str]:
    """Return the held-out `example_id` set.

    Idempotent: if `val_split_path` already exists, load and return it
    without re-carving. Otherwise sample `val_fraction` of the SFT-positive
    rows stratified by `stratify_by`, write `{example_ids: [...]}` to disk,
    and return the set.
    """
    if val_split_path.exists():
        return load_val_ids(val_split_path)

    # Collect SFT-positive rows grouped by stratify_by key.
    buckets: dict[str, list[str]] = defaultdict(list)
    with dataset_path.open() as f:
        for line in f:
            row = json.loads(line)
            has_pass = any(
                (a.get("correctness") or {}).get("status") == "passed"
                for a in row["attempts"]
            )
            if not has_pass:
                continue
            key = row["source"].get(stratify_by, "unknown")
            buckets[key].append(row["source"]["example_id"])

    rng = random.Random(seed)
    val_ids: list[str] = []
    for ids in buckets.values():
        ids_copy = ids[:]
        rng.shuffle(ids_copy)
        n = max(1, math.ceil(len(ids_copy) * val_fraction))
        val_ids.extend(ids_copy[:n])

    val_split_path.parent.mkdir(parents=True, exist_ok=True)
    with val_split_path.open("w") as f:
        json.dump({"example_ids": sorted(val_ids)}, f, indent=2)

    return set(val_ids)


def load_val_ids(val_split_path: Path) -> set[str]:
    """Load held-out `example_id`s from a previously carved split file."""
    with val_split_path.open() as f:
        data = json.load(f)
    return set(data["example_ids"])
