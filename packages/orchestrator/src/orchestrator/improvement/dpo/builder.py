"""
Build DPO data: dataset.jsonl → dpo.jsonl.

For each row that has BOTH a passing attempt AND at least one
non-passing attempt, emit one same-row preference pair:

    {"prompt":   <attempt-0 user prompt>,
     "chosen":   <winning triton_code>,
     "rejected": <earliest non-passing triton_code>}

Failure-mode mix of the rejected side is kept as-is — see
docs/finetuning.md for the empirical 85/15/0 runtime/numeric/compile
breakdown and the rationale.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.prompts.generator import build_user_prompt


def build_dpo(
    dataset_path: Path,
    output_path: Path,
    val_ids: set[str],
) -> int:
    """Write DPO pairs to `output_path`. Returns the count written.

    Excludes any row whose source `example_id` is in `val_ids`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dataset_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["source"]["example_id"] in val_ids:
                continue

            pair = _find_pair(row)
            if pair is None:
                continue

            chosen_code, rejected_code = pair
            src = row["source"]
            prompt = build_user_prompt(
                pytorch_code=src["pytorch_code"],
                input_shapes=src["input_shapes"],
                input_dtypes=src["input_dtypes"],
            )

            record = {
                "prompt":   prompt,
                "chosen":   chosen_code,
                "rejected": rejected_code,
            }
            fout.write(json.dumps(record) + "\n")
            count += 1

    return count


def _find_pair(row: dict) -> tuple[str, str] | None:
    """Return (chosen, rejected) or None if the row doesn't qualify."""
    chosen: str | None = None
    rejected: str | None = None  # earliest failing attempt

    for attempt in row["attempts"]:
        correctness = attempt.get("correctness") or {}
        if correctness.get("status") == "passed":
            if chosen is None:
                chosen = attempt["triton_code"]
        else:
            if rejected is None:
                rejected = attempt["triton_code"]

    if chosen is not None and rejected is not None:
        return chosen, rejected
    return None
