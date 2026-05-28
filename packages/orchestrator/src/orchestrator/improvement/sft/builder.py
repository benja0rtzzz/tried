"""
Build SFT data: dataset.jsonl → sft.jsonl.

For each row with at least one passing attempt, emit one chat-formatted
example in the `mlx_lm.lora` chat format:

    {"messages": [
        {"role": "system",    "content": <SYSTEM>},
        {"role": "user",      "content": <attempt-0 user prompt>},
        {"role": "assistant", "content": <winning triton_code>},
    ]}

The user prompt is always rendered with empty `prior_attempt_section`
(the attempt-0 prompt), regardless of which attempt index actually won —
the locked holdout measures one attempt per row with no retry.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.prompts.generator import SYSTEM, build_user_prompt


def build_sft(
    dataset_path: Path,
    output_path: Path,
    val_ids: set[str],
) -> int:
    """Write SFT examples to `output_path`. Returns the count written.

    Excludes any row whose source `example_id` is in `val_ids`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dataset_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["source"]["example_id"] in val_ids:
                continue

            winning_code = _find_winning_code(row)
            if winning_code is None:
                continue

            src = row["source"]
            user_prompt = build_user_prompt(
                pytorch_code=src["pytorch_code"],
                input_shapes=src["input_shapes"],
                input_dtypes=src["input_dtypes"],
                # attempt-0 prompt: no prior code or advice
            )

            record = {
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": winning_code},
                ]
            }
            fout.write(json.dumps(record) + "\n")
            count += 1

    return count


def _find_winning_code(row: dict) -> str | None:
    for attempt in row["attempts"]:
        correctness = attempt.get("correctness") or {}
        if correctness.get("status") == "passed":
            return attempt["triton_code"]
    return None
