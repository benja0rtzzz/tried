"""
SFT trainer wrapper.

Renders the SFT data dir (`data/improvement/sft/{train,valid}.jsonl`) from
`data/dataset/dataset.jsonl` with the carved val split held out, runs the
preflight + schema checks, then subprocesses mlx-lm-lora with the technical
stage config at `config/finetuning_sft.yaml`.

All hyperparameters come from the stage YAML — this wrapper never passes
them on the CLI. That keeps a single source of truth (the YAML) and lets
`preflight_check` enforce parity with `config/config.yaml`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from orchestrator.improvement.shared.config import load_config
from orchestrator.improvement.shared.preflight import preflight_check
from orchestrator.improvement.shared.subprocess_utils import (
    materialize_resolved_yaml,
    stream_subprocess,
)
from orchestrator.improvement.shared.val_split import carve_val_split
from orchestrator.improvement.shared.validate_schema import validate_schema
from orchestrator.improvement.sft.builder import build_sft
from orchestrator.prompts.generator import SYSTEM, build_user_prompt
from shared.logging import get_logger

logger = get_logger(__name__)

_STAGE_YAML = Path("config/finetuning_sft.yaml")
_DATA_DIR = Path("data/improvement/sft")
_DATASET_PATH = Path("data/dataset/dataset.jsonl")


def run_sft() -> None:
    """Train SFT LoRA adapters. Reads everything from the config files."""
    preflight_check("sft")
    validate_schema("sft")

    cfg = load_config()
    val_ids = carve_val_split(
        dataset_path=_DATASET_PATH,
        val_split_path=Path(cfg["training"]["val_split_path"]),
        seed=cfg["inference"]["seed"],
        val_fraction=cfg["training"]["val_fraction"],
        stratify_by=cfg["training"]["stratify_by"],
    )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_train = build_sft(_DATASET_PATH, _DATA_DIR / "train.jsonl", val_ids)
    n_valid = _build_sft_valid(_DATASET_PATH, _DATA_DIR / "valid.jsonl", val_ids)
    logger.info("sft data: %d train rows, %d valid rows", n_train, n_valid)
    if n_train == 0:
        raise RuntimeError("SFT train set is empty; aborting before mlx-lm-lora call")

    # mlx-lm-lora reads the YAML directly and does not expanduser on path
    # fields, so we materialise a resolved copy with ~/ expanded before
    # launching. Kept on disk for post-mortem inspection.
    resolved_yaml = _DATA_DIR / "_resolved_config.yaml"
    materialize_resolved_yaml(_STAGE_YAML, resolved_yaml)

    logger.info("launching mlx-lm-lora SFT with %s", resolved_yaml)
    stream_subprocess(
        [sys.executable, "-m", "mlx_lm_lora.train", "-c", str(resolved_yaml), "--train"],
        log_prefix="[mlx_lm_lora]",
    )
    logger.info("sft training complete; adapters under data/improvement/checkpoints/sft-adapters/")


def _build_sft_valid(dataset_path: Path, dst: Path, val_ids: set[str]) -> int:
    """Render the held-out SFT-positive rows into the same chat format as training."""
    count = 0
    with dataset_path.open() as fin, dst.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["source"]["example_id"] not in val_ids:
                continue
            winning_code = _find_winning_code(row)
            if winning_code is None:
                continue
            src = row["source"]
            user_prompt = build_user_prompt(
                pytorch_code=src["pytorch_code"],
                input_shapes=src["input_shapes"],
                input_dtypes=src["input_dtypes"],
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
        if (attempt.get("correctness") or {}).get("status") == "passed":
            return attempt["triton_code"]
    return None


if __name__ == "__main__":
    run_sft()
