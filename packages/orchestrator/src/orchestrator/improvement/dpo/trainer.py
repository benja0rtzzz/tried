"""
DPO trainer wrapper.

Renders the DPO data dir (`data/improvement/dpo/{train,valid}.jsonl`) from
`data/dataset/dataset.jsonl` with the carved val split held out, runs the
preflight + schema checks, then subprocesses mlx-lm-lora with the technical
stage config at `config/finetuning_dpo.yaml`.

Per the SFT-then-DPO recipe, the DPO base is the SFT-merged 4-bit checkpoint
at `~/models/qwen-coder-14b-instruct-4bit-tried-sft/` (Design A: the fp16
intermediate from merge is transient and not persisted). The YAML sets both
`model` and `reference_model_path` to that directory; this wrapper enforces
the directory exists before launching mlx-lm-lora.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from orchestrator.improvement.dpo.builder import build_dpo
from orchestrator.improvement.shared.config import load_config
from orchestrator.improvement.shared.preflight import preflight_check
from orchestrator.improvement.shared.subprocess_utils import (
    materialize_resolved_yaml,
    stream_subprocess,
)
from orchestrator.improvement.shared.val_split import carve_val_split
from orchestrator.improvement.shared.validate_schema import validate_schema
from orchestrator.prompts.generator import build_user_prompt
from shared.logging import get_logger

logger = get_logger(__name__)

_STAGE_YAML = Path("config/finetuning_dpo.yaml")
_DATA_DIR = Path("data/improvement/dpo")
_DATASET_PATH = Path("data/dataset/dataset.jsonl")
_SFT_MERGED_DIR = Path("~/models/qwen-coder-14b-instruct-4bit-tried-sft").expanduser()


def run_dpo() -> None:
    """Train DPO LoRA adapters starting from the SFT-merged 4-bit base."""
    if not _SFT_MERGED_DIR.exists():
        raise RuntimeError(
            f"DPO base {_SFT_MERGED_DIR} does not exist. Run SFT + merge first."
        )

    preflight_check("dpo")
    validate_schema("dpo")

    cfg = load_config()
    val_ids = carve_val_split(
        dataset_path=_DATASET_PATH,
        val_split_path=Path(cfg["training"]["val_split_path"]),
        seed=cfg["inference"]["seed"],
        val_fraction=cfg["training"]["val_fraction"],
        stratify_by=cfg["training"]["stratify_by"],
    )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_train = build_dpo(_DATASET_PATH, _DATA_DIR / "train.jsonl", val_ids)
    n_valid = _build_dpo_valid(_DATASET_PATH, _DATA_DIR / "valid.jsonl", val_ids)
    logger.info("dpo data: %d train pairs, %d valid pairs", n_train, n_valid)
    if n_train == 0:
        raise RuntimeError("DPO train set is empty; aborting before mlx-lm-lora call")

    resolved_yaml = _DATA_DIR / "_resolved_config.yaml"
    materialize_resolved_yaml(_STAGE_YAML, resolved_yaml)

    logger.info("launching mlx-lm-lora DPO with %s", resolved_yaml)
    stream_subprocess(
        [sys.executable, "-m", "mlx_lm_lora.train", "-c", str(resolved_yaml), "--train"],
        log_prefix="[mlx_lm_lora]",
    )
    logger.info("dpo training complete; adapters under data/improvement/checkpoints/sft-dpo-adapters/")


def _build_dpo_valid(dataset_path: Path, dst: Path, val_ids: set[str]) -> int:
    """Render the held-out preference pairs to the same format as training."""
    count = 0
    with dataset_path.open() as fin, dst.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["source"]["example_id"] not in val_ids:
                continue
            pair = _find_pair(row)
            if pair is None:
                continue
            src = row["source"]
            prompt = build_user_prompt(
                pytorch_code=src["pytorch_code"],
                input_shapes=src["input_shapes"],
                input_dtypes=src["input_dtypes"],
            )
            fout.write(json.dumps({
                "prompt":   prompt,
                "chosen":   pair[0],
                "rejected": pair[1],
            }) + "\n")
            count += 1
    return count


def _find_pair(row: dict) -> tuple[str, str] | None:
    chosen: str | None = None
    rejected: str | None = None
    for attempt in row["attempts"]:
        if (attempt.get("correctness") or {}).get("status") == "passed":
            if chosen is None:
                chosen = attempt["triton_code"]
        else:
            if rejected is None:
                rejected = attempt["triton_code"]
    if chosen is not None and rejected is not None:
        return chosen, rejected
    return None


if __name__ == "__main__":
    run_dpo()
