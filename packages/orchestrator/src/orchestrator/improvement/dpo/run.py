"""
DPO pipeline entry point: train DPO LoRA adapters starting from the
SFT-merged 4-bit checkpoint, then merge into a 4-bit final checkpoint at
`~/models/qwen-coder-14b-instruct-4bit-tried-sft-dpo/`.

Run from the repo root on the MacBook:
    uv run python -m orchestrator.improvement.dpo.run

Prerequisite: `~/models/qwen-coder-14b-instruct-4bit-tried-sft/` must
exist (produced by `python -m orchestrator.improvement.sft.run`). This
entry point fails fast if it doesn't.

Resume behaviour: mlx-lm-lora will pick up an existing adapter file under
`data/improvement/checkpoints/sft-dpo-adapters/` if present. To start
fresh, delete that directory before re-running.
"""
from __future__ import annotations

from pathlib import Path

from orchestrator.improvement.merge import merge_stage
from orchestrator.improvement.dpo.trainer import run_dpo
from shared.logging import get_logger

logger = get_logger(__name__)

_SFT_MERGED_DIR = Path("~/models/qwen-coder-14b-instruct-4bit-tried-sft").expanduser()


def main() -> None:
    if not _SFT_MERGED_DIR.exists():
        raise RuntimeError(
            f"{_SFT_MERGED_DIR} not found. Run `python -m orchestrator.improvement.sft.run` first."
        )
    logger.info("=== DPO pipeline: train -> merge ===")
    run_dpo()
    merge_stage("dpo")
    logger.info("=== DPO pipeline complete ===")


if __name__ == "__main__":
    main()
