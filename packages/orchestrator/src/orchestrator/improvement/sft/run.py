"""
SFT pipeline entry point: train SFT LoRA adapters, then merge into a 4-bit
checkpoint at `~/models/qwen-coder-14b-instruct-4bit-tried-sft/`.

Run from the repo root on the MacBook:
    uv run python -m orchestrator.improvement.sft.run

Resume behaviour: mlx-lm-lora will pick up an existing adapter file under
`data/improvement/checkpoints/sft-adapters/` if present. To start fresh,
delete that directory before re-running.
"""
from __future__ import annotations

from orchestrator.improvement.merge import merge_stage
from orchestrator.improvement.sft.trainer import run_sft
from shared.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== SFT pipeline: train -> merge ===")
    run_sft()
    merge_stage("sft")
    logger.info("=== SFT pipeline complete ===")


if __name__ == "__main__":
    main()
