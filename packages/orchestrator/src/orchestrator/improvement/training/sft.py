"""
SFT trainer: wraps `mlx_lm.lora` for the SFT stage.

Reads hyperparameters from `config/experiment.yaml` (`training.sft`),
trains response-only-masked LoRA adapters on `data/improvement/sft.jsonl`
with the carved val split held out, and writes adapters to
`data/improvement/checkpoints/sft-adapters/`.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from orchestrator.prompts.generator import SYSTEM, build_user_prompt

_BASE_MODEL = "mlx-community/Qwen2.5-Coder-14B-4bit"
_MICRO_BATCH = 2  # conservative for 48 GB + rank-64 + 2048 ctx


def run_sft(
    config: dict[str, Any],
    sft_data_path: Path,
    dataset_path: Path,
    val_ids: set[str],
    output_adapter_path: Path,
) -> None:
    """Train SFT LoRA adapters and write them to `output_adapter_path`.

    `config` is the parsed `config/experiment.yaml`. Uses
    `inference.chat_template` for tokenization and `inference.seed` for
    shuffle / adapter init reproducibility.
    """
    sft_cfg = config["training"]["sft"]
    lora_cfg = sft_cfg["lora"]
    seed = config["inference"]["seed"]

    n_train = _count_lines(sft_data_path)
    steps_per_epoch = math.ceil(n_train / _MICRO_BATCH)
    total_steps = steps_per_epoch * sft_cfg["epochs"]
    warmup_steps = max(1, math.ceil(total_steps * 0.05))
    grad_accum = sft_cfg["effective_batch"] // _MICRO_BATCH

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _copy(sft_data_path, data_dir / "train.jsonl")
        _write_val_sft(dataset_path, data_dir / "valid.jsonl", val_ids)

        output_adapter_path.mkdir(parents=True, exist_ok=True)
        _run(
            "mlx_lm.lora",
            "--model",           _BASE_MODEL,
            "--train",
            "--data",            str(data_dir),
            "--adapter-path",    str(output_adapter_path),
            "--num-epochs",      str(sft_cfg["epochs"]),
            "--batch-size",      str(_MICRO_BATCH),
            "--gradient-accumulation-steps", str(grad_accum),
            "--learning-rate",   str(sft_cfg["lr"]),
            "--lr-schedule",     "cosine",
            "--warmup-steps",    str(warmup_steps),
            "--max-seq-length",  str(sft_cfg["max_seq_len"]),
            "--lora-rank",       str(lora_cfg["rank"]),
            "--lora-alpha",      str(lora_cfg["alpha"]),
            "--lora-dropout",    str(lora_cfg["dropout"]),
            "--grad-checkpoint",
            "--seed",            str(seed),
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_val_sft(dataset_path: Path, dst: Path, val_ids: set[str]) -> None:
    """Build valid.jsonl (messages format) from the val_ids subset of dataset."""
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


def _find_winning_code(row: dict) -> str | None:
    for attempt in row["attempts"]:
        if (attempt.get("correctness") or {}).get("status") == "passed":
            return attempt["triton_code"]
    return None


def _copy(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def _count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def _run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)
