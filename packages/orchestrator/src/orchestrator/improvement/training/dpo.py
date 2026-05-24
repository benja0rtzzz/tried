"""
DPO trainer: wraps `mlx_lm.lora` for the DPO stage.

Started from the SFT checkpoint as both policy and reference. Reads
hyperparameters from `config/experiment.yaml` (`training.dpo`), trains
on `data/improvement/dpo.jsonl` with the carved val split held out,
and writes adapters to `data/improvement/checkpoints/sft-dpo-adapters/`.

Validation strategy (reward margin / chosen-rejected logprob gap on a
held-out slice) is deferred — see docs/finetuning.md "Open decisions".
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from orchestrator.prompts.generator import build_user_prompt

_BASE_MODEL = "mlx-community/Qwen2.5-Coder-14B-4bit"
_MICRO_BATCH = 2


def run_dpo(
    config: dict[str, Any],
    dpo_data_path: Path,
    dataset_path: Path,
    val_ids: set[str],
    sft_adapter_path: Path,
    output_adapter_path: Path,
) -> None:
    """Train DPO LoRA adapters starting from the SFT checkpoint.

    The SFT adapter is used to initialize the policy; mlx_lm.lora uses the
    same checkpoint as the frozen reference when no separate ref-model is
    given — matching the SFT-then-DPO recipe in docs/finetuning.md.
    """
    dpo_cfg = config["training"]["dpo"]
    sft_lora_cfg = config["training"]["sft"]["lora"]  # same adapter shape
    seed = config["inference"]["seed"]

    n_train = _count_lines(dpo_data_path)
    steps_per_epoch = math.ceil(n_train / _MICRO_BATCH)
    total_steps = steps_per_epoch * dpo_cfg["epochs"]
    warmup_steps = max(1, math.ceil(total_steps * 0.10))
    grad_accum = dpo_cfg["effective_batch"] // _MICRO_BATCH

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        _copy(dpo_data_path, data_dir / "train.jsonl")
        _write_val_dpo(dataset_path, data_dir / "valid.jsonl", val_ids)

        output_adapter_path.mkdir(parents=True, exist_ok=True)
        _run(
            "mlx_lm.lora",
            "--model",           _BASE_MODEL,
            "--resume-adapter-file", str(sft_adapter_path),
            "--train",
            "--train-type",      "dpo",
            "--data",            str(data_dir),
            "--adapter-path",    str(output_adapter_path),
            "--num-epochs",      str(dpo_cfg["epochs"]),
            "--batch-size",      str(_MICRO_BATCH),
            "--gradient-accumulation-steps", str(grad_accum),
            "--learning-rate",   str(dpo_cfg["lr"]),
            "--lr-schedule",     "linear",
            "--warmup-steps",    str(warmup_steps),
            "--max-seq-length",  str(dpo_cfg["max_seq_len"]),
            "--lora-rank",       str(sft_lora_cfg["rank"]),
            "--lora-alpha",      str(sft_lora_cfg["alpha"]),
            "--lora-dropout",    str(sft_lora_cfg["dropout"]),
            "--dpo-beta",        str(dpo_cfg["beta"]),
            "--grad-checkpoint",
            "--seed",            str(seed),
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_val_dpo(dataset_path: Path, dst: Path, val_ids: set[str]) -> None:
    """Build valid.jsonl (dpo format) from the val_ids subset of dataset."""
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
            fout.write(json.dumps({"prompt": prompt, "chosen": pair[0], "rejected": pair[1]}) + "\n")


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


def _copy(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def _count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def _run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)
