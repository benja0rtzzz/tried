"""
Adapter merge + 4-bit re-quantization.

Wraps `mlx_lm.merge` (adapters → transient fp16) and `mlx_lm.convert`
(fp16 → 4-bit). Only the final 4-bit merged checkpoint is kept; the
intermediate fp16 artifact is discarded after conversion succeeds.

Producing the eval-ready checkpoints:
    sft-adapters/     →  sft-merged-4bit/
    sft-dpo-adapters/ →  sft-dpo-merged-4bit/
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def merge_and_quantize(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
) -> None:
    """Merge LoRA adapters into `base_model` and re-quantize to 4-bit.

    `base_model` is the HF / mlx-community id of the 4-bit base
    (mlx-community/Qwen2.5-Coder-14B-4bit). `output_path` receives the
    final 4-bit merged checkpoint directory.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fp16_path = Path(tmp) / "merged-fp16"

        _run(
            "mlx_lm.fuse",
            "--model", base_model,
            "--adapter-path", str(adapter_path),
            "--save-path", str(fp16_path),
            "--de-quantize",   # produce fp16 weights, not re-quantized
        )

        output_path.mkdir(parents=True, exist_ok=True)
        _run(
            "mlx_lm.convert",
            "--hf-path", str(fp16_path),
            "--mlx-path", str(output_path),
            "--quantize",
            "--q-bits", "4",
        )
    # fp16 artifact lives only inside the TemporaryDirectory and is gone here.


def _run(*args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", *args],
        check=True,
    )
