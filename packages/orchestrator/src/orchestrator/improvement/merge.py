"""
Adapter merge + 4-bit re-quantization.

Wraps `mlx_lm fuse` (LoRA adapters merged into the base, dequantized to
fp16 in a TemporaryDirectory) and `mlx_lm convert -q` (fp16 → 4-bit at the
same quantization shape as the base). Only the final 4-bit checkpoint is
kept; the fp16 intermediate lives only inside the TemporaryDirectory.

Per-stage paths (Design A — no fp16 staging persisted):

    sft  : base=config.yaml::training.base_model
           adapters=data/improvement/checkpoints/sft-adapters/
           output=~/models/qwen-coder-14b-instruct-4bit-tried-sft/

    dpo  : base=~/models/qwen-coder-14b-instruct-4bit-tried-sft/
           adapters=data/improvement/checkpoints/sft-dpo-adapters/
           output=~/models/qwen-coder-14b-instruct-4bit-tried-sft-dpo/

Merged checkpoints live under ~/models/ alongside the base 4-bit model
(loadable as MLX checkpoints); adapters stay under data/improvement/
checkpoints/ because they are training intermediates, not standalone models.

Quantization shape is read from the base model's config.json so the merged
output is byte-compatible with what mlx-lm-lora and the eval client expect.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from orchestrator.improvement.shared.config import load_config
from orchestrator.improvement.shared.subprocess_utils import stream_subprocess
from shared.logging import get_logger

logger = get_logger(__name__)

_CHECKPOINT_ROOT = Path("data/improvement/checkpoints")
_SFT_ADAPTERS    = _CHECKPOINT_ROOT / "sft-adapters"
_DPO_ADAPTERS    = _CHECKPOINT_ROOT / "sft-dpo-adapters"

_MODELS_ROOT = Path("~/models").expanduser()
_SFT_MERGED  = _MODELS_ROOT / "qwen-coder-14b-instruct-4bit-tried-sft"
_DPO_MERGED  = _MODELS_ROOT / "qwen-coder-14b-instruct-4bit-tried-sft-dpo"


def merge_stage(stage: str) -> None:
    """Merge adapters for `stage` (sft|dpo) into a 4-bit checkpoint."""
    if stage == "sft":
        base_model    = Path(load_config()["training"]["base_model"]).expanduser()
        adapter_path  = _SFT_ADAPTERS
        output_path   = _SFT_MERGED
    elif stage == "dpo":
        base_model    = _SFT_MERGED
        adapter_path  = _DPO_ADAPTERS
        output_path   = _DPO_MERGED
    else:
        raise ValueError(f"unknown stage {stage!r}; expected 'sft' or 'dpo'")

    if not Path(base_model).exists():
        raise RuntimeError(f"merge base {base_model} does not exist")
    if not adapter_path.exists():
        raise RuntimeError(f"adapter path {adapter_path} does not exist")

    q = _read_quantization(Path(base_model))
    logger.info("merge %s: base=%s adapters=%s output=%s quant=%s",
                stage, base_model, adapter_path, output_path, q)

    merge_and_quantize(
        base_model=str(base_model),
        adapter_path=adapter_path,
        output_path=output_path,
        q_bits=q["bits"],
        q_group_size=q["group_size"],
        q_mode=q.get("mode", "affine"),
    )
    logger.info("merge %s complete: %s", stage, output_path)


def merge_and_quantize(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    q_bits: int,
    q_group_size: int,
    q_mode: str,
) -> None:
    """Merge LoRA into `base_model`, then re-quantize to (q_bits, q_group_size, q_mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        fp16_path = Path(tmp) / "merged-fp16"

        # 1) Fuse: merge bf16 LoRA into the (possibly 4-bit) base, dequantize to fp16.
        stream_subprocess(
            [sys.executable, "-m", "mlx_lm", "fuse",
             "--model",        base_model,
             "--adapter-path", str(adapter_path),
             "--save-path",    str(fp16_path),
             "--dequantize"],
            log_prefix="[mlx_lm fuse]",
        )

        # 2) Convert: re-quantize fp16 → 4-bit with the base's quant shape.
        # mlx_lm convert creates --mlx-path itself and refuses to write into an
        # existing directory, so we must not pre-create output_path here.
        stream_subprocess(
            [sys.executable, "-m", "mlx_lm", "convert",
             "--hf-path",      str(fp16_path),
             "--mlx-path",     str(output_path),
             "--quantize",
             "--q-bits",       str(q_bits),
             "--q-group-size", str(q_group_size),
             "--q-mode",       q_mode],
            log_prefix="[mlx_lm convert]",
        )
    # fp16 artifact is inside the TemporaryDirectory and is gone here.


def _read_quantization(model_dir: Path) -> dict:
    """Read the `quantization` block from a local MLX checkpoint's config.json."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(
            f"{cfg_path} not found; merge needs a local 4-bit checkpoint to "
            f"read group_size / bits / mode from"
        )
    with cfg_path.open() as f:
        cfg = json.load(f)
    q = cfg.get("quantization")
    if not q:
        raise RuntimeError(
            f"{cfg_path} has no `quantization` block; base must be a quantized "
            f"MLX checkpoint"
        )
    return q


def _main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and re-quantize to 4-bit.")
    parser.add_argument("--stage", choices=("sft", "dpo"), required=True)
    args = parser.parse_args()
    merge_stage(args.stage)


if __name__ == "__main__":
    _main()
