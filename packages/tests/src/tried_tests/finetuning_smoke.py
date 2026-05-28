"""
Layer-2 smoke test for the SFT trainer wrapper.

Builds a 2-row synthetic dataset, writes a smoke-mode resolved YAML on top
of `config/finetuning_sft.yaml`, and subprocesses
`python -m mlx_lm_lora.train -c <smoke yaml> --train` against the real
local base model.

Catches:
- mlx-lm-lora YAML acceptance regressions (deprecated keys, removed flags).
- "Cannot quantize already quantized model" crashes from `load_in_4bits` on
  a pre-quantized base.
- Chat data-format mismatches between our SFT builder and mlx-lm-lora's loader.
- Basic LoRA-attach / save bugs.

What it does NOT catch:
- Convergence (only 2 iterations).
- Memory pressure of real training (max_seq_length=256 here vs 2048).

Usage from the repo root on the MacBook:
    TRIED_ROLE=orchestrator uv run python -m tried_tests.finetuning_smoke
    TRIED_ROLE=orchestrator uv run python -m tried_tests.finetuning_smoke --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from shared.logging import get_logger

logger = get_logger(__name__)

_SFT_YAML = Path("config/finetuning_sft.yaml")

# Smoke-only overrides applied on top of finetuning_sft.yaml before launching.
# These reduce real training to ~2 optimizer steps on tiny inputs while still
# exercising every load + train + save code path mlx-lm-lora touches.
_SMOKE_OVERRIDES = {
    "iters":                       2,
    "epochs":                      None,
    "batch_size":                  1,
    "gradient_accumulation_steps": 1,
    "max_seq_length":              256,
    "lr_schedule":                 None,   # warmup>iters would error
    "save_every":                  1,
    "steps_per_report":            1,
    "steps_per_eval":              2,
    "val_batches":                 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live smoke test for the SFT trainer wrapper.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the smoke YAML but do not launch mlx-lm-lora.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tried-smoke-") as tmp:
        workdir = Path(tmp)
        logger.info("smoke workdir: %s", workdir)
        _smoke_sft(workdir, dry_run=args.dry_run)
        logger.info("=== SFT smoke OK ===")


def _smoke_sft(workdir: Path, *, dry_run: bool) -> None:
    data_dir    = workdir / "sft-data"
    adapter_dir = workdir / "sft-adapters"
    data_dir.mkdir()
    adapter_dir.mkdir()

    _write_sft_rows(data_dir / "train.jsonl", count=2)
    _write_sft_rows(data_dir / "valid.jsonl", count=1)

    smoke_yaml = _build_smoke_yaml(
        src_yaml=_SFT_YAML,
        dst_yaml=workdir / "smoke_sft.yaml",
        data_dir=data_dir,
        adapter_dir=adapter_dir,
    )
    _launch_or_skip(smoke_yaml, adapter_dir, dry_run=dry_run)


def _write_sft_rows(dst: Path, *, count: int) -> None:
    """Minimal chat-format rows. Content is not meaningful — just structurally valid."""
    row = {
        "messages": [
            {"role": "system",    "content": "You write Triton kernels."},
            {"role": "user",      "content": "Write a no-op kernel."},
            {"role": "assistant", "content": "import triton\nimport triton.language as tl\n\n@triton.jit\ndef noop(): pass\n"},
        ]
    }
    with dst.open("w") as f:
        for _ in range(count):
            f.write(json.dumps(row) + "\n")


def _build_smoke_yaml(
    *,
    src_yaml: Path,
    dst_yaml: Path,
    data_dir: Path,
    adapter_dir: Path,
) -> Path:
    with src_yaml.open() as f:
        cfg = yaml.safe_load(f)

    # Expanduser the path fields mlx-lm doesn't resolve itself.
    for field in ("model", "data", "adapter_path"):
        if isinstance(cfg.get(field), str):
            cfg[field] = str(Path(cfg[field]).expanduser())

    cfg["data"]         = str(data_dir)
    cfg["adapter_path"] = str(adapter_dir)
    for key, value in _SMOKE_OVERRIDES.items():
        cfg[key] = value

    with dst_yaml.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info("wrote smoke YAML: %s", dst_yaml)
    logger.info("base model: %s", cfg["model"])
    return dst_yaml


def _launch_or_skip(smoke_yaml: Path, adapter_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would launch: mlx_lm_lora.train -c %s --train", smoke_yaml)
        return

    logger.info("launching mlx-lm-lora SFT smoke...")
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_lm_lora.train", "-c", str(smoke_yaml), "--train"],
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sft smoke: mlx_lm_lora.train exited with code {proc.returncode}")

    adapters = list(adapter_dir.glob("*.safetensors"))
    if not adapters:
        raise RuntimeError(
            f"sft smoke: no *.safetensors file written under {adapter_dir}. "
            f"mlx-lm-lora exited cleanly but never saved an adapter."
        )
    logger.info("sft smoke OK: %d adapter file(s): %s",
                len(adapters), [a.name for a in adapters])


if __name__ == "__main__":
    main()
