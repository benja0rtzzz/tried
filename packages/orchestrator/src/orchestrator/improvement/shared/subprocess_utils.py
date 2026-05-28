"""
Helpers shared by the SFT and DPO trainers and by the merge step.

- `stream_subprocess`: runs a command, streams its combined stdout/stderr
  through the project logger one line at a time, and raises on non-zero exit.
- `materialize_resolved_yaml`: copies a finetuning_<stage>.yaml to a target
  path with `~/` expanded on the known path-valued fields. mlx-lm does not
  expanduser these itself.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import yaml

from shared.logging import get_logger

logger = get_logger(__name__)

_PATH_FIELDS = ("model", "data", "adapter_path", "reference_model_path")


def materialize_resolved_yaml(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` after expanduser-resolving path-valued fields."""
    with src.open() as f:
        cfg = yaml.safe_load(f)
    for field in _PATH_FIELDS:
        if isinstance(cfg.get(field), str):
            cfg[field] = str(Path(cfg[field]).expanduser())
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def stream_subprocess(args: Iterable[str], *, log_prefix: str) -> None:
    """Run `args`, stream stdout+stderr through the logger, raise on non-zero.

    `log_prefix` is prepended to each captured line so the operator can
    tell which subprocess produced what (e.g. "[mlx_lm_lora]", "[mlx_lm]").
    """
    argv = list(args)
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("%s %s", log_prefix, line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{' '.join(argv)} exited with code {rc}")
