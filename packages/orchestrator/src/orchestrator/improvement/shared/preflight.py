"""
Cross-file consistency check for fine-tuning configs.

config/config.yaml is the authoritative spec; config/finetuning_sft.yaml and
config/finetuning_dpo.yaml are what mlx-lm-lora actually reads. They must
agree on every shared hyperparameter or the experiment is mis-specified.

Each trainer wrapper calls `preflight_check(stage)` before invoking
mlx-lm-lora; on any mismatch we raise PreflightError and abort the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchestrator.improvement.shared.config import load_config

_STAGE_YAML = {
    "sft": Path("config/finetuning_sft.yaml"),
    "dpo": Path("config/finetuning_dpo.yaml"),
}

# Keys whose values must be byte-identical between config.yaml::training.<stage>
# and finetuning_<stage>.yaml. (config_path, stage_yaml_path) — both dotted.
_SHARED_KEYS_COMMON = [
    ("learning_rate",               "learning_rate"),
    ("epochs",                      "epochs"),
    ("batch_size",                  "batch_size"),
    ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ("max_seq_length",              "max_seq_length"),
    ("lora.rank",                   "lora_parameters.rank"),
    ("lora.dropout",                "lora_parameters.dropout"),
    ("lora.scale",                  "lora_parameters.scale"),
]
_SHARED_KEYS_SFT = _SHARED_KEYS_COMMON + [
    ("mask_prompt", "mask_prompt"),
]
_SHARED_KEYS_DPO = _SHARED_KEYS_COMMON + [
    ("beta", "beta"),
]


class PreflightError(RuntimeError):
    """Raised when finetuning_<stage>.yaml drifts from config.yaml."""


def preflight_check(stage: str) -> None:
    """Verify finetuning_<stage>.yaml mirrors config.yaml::training.<stage>.

    Also checks that the stage yaml's `seed` matches config.yaml::inference.seed
    (the seed is shared experiment-wide).
    """
    if stage not in _STAGE_YAML:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGE_YAML)}")

    cfg = load_config()
    with _STAGE_YAML[stage].open() as f:
        stage_cfg = yaml.safe_load(f)

    spec = cfg["training"][stage]
    shared_keys = _SHARED_KEYS_SFT if stage == "sft" else _SHARED_KEYS_DPO

    mismatches: list[str] = []

    # 1) Per-stage hyperparams must match.
    for spec_key, stage_key in shared_keys:
        spec_val = _dotted_get(spec, spec_key)
        stage_val = _dotted_get(stage_cfg, stage_key)
        if spec_val != stage_val:
            mismatches.append(
                f"  training.{stage}.{spec_key}={spec_val!r}  vs  "
                f"finetuning_{stage}.yaml::{stage_key}={stage_val!r}"
            )

    # 2) Seed must match the experiment-wide value.
    inference_seed = cfg["inference"]["seed"]
    stage_seed = stage_cfg.get("seed")
    if stage_seed != inference_seed:
        mismatches.append(
            f"  inference.seed={inference_seed!r}  vs  "
            f"finetuning_{stage}.yaml::seed={stage_seed!r}"
        )

    # 3) SFT-only: the trainer's `model` must be the locked base. DPO is
    # exempt because it loads the SFT-merged fp16 checkpoint, not the base.
    if stage == "sft":
        spec_base = cfg["training"]["base_model"]
        stage_model = stage_cfg.get("model")
        if spec_base != stage_model:
            mismatches.append(
                f"  training.base_model={spec_base!r}  vs  "
                f"finetuning_sft.yaml::model={stage_model!r}"
            )

    if mismatches:
        raise PreflightError(
            f"preflight: finetuning_{stage}.yaml does not mirror config.yaml.\n"
            "Update both files to match, then re-run.\n"
            + "\n".join(mismatches)
        )


def _dotted_get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
