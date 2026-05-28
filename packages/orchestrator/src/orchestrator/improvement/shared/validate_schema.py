"""
Static schema check for the finetuning_{sft,dpo}.yaml files.

mlx-lm-lora silently ignores unknown YAML keys (train.py:1187-1189: it only
copies values for keys whose default is None). A typo like `lr` for
`learning_rate` therefore never raises — the run just trains at the default
learning rate. This module catches that class of bug before mlx-lm-lora is
invoked.

Companion to `preflight.py`:
- `preflight_check`  — value parity between config.yaml and stage YAMLs.
- `validate_schema`  — every key in the stage YAML is recognized by mlx-lm-lora,
                       and every enumerated value is in the accepted set.

Allow-lists are derived from the installed mlx-lm-lora package at import time
so they track upstream automatically when the dependency is bumped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.optimizers.schedulers as _mx_schedulers
import yaml
from mlx_lm_lora.train import CONFIG_DEFAULTS as _MLX_CONFIG_DEFAULTS

_STAGE_YAML = {
    "sft": Path("config/finetuning_sft.yaml"),
    "dpo": Path("config/finetuning_dpo.yaml"),
}

# Top-level keys mlx-lm-lora recognises. Anything else in a stage YAML is a typo.
_ALLOWED_TOP_LEVEL_KEYS = frozenset(_MLX_CONFIG_DEFAULTS.keys())

# Subkeys mlx-lm-lora reads off `lora_parameters` (see utils.py::from_pretrained
# and tuner/utils.py::linear_to_lora_layers).
_ALLOWED_LORA_PARAM_KEYS = frozenset({"rank", "dropout", "scale", "use_dora"})

# Enumerated values. Sourced by inspection of mlx-lm-lora train.py / xpo_trainer.py.
_ALLOWED_TRAIN_MODES = frozenset({
    "sft", "dpo", "cpo", "orpo", "grpo",
    "online_dpo", "ppo", "rlhf_reinforce", "xpo",
})
_ALLOWED_TRAIN_TYPES = frozenset({"lora", "dora"})
_ALLOWED_OPTIMIZERS = frozenset({"adam", "adamw", "muon"})
_ALLOWED_DPO_CPO_LOSS_TYPES = frozenset({"sigmoid", "hinge", "ipo", "dpop"})

# `lr_schedule.name` is resolved via getattr on mlx.optimizers.schedulers
# (see mlx_lm.tuner.utils::build_schedule). Any public callable there is valid.
_ALLOWED_SCHEDULE_NAMES = frozenset(
    name for name in dir(_mx_schedulers)
    if not name.startswith("_") and callable(getattr(_mx_schedulers, name))
)

# Keys that belong to one stage only — flag if present in the other.
_SFT_ONLY_KEYS = frozenset({"mask_prompt"})
_DPO_ONLY_KEYS = frozenset({"beta", "dpo_cpo_loss_type", "reference_model_path", "delta"})


class SchemaError(RuntimeError):
    """Raised when a stage YAML contains unrecognised keys or values."""


def validate_schema(stage: str) -> None:
    """Validate that finetuning_<stage>.yaml uses only keys mlx-lm-lora understands.

    Checks: top-level keys against CONFIG_DEFAULTS; lora_parameters subkeys;
    enumerated values for train_mode, train_type, optimizer, dpo_cpo_loss_type,
    lr_schedule.name; cross-stage key bleed (DPO-only keys in SFT, vice versa);
    train_mode consistency with the file's stage.
    """
    if stage not in _STAGE_YAML:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGE_YAML)}")

    with _STAGE_YAML[stage].open() as f:
        cfg = yaml.safe_load(f)

    errors: list[str] = []

    # 1) Top-level keys must be recognised by mlx-lm-lora.
    for key in cfg:
        if key not in _ALLOWED_TOP_LEVEL_KEYS:
            errors.append(
                f"  unknown top-level key {key!r} — not in mlx_lm_lora.train.CONFIG_DEFAULTS. "
                f"Likely a typo; mlx-lm-lora silently ignores it."
            )

    # 2) lora_parameters subkeys.
    lora_params = cfg.get("lora_parameters") or {}
    for key in lora_params:
        if key not in _ALLOWED_LORA_PARAM_KEYS:
            errors.append(
                f"  unknown lora_parameters subkey {key!r} — "
                f"allowed: {sorted(_ALLOWED_LORA_PARAM_KEYS)}"
            )

    # 3) Enumerated values.
    _check_enum(cfg, "train_mode",         _ALLOWED_TRAIN_MODES,         errors)
    _check_enum(cfg, "train_type",         _ALLOWED_TRAIN_TYPES,         errors)
    _check_enum(cfg, "optimizer",          _ALLOWED_OPTIMIZERS,          errors)
    _check_enum(cfg, "dpo_cpo_loss_type",  _ALLOWED_DPO_CPO_LOSS_TYPES,  errors)

    sched = cfg.get("lr_schedule") or {}
    sched_name = sched.get("name") if isinstance(sched, dict) else None
    if sched_name is not None and sched_name not in _ALLOWED_SCHEDULE_NAMES:
        errors.append(
            f"  lr_schedule.name={sched_name!r} is not an attribute of "
            f"mlx.optimizers.schedulers. Valid: {sorted(_ALLOWED_SCHEDULE_NAMES)}"
        )

    # 4) Cross-stage key bleed.
    forbidden = _DPO_ONLY_KEYS if stage == "sft" else _SFT_ONLY_KEYS
    for key in forbidden & cfg.keys():
        errors.append(
            f"  key {key!r} appears in finetuning_{stage}.yaml but is "
            f"only valid for the other stage."
        )

    # 5) train_mode must match the file's stage.
    if cfg.get("train_mode") not in (stage, None):
        errors.append(
            f"  train_mode={cfg.get('train_mode')!r} in finetuning_{stage}.yaml; "
            f"expected {stage!r}."
        )

    if errors:
        raise SchemaError(
            f"validate_schema: finetuning_{stage}.yaml has unrecognised keys or values.\n"
            + "\n".join(errors)
        )


def _check_enum(
    cfg: dict[str, Any], key: str, allowed: frozenset[str], errors: list[str]
) -> None:
    value = cfg.get(key)
    if value is not None and value not in allowed:
        errors.append(
            f"  {key}={value!r} is not in the accepted set {sorted(allowed)}"
        )
