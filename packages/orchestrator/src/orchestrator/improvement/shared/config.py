"""
Loader for config/config.yaml. Authoritative source of truth for the
random seed, generation params, and training hyperparameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path("config/config.yaml")


def load_config(path: Path = _DEFAULT_PATH) -> dict[str, Any]:
    """Parse `config/config.yaml` and return it as a dict.

    Top-level keys: `schema_version`, `inference`, `openrouter`, `training`.
    Callers index into the nested dicts directly — no promotion to a typed
    model until we have a reason to.
    """
    with path.open() as f:
        return yaml.safe_load(f)
