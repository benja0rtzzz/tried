"""Dedup checks against the locked eval holdout set."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from shared.logging import get_logger

logger = get_logger(__name__)

_EXAMPLE_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000eab")


def derive_example_id(pytorch_code: str) -> str:
    """UUIDv5 rule used by the eval set: uuid5(namespace, sha256(code))."""
    digest = hashlib.sha256(pytorch_code.encode()).hexdigest()
    return str(uuid.uuid5(_EXAMPLE_NAMESPACE, digest))


def _extract_pytorch_code(row: dict) -> str | None:
    code = row.get("pytorch_code")
    if isinstance(code, str):
        return code

    source = row.get("source")
    if isinstance(source, dict):
        nested = source.get("pytorch_code")
        if isinstance(nested, str):
            return nested

    return None


class EvalDedup:
    def __init__(self, eval_path: Path) -> None:
        self._eval_codes: set[str] = set()
        self._eval_example_ids: set[str] = set()

        if not eval_path.exists():
            raise FileNotFoundError(f"locked eval set not found: {eval_path}")

        with eval_path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(row, dict):
                    continue

                code = _extract_pytorch_code(row)
                if code is not None:
                    self._eval_codes.add(code)

                example_id = row.get("example_id")
                if isinstance(example_id, str):
                    self._eval_example_ids.add(example_id)

        logger.info(
            "loaded eval dedup index: %d codes, %d example_ids",
            len(self._eval_codes),
            len(self._eval_example_ids),
        )

    def is_collision(self, pytorch_code: str, example_id: str) -> str | None:
        if pytorch_code in self._eval_codes:
            return "byte-identical pytorch_code collision with locked eval set"
        if example_id in self._eval_example_ids:
            return "example_id collision with locked eval set"
        return None
