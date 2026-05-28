"""
Shared logger for both machines.
Set TRIED_ROLE=orchestrator on the MacBook and TRIED_ROLE=verification on the Lenovo
so every log line is stamped with which machine produced it.

The role (and any other env vars) is read at import time. If TRIED_ROLE is not
already set in the process environment, we walk parent directories looking for
`packages/orchestrator/.env` or `packages/verification/.env` and load whichever
exists first. python-dotenv is a soft dependency — if it's not installed (e.g.
on a stripped-down deployment), the env discovery is skipped and the role
falls back to "unknown".
"""
from __future__ import annotations

import logging as _logging
import os
from pathlib import Path


def _try_load_dotenv() -> None:
    """Auto-load packages/{orchestrator,verification}/.env walking up from CWD."""
    if os.getenv("TRIED_ROLE"):
        return  # already set in shell env; respect it
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        for sub in ("packages/orchestrator/.env", "packages/verification/.env"):
            p = d / sub
            if p.exists():
                load_dotenv(p, override=False)
                return


_try_load_dotenv()
_ROLE = os.getenv("TRIED_ROLE", "unknown")
_FORMAT = "[%(asctime)s] [%(role)s] %(levelname)-8s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RoleFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        record.role = _ROLE  # type: ignore[attr-defined]
        return True


def get_logger(name: str) -> _logging.Logger:
    """Return a logger configured with the shared format.

    Call once per module: logger = get_logger(__name__)
    """
    logger = _logging.getLogger(name)
    if not logger.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        handler.addFilter(_RoleFilter())
        logger.addHandler(handler)
        logger.setLevel(_logging.DEBUG)
        logger.propagate = False
    return logger
