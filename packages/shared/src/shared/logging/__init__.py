"""
Shared logger for both machines.
Set TRIED_ROLE=orchestrator on the MacBook and TRIED_ROLE=verification on the Lenovo
so every log line is stamped with which machine produced it.
"""
from __future__ import annotations

import logging as _logging
import os

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
