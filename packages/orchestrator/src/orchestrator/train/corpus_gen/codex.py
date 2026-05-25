"""Codex CLI wrapper for training-corpus skeleton synthesis."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.enums import Dtype

from .patterns import SkeletonSpec
from .prompt import render

_PROFILE = "gpt-5-3-codex"
_TIMEOUT_S = 300

_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "quota", "429", "too many requests")


class RateLimitError(Exception):
    """Codex CLI hit a rate-limit or quota wall."""


class CodexCallError(Exception):
    """Codex subprocess failed for a non-rate-limit reason."""


class ParseError(Exception):
    """Codex output did not match the expected structured schema."""


class SkeletonResponse(BaseModel):
    pytorch_code: str
    input_shapes: list[list[int]]
    input_dtypes: list[Dtype]
    rationale: str


_schema_cache: Path | None = None


def _enforce_strict_schema(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for value in node.values():
            _enforce_strict_schema(value)
    elif isinstance(node, list):
        for value in node:
            _enforce_strict_schema(value)


def _schema_path() -> Path:
    global _schema_cache
    if _schema_cache is None or not _schema_cache.exists():
        schema = SkeletonResponse.model_json_schema()
        _enforce_strict_schema(schema)
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(schema, fh)
        fh.flush()
        _schema_cache = Path(fh.name)
    return _schema_cache


def _build_prompt(spec: SkeletonSpec) -> str:
    system, user = render(spec)
    return f"<system>\n{system}\n</system>\n\n<user>\n{user}\n</user>"


def _is_rate_limit(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _RATE_LIMIT_HINTS)


def _tail(label: str, text: str, limit: int = 1200) -> str:
    text = text.strip()
    if not text:
        return ""
    return f"{label}={text[-limit:]}"


def _failure_details(proc: subprocess.CompletedProcess[str], last_message_path: Path) -> str:
    last_message = last_message_path.read_text() if last_message_path.exists() else ""
    parts = [
        _tail("stderr", proc.stderr),
        _tail("stdout", proc.stdout),
        _tail("last_message", last_message),
    ]
    return " | ".join(part for part in parts if part) or "no stderr/stdout/last_message captured"


def synthesize(spec: SkeletonSpec) -> SkeletonResponse:
    if shutil.which("codex") is None:
        raise CodexCallError("codex CLI not found on PATH")

    prompt = _build_prompt(spec)
    with tempfile.TemporaryDirectory() as td:
        last_message_path = Path(td) / "last_message.txt"
        cmd = [
            "codex",
            "exec",
            "--profile",
            _PROFILE,
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(_schema_path()),
            "--output-last-message",
            str(last_message_path),
            "--json",
            "--color",
            "never",
            prompt,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexCallError(f"codex exec timed out after {_TIMEOUT_S}s") from exc
        except OSError as exc:
            raise CodexCallError(f"failed to execute codex CLI: {exc}") from exc

        if proc.returncode != 0:
            details = _failure_details(proc, last_message_path)
            if _is_rate_limit("\n".join([proc.stderr, proc.stdout, details])):
                raise RateLimitError(f"codex CLI rate limit: {details}")
            raise CodexCallError(f"codex exec exited {proc.returncode}: {details}")

        raw = last_message_path.read_text() if last_message_path.exists() else ""

    if not raw.strip():
        raise ParseError("codex exec produced empty last_message")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"codex output is not valid JSON: {exc}; raw={raw[:300]!r}"
        ) from exc

    try:
        return SkeletonResponse.model_validate(payload)
    except ValidationError as exc:
        raise ParseError(f"codex output failed schema validation: {exc}") from exc
