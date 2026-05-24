"""
Codex CLI client for the judge (profile gpt-5-3-codex).

Classifies each attempt and returns a targeted fix suggestion. Talks to a
local `codex exec` subprocess with --output-schema for structured output
and --json for token-usage extraction.

Raises RateLimitError when the CLI returns a rate-limit / quota error so
the orchestrator main loop can stop cleanly and resume later.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator
from shared.enums import JudgeClassification, JudgeRepairAction, JudgeRootCause

from orchestrator.prompts.judge import SYSTEM, AttemptContext, build_user_prompt

_PROFILE = "gpt-5-3-codex"
_TIMEOUT_S = 180

_RATE_LIMIT_HINTS = ("rate limit", "rate_limit", "quota", "429", "too many requests")


class RateLimitError(Exception):
    """Codex CLI hit a rate-limit / quota wall."""


class _JudgeResponse(BaseModel):
    classification: JudgeClassification
    root_cause: JudgeRootCause | None
    repair_action: JudgeRepairAction | None
    fix_suggestion: str | None

    @model_validator(mode="after")
    def _labels_match_outcome(self) -> _JudgeResponse:
        if self.classification == JudgeClassification.COMPILED_CORRECT:
            if (
                self.root_cause is not None
                or self.repair_action is not None
                or self.fix_suggestion is not None
            ):
                raise ValueError(
                    "compiled_correct judge responses must not include repair labels"
                )
            return self
        if (
            self.root_cause is None
            or self.repair_action is None
            or self.fix_suggestion is None
        ):
            raise ValueError(
                "failed judge responses require root_cause, repair_action, "
                "and fix_suggestion"
            )
        return self


@dataclass
class JudgeResult:
    classification: JudgeClassification
    root_cause: JudgeRootCause | None
    repair_action: JudgeRepairAction | None
    fix_suggestion: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def _enforce_strict_schema(node: Any) -> None:
    """Add additionalProperties:false to every object node — required by
    the strict structured-output validator the codex CLI proxies to."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _enforce_strict_schema(v)
    elif isinstance(node, list):
        for v in node:
            _enforce_strict_schema(v)


_schema_cache: Path | None = None


def _schema_path() -> Path:
    global _schema_cache
    if _schema_cache is None or not _schema_cache.exists():
        schema = _JudgeResponse.model_json_schema()
        _enforce_strict_schema(schema)
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(schema, f)
        f.flush()
        _schema_cache = Path(f.name)
    return _schema_cache


def _build_prompt(pytorch_code: str, attempts: list[AttemptContext]) -> str:
    user_msg = build_user_prompt(pytorch_code, attempts)
    return f"<system>\n{SYSTEM}\n</system>\n\n<user>\n{user_msg}\n</user>"


def _is_rate_limit(stderr: str) -> bool:
    s = stderr.lower()
    return any(h in s for h in _RATE_LIMIT_HINTS)


def _parse_usage(jsonl_stdout: str) -> tuple[int, int]:
    in_tokens = 0
    out_tokens = 0
    for line in jsonl_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            in_tokens = int(usage.get("input_tokens") or 0)
            out_tokens = int(usage.get("output_tokens") or 0)
    return in_tokens, out_tokens


def judge(
    pytorch_code: str,
    attempts: list[AttemptContext],
) -> JudgeResult:
    """Call the codex CLI judge and return a classification with optional fix advice.

    attempts is the full list of attempt contexts in order; the last entry is
    the current attempt being judged.
    """
    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    prompt = _build_prompt(pytorch_code, attempts)

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "last_message.txt"
        cmd = [
            "codex", "exec",
            "--profile", _PROFILE,
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox", "read-only",
            "--output-schema", str(_schema_path()),
            "--output-last-message", str(out_path),
            "--json",
            "--color", "never",
            prompt,
        ]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codex exec timed out after {_TIMEOUT_S}s") from exc
        latency_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            if _is_rate_limit(proc.stderr):
                raise RateLimitError(f"codex CLI rate limit: {proc.stderr[-300:]}")
            raise RuntimeError(
                f"codex exec exited {proc.returncode}: {proc.stderr[-400:]}"
            )

        last_msg = out_path.read_text() if out_path.exists() else ""
        if not last_msg.strip():
            raise RuntimeError("codex exec produced empty last_message")

        try:
            payload = json.loads(last_msg)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"codex output failed to parse as JSON: {exc}; raw={last_msg[:300]!r}"
            ) from exc
        parsed = _JudgeResponse.model_validate(payload)

        in_tokens, out_tokens = _parse_usage(proc.stdout)

    return JudgeResult(
        classification=parsed.classification,
        root_cause=parsed.root_cause,
        repair_action=parsed.repair_action,
        fix_suggestion=parsed.fix_suggestion,
        prompt_tokens=in_tokens,
        completion_tokens=out_tokens,
        latency_ms=latency_ms,
    )
