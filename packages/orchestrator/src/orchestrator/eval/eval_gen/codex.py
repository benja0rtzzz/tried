"""Stage 2 — `codex exec` subprocess wrapper.

Each spec gets one hermetic Codex call (no tools, no context bleed
between specs). The CLI invocation is pinned at module top so changes
to Codex's flags only need a single edit.

We pass `--output-last-message <tempfile>` so Codex writes ONLY its final
message to a file. That sidesteps the problem of parsing the verbose
event stream Codex prints to stdout (which can include multiple fenced
blocks during reasoning).

Response handling:
  - Extracts the first ```python ... ``` fenced block from the
    last-message file.
  - If the message is a `{"error": "..."}` JSON object, returns it as
    an UnrealizableSpec result so the driver can log and skip.
  - Anything else is a ParseError.

If `codex exec` fails (non-zero exit, timeout) we surface the stderr in
a CodexCallError so the driver can decide whether to retry.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Pinned at module top so the driver doesn't need to know CLI details.
# Model + reasoning effort come from the profile (~/.codex/config.toml).
CODEX_BIN = "codex"
CODEX_PROFILE = "oca-gpt-5-5"
# Static args; -o <path> is appended per call.
CODEX_EXEC_ARGS_STATIC: list[str] = [
    "exec",
    "-p", CODEX_PROFILE,
    "--ephemeral",            # don't persist a session per call
    "--skip-git-repo-check",  # allow running anywhere
    "--sandbox", "read-only", # defense in depth — Codex shouldn't run tools anyway
    "--color", "never",
]
CODEX_TIMEOUT_S = 300  # high-reasoning calls can take a couple minutes


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CodexCallError(RuntimeError):
    """`codex exec` failed (non-zero return, timeout, etc.)."""


@dataclass(frozen=True)
class ParseError:
    """Codex returned something we couldn't extract code from."""
    raw: str
    detail: str


@dataclass(frozen=True)
class UnrealizableSpec:
    """Codex declined to write code; spec should be skipped."""
    explanation: str


@dataclass(frozen=True)
class GeneratedCode:
    """Successful extraction — `code` is the function source ready for
    AST validation in stage 3."""
    code: str


ParseResult = GeneratedCode | UnrealizableSpec | ParseError


# ---------------------------------------------------------------------------
# Subprocess call
# ---------------------------------------------------------------------------

def call_codex(prompt: str) -> str:
    """Run `codex exec` once with `prompt` on stdin. Returns the contents
    of the last-message file (just the model's final answer, not the
    full event stream). Raises CodexCallError on non-zero exit or
    timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        out_path = Path(f.name)

    try:
        try:
            result = subprocess.run(
                [CODEX_BIN, *CODEX_EXEC_ARGS_STATIC, "-o", str(out_path), "-"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=CODEX_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise CodexCallError(f"codex timed out after {CODEX_TIMEOUT_S}s") from e
        except FileNotFoundError as e:
            raise CodexCallError(
                f"codex binary '{CODEX_BIN}' not found on PATH"
            ) from e

        if result.returncode != 0:
            raise CodexCallError(
                f"codex exec exited {result.returncode}: {result.stderr.strip()}"
            )
        return out_path.read_text()
    finally:
        out_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# First fenced ```python or ```py block. Matches multiline non-greedy.
_PYTHON_FENCE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)

# JSON `{"error": "..."}` — first top-level object that has only an
# `error` field. Matches multiline; we don't try to parse general JSON
# soup, just the locked escape hatch.
_ERROR_JSON = re.compile(
    r"\{\s*\"error\"\s*:\s*\"[^\"]*\"\s*\}",
    flags=re.DOTALL,
)


def parse_response(response: str) -> ParseResult:
    """Extract code or error from a Codex response. Tries the python
    code fence first (the locked happy path), then falls through to the
    error-JSON escape hatch, then ParseError."""
    m = _PYTHON_FENCE.search(response)
    if m:
        code = m.group(1).strip()
        if not code:
            return ParseError(raw=response, detail="empty python fence")
        return GeneratedCode(code=code)

    m = _ERROR_JSON.search(response)
    if m:
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return ParseError(raw=response, detail=f"error-JSON malformed: {e}")
        return UnrealizableSpec(explanation=str(payload.get("error", "<empty>")))

    return ParseError(raw=response, detail="no python fence or error JSON found")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def synthesize(prompt: str) -> ParseResult:
    """One-shot: call Codex + parse the response. Raises CodexCallError
    on subprocess failure; returns ParseResult on parse outcome."""
    return parse_response(call_codex(prompt))
