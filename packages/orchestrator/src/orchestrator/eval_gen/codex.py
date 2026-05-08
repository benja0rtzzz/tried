"""Stage 2 — `codex exec` subprocess wrapper.

Each spec gets one hermetic Codex call (no tools, no context bleed
between specs). The CLI invocation is pinned at module top so changes
to Codex's flags only need a single edit.

Response handling:
  - Extracts the first ```python ... ``` fenced block.
  - If the response is a `{"error": "..."}` JSON object, returns it as
    an UnrealizableSpec result so the driver can log and skip.
  - Anything else is a ParseError.

If `codex exec` fails (non-zero exit, timeout) we surface the stderr in
a CodexCallError so the driver can decide whether to retry.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass


# Tweak if the Codex CLI's flags change. Pinned here so the driver
# doesn't need to know.
CODEX_BIN = "codex"
CODEX_EXEC_ARGS: list[str] = [
    "exec",
    "--model", "gpt-5.5",
    "--reasoning-effort", "high",
    "--no-tools",  # <-- the flag name; adjust if Codex CLI uses a different name
    "-",  # read prompt from stdin
]
CODEX_TIMEOUT_S = 300  # high-reasoning calls can take ~minute-ish


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
    """Run `codex exec` once with `prompt` on stdin. Returns stdout text.
    Raises CodexCallError on non-zero exit or timeout."""
    try:
        result = subprocess.run(
            [CODEX_BIN, *CODEX_EXEC_ARGS],
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
    return result.stdout


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
