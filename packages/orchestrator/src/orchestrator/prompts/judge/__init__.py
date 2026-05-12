"""Judge prompt loader."""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

_DIR = Path(__file__).parent

SYSTEM: str = (_DIR / "judge_system.txt").read_text()
_USER_TEMPLATE: str = (_DIR / "judge_user.txt").read_text()

__all__ = ["SYSTEM", "build_user_prompt", "AttemptContext"]


class AttemptContext(TypedDict):
    attempt_n:          int
    triton_code:        str
    compile_status:     str        # "success" | "failed"
    compile_error:      str | None
    run_error:          str | None
    correctness_status: str | None # "passed" | "failed" | None
    max_abs_diff:       float | None
    pct_exceeding:      float | None
    fix_suggestion:     str | None  # None for the current attempt


def _render_attempt(a: AttemptContext, is_current: bool) -> str:
    label = f"--- Attempt {a['attempt_n']}{' (current)' if is_current else ''} ---"

    compile_line = f"Compile: {a['compile_status']}"
    if a["compile_error"]:
        compile_line += f" — {a['compile_error']}"

    correctness_line = ""
    if a["correctness_status"] is not None:
        correctness_line = f"Correctness: {a['correctness_status']}"
        if a["correctness_status"] == "failed" and a["max_abs_diff"] is not None:
            correctness_line += (
                f" — max_abs_diff={a['max_abs_diff']:.4f},"
                f" pct_exceeding={a['pct_exceeding']:.1f}%"
            )

    run_error_line = ""
    if a["run_error"]:
        run_error_line = f"Run error: {a['run_error']}"

    fix_line = ""
    if a["fix_suggestion"]:
        fix_line = f"Fix suggested: {a['fix_suggestion']}"

    parts = [
        label,
        f"Triton code:\n{a['triton_code']}",
        compile_line,
    ]
    for line in (run_error_line, correctness_line, fix_line):
        if line:
            parts.append(line)

    return "\n".join(parts)


def build_user_prompt(
    pytorch_code: str,
    attempts: list[AttemptContext],
) -> str:
    """Render the user message for a judge request.

    attempts is a list of all attempts in order; the last entry is the current
    one being judged (fix_suggestion should be None for it).
    """
    blocks = [
        _render_attempt(a, is_current=(i == len(attempts) - 1))
        for i, a in enumerate(attempts)
    ]
    return _USER_TEMPLATE.format(
        pytorch_code=pytorch_code,
        attempts_section="\n\n".join(blocks),
    )
