"""
Generator prompt loader. Do not modify the .txt files once data collection has started.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent

SYSTEM: str = (_DIR / "generator_system.txt").read_text()
_USER_TEMPLATE: str = (_DIR / "generator_user.txt").read_text()

__all__ = ["SYSTEM", "build_user_prompt"]


def build_user_prompt(
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    prior_code: str | None = None,
    prior_advice: str | None = None,
) -> str:
    """
    Render the user message for a generation request.

    prior_code is the Triton code produced by the previous attempt.
    prior_advice is the judge's fix_suggestion for that attempt.
    Both are None on attempt 0 and passed together on retries.
    """
    if prior_code and prior_advice:
        prior_section = (
            "\nThe previous attempt produced this Triton code:\n"
            f"```python\n{prior_code}\n```\n\n"
            "It failed. Apply the following fix to that code, preserving the "
            "parts that were already correct — do not rewrite from scratch:\n"
            f"{prior_advice}\n"
        )
    else:
        prior_section = ""

    return _USER_TEMPLATE.format(
        pytorch_code=pytorch_code,
        input_shapes=", ".join(str(s) for s in input_shapes),
        input_dtypes=", ".join(input_dtypes),
        prior_attempt_section=prior_section,
    )
