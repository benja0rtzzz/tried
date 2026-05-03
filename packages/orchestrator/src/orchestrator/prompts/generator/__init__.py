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
    prior_advice: str | None = None,
) -> str:
    """
    Render the user message for a generation request.

    prior_advice is the judge's fix_suggestion from the previous attempt.
    Pass None (or omit) on attempt 0.
    """
    advice_section = (
        f"The previous attempt failed. Fix advice:\n{prior_advice}\n"
        if prior_advice
        else ""
    )
    return _USER_TEMPLATE.format(
        pytorch_code=pytorch_code,
        input_shapes=", ".join(str(s) for s in input_shapes),
        input_dtypes=", ".join(input_dtypes),
        prior_advice_section=advice_section,
    )
