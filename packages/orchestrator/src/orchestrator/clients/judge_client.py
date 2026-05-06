"""
OpenAI client for the judge (o4-mini).
Classifies each attempt and returns a targeted fix suggestion.
Env vars required: OPENAI_API_KEY.

Raises RateLimitError when the OpenAI quota is exhausted (HTTP 429). The
orchestrator main loop catches this and stops cleanly so the run can be resumed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import openai
from pydantic import BaseModel
from shared.enums import JudgeClassification

from orchestrator.prompts.judge import SYSTEM, AttemptContext, build_user_prompt

_MODEL = "o4-mini-2025-04-16"


class RateLimitError(Exception):
    """OpenAI rate limit or quota exhausted."""


_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


class _JudgeResponse(BaseModel):
    classification: JudgeClassification
    fix_suggestion: str | None


@dataclass
class JudgeResult:
    classification: JudgeClassification
    fix_suggestion: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def judge(
    pytorch_code: str,
    attempts: list[AttemptContext],
) -> JudgeResult:
    """Call the o4-mini judge and return a classification with optional fix advice.

    attempts is the full list of attempt contexts in order; the last entry is
    the current attempt being judged.
    """
    user_msg = build_user_prompt(pytorch_code, attempts)

    t0 = time.monotonic()
    try:
        response = _get_client().beta.chat.completions.parse(
            model=_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=_JudgeResponse,
            reasoning_effort="high",
        )
    except openai.RateLimitError as exc:
        raise RateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc
    latency_ms = int((time.monotonic() - t0) * 1000)

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Judge response could not be parsed")

    usage = response.usage
    return JudgeResult(
        classification=parsed.classification,
        fix_suggestion=parsed.fix_suggestion,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
    )
