"""
Google AI Studio client for the judge (Gemini 2.5 Flash).
Classifies each attempt and returns a targeted fix suggestion.
Env var required: GEMINI_API_KEY. Optional: GEMINI_MODEL (default: gemini-2.5-flash).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import BaseModel

from orchestrator.prompts.judge import SYSTEM, AttemptContext, build_user_prompt
from shared.enums import JudgeClassification

_DEFAULT_MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _get_model() -> str:
    return os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)


class _JudgeResponse(BaseModel):
    classification: JudgeClassification
    fix_suggestion: str | None


@dataclass
class JudgeResult:
    classification: JudgeClassification
    fix_suggestion: str | None
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        int


def judge(
    pytorch_code: str,
    attempts: list[AttemptContext],
) -> JudgeResult:
    """Call the Gemini judge and return a classification with optional fix advice.

    attempts is the full list of attempt contexts in order; the last entry is
    the current attempt being judged.
    """
    user_msg = build_user_prompt(pytorch_code, attempts)

    t0 = time.monotonic()
    response = _get_client().models.generate_content(
        model=_get_model(),
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=_JudgeResponse,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    raw = response.text
    if not raw:
        raise RuntimeError("Judge response was empty")

    parsed = _JudgeResponse.model_validate_json(raw)
    usage = response.usage_metadata

    return JudgeResult(
        classification=parsed.classification,
        fix_suggestion=parsed.fix_suggestion,
        prompt_tokens=usage.prompt_token_count or 0 if usage else 0,
        completion_tokens=usage.candidates_token_count or 0 if usage else 0,
        latency_ms=latency_ms,
    )
