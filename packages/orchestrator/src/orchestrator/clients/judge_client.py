"""
Azure OpenAI client for the judge (o4-mini).
Classifies each attempt and returns a targeted fix suggestion.
Env vars required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import AzureOpenAI
from pydantic import BaseModel

from orchestrator.prompts.judge import SYSTEM, AttemptContext, build_user_prompt
from shared.enums import JudgeClassification

_API_VERSION = "2025-01-01-preview"

_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=_API_VERSION,
        )
    return _client


def _get_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "o4-mini")


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
    """Call the Azure judge and return a classification with optional fix advice.

    attempts is the full list of attempt contexts in order; the last entry is
    the current attempt being judged.
    """
    user_msg = build_user_prompt(pytorch_code, attempts)

    t0 = time.monotonic()
    completion = _get_client().beta.chat.completions.parse(
        model=_get_deployment(),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        response_format=_JudgeResponse,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    parsed = completion.choices[0].message.parsed
    usage = completion.usage

    return JudgeResult(
        classification=parsed.classification,
        fix_suggestion=parsed.fix_suggestion,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
    )
