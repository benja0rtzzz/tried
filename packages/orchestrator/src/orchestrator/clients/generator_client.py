"""
Ollama client for the generator (qwen2.5-coder:14b).
Thin wrapper: calls the model, strips markdown fences, returns typed result.
Error handling is left to the agent loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import ollama

from orchestrator.prompts.generator import SYSTEM, build_user_prompt

_MODEL = "qwen2.5-coder:14b"

@dataclass
class GeneratorResult:
    triton_code:       str
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        int


def generate(
    pytorch_code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    prior_advice: str | None = None,
) -> GeneratorResult:
    """Call the local LLM and return the generated Triton code with metadata.

    prior_advice is the judge's fix_suggestion from the previous attempt.
    Pass None (or omit) on attempt 0.
    """
    user_msg = build_user_prompt(pytorch_code, input_shapes, input_dtypes, prior_advice)

    t0 = time.monotonic()
    response = ollama.chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        options={"temperature": 0},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    return GeneratorResult(
        triton_code=_strip_fences(response.message.content or ""),
        prompt_tokens=response.prompt_eval_count or 0,
        completion_tokens=response.eval_count or 0,
        latency_ms=latency_ms,
    )


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped its output."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
        if text.endswith("```"):
            text = text[:text.rfind("```")].rstrip()
    return text
