"""
OpenRouter client for the generator (reference baseline eval conditions).

Uses the standard openai SDK pointed at OpenRouter's OpenAI-compatible API.
All connection params come from config/experiment.yaml (openrouter:); the API
key is read from OPENROUTER_API_KEY, which must be set in the orchestrator
.env before running any eval condition that uses this client.

Set TRIED_OPENROUTER_MODEL to a key under openrouter.models in
experiment.yaml (e.g. "llama" or "deepseek") to select which model runs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from orchestrator.improvement.config import load_config
from orchestrator.prompts.generator import SYSTEM, build_user_prompt

_client: OpenAI | None = None
_config: dict[str, Any] | None = None


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
    prior_code: str | None = None,
    prior_advice: str | None = None,
) -> GeneratorResult:
    """Generate a Triton candidate via OpenRouter. Mirrors mlx_generator_client.generate.

    Reads TRIED_OPENROUTER_MODEL from the environment to select the model key
    (must match a key under openrouter.models in experiment.yaml). The OpenAI
    client is initialised lazily and reused across calls.
    """
    _ensure_client()
    cfg = _config["openrouter"]

    model_key = os.environ.get("TRIED_OPENROUTER_MODEL")
    if not model_key:
        raise RuntimeError(
            "TRIED_OPENROUTER_MODEL must be set to a model key defined in "
            "config/experiment.yaml openrouter.models (e.g. 'llama' or 'deepseek')"
        )
    try:
        model_id = cfg["models"][model_key]["model_id"]
    except KeyError:
        raise RuntimeError(
            f"TRIED_OPENROUTER_MODEL={model_key!r} not found under "
            "openrouter.models in config/experiment.yaml"
        )

    user_msg = build_user_prompt(
        pytorch_code, input_shapes, input_dtypes, prior_code, prior_advice
    )

    t0 = time.monotonic()
    response = _client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=float(cfg["temperature"]),
        max_completion_tokens=int(cfg["max_completion_tokens"]),
        seed=int(cfg["seed"]),
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    content = response.choices[0].message.content or ""
    usage = response.usage

    return GeneratorResult(
        triton_code=_strip_fences(content),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
    )


def _ensure_client() -> None:
    global _client, _config
    if _client is not None:
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set (add it to packages/orchestrator/.env)"
        )

    _config = load_config()
    _client = OpenAI(
        api_key=api_key,
        base_url=_config["openrouter"]["base_url"],
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
        if text.endswith("```"):
            text = text[:text.rfind("```")].rstrip()
    return text
