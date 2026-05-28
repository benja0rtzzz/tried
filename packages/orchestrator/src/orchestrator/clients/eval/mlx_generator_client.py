"""
MLX client for the generator. Loads a local MLX checkpoint and exposes
the same `generate(...)` interface as the Ollama-backed client at
`clients/dataset/generator_client.py`.

Used for all three eval conditions (base 4-bit, SFT-only 4-bit, SFT+DPO
4-bit). The checkpoint path is supplied by the caller; chat-template
wrapping, temperature, and max_tokens come from `config/config.yaml`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.improvement.shared.config import load_config
from orchestrator.prompts.generator import SYSTEM, build_user_prompt

_config: dict[str, Any] | None = None
_model: Any = None
_tokenizer: Any = None
_checkpoint_path: str | None = None


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
    """Generate a Triton candidate. Mirrors `generator_client.generate`.

    The MLX checkpoint is held module-level (loaded lazily on first call);
    the tokenizer's `apply_chat_template` wraps [system, user] before
    `mlx_lm.generate`, matching what the Ollama path does internally. The
    response is post-processed by the same `_strip_fences` helper.

    Set TRIED_MLX_CHECKPOINT to the local checkpoint directory path before use.
    """
    _ensure_loaded()
    cfg = _config["inference"]

    user_msg = build_user_prompt(
        pytorch_code, input_shapes, input_dtypes, prior_code, prior_advice
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(int(cfg["seed"]))
    sampler = make_sampler(temp=float(cfg["temperature"]))

    t0 = time.monotonic()
    response = mlx_generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=cfg["max_tokens"],
        sampler=sampler,
        verbose=False,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    # mlx_lm.generate returns the generated text only (not the prompt).
    prompt_tokens = len(_tokenizer.encode(prompt))
    completion_tokens = len(_tokenizer.encode(response))

    return GeneratorResult(
        triton_code=_strip_fences(response),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


def _ensure_loaded() -> None:
    global _config, _model, _tokenizer, _checkpoint_path

    checkpoint = os.environ.get("TRIED_MLX_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError(
            "TRIED_MLX_CHECKPOINT must be set to the local MLX checkpoint directory"
        )

    if _model is not None and checkpoint == _checkpoint_path:
        return  # already loaded

    from mlx_lm import load as mlx_load
    from transformers import AutoTokenizer

    _config = load_config()
    chat_template_id = _config["inference"]["chat_template"]

    _model, _ = mlx_load(checkpoint)
    # Use the HF tokenizer (not the mlx_lm bundled one) so apply_chat_template
    # uses the canonical ChatML template — same as training.
    _tokenizer = AutoTokenizer.from_pretrained(chat_template_id)
    _checkpoint_path = checkpoint


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
        if text.endswith("```"):
            text = text[:text.rfind("```")].rstrip()
    return text
