# Model Choices

## Generator (local LLM on MacBook)

**Decision made for:** Qwen2.5-Coder-14B.

## Judge (OpenAI)

**Current choice:** o4-mini.

The judge only classifies outcomes and provides retry advice. It never generates Triton kernels.

Reasoning is enabled (`reasoning_effort="high"`) for fix suggestion quality. Structured output enforced via `response_format` Pydantic model (`.beta.chat.completions.parse`). Estimated cost for the full experiment (~600 calls): ~$6–11 depending on reasoning token usage. Model and reasoning effort are hardcoded (`o4-mini`, `high`).