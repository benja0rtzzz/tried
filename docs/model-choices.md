# Model Choices

## Generator (local LLM on MacBook)

**Decision made for:** Qwen2.5-Coder-14B.

## Judge (Google AI Studio)

**Current choice:** Gemini 2.5 Flash.

The judge only classifies outcomes and provides retry advice. It never generates Triton kernels.

Thinking is enabled (`thinking_budget=1024`) for fix suggestion quality. Temperature is 0 for deterministic classification. Structured output enforced via `response_mime_type="application/json"` + `response_schema`. The free tier (1,500 req/day) covers the full experiment (~600 calls).