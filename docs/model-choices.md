# Model Choices

## Generator (local LLM on MacBook)

**Decision made for:** Qwen2.5-Coder-14B.

## Judge (OpenAI)

**Current choice:** Codex CLI profile `gpt-5-3-codex`.

The judge only classifies outcomes and provides retry advice. It never generates Triton kernels.

The judge is invoked through `codex exec` with `--output-schema`, `--output-last-message`, `--json`, `--ephemeral`, and a read-only sandbox. The switch from the direct `o4-mini` client was cost-driven: it keeps the same judge role and structured-output contract while using the local Codex CLI profile already available for the project. The current profile is hardcoded in `packages/orchestrator/src/orchestrator/clients/judge_client.py`.
