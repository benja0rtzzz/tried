# Dev Setup — M4 MacBook Pro

## Prerequisites

- macOS with Xcode Command Line Tools
- [UV](https://docs.astral.sh/uv/) installed
- [Ollama](https://ollama.com/) installed and running

## Clone and install workspace

```bash
git clone <repo-url> tried
cd tried
uv sync --all-packages
```

This installs `tried-shared` and `tried-orchestrator` into a single virtualenv at `.venv/`.

The `tried-verification` package is **not** installed here — it's Lenovo-only (CUDA required).

## Pull generator models via Ollama

```bash
ollama pull qwen2.5-coder:14b
```

## Azure OpenAI (judge)

Set environment variables (do not commit these):

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="o4-mini"
```

## Running the orchestrator

```bash
uv run python -m orchestrator.main
```

## Fine-tuning (MLX)

Fine-tuning runs are separate from the agent loop. See `packages/orchestrator/src/orchestrator/improvement/` for scripts. Requires `mlx-lm` (installed via the orchestrator package deps).

## Notes

- The MacBook never runs Triton code. All CUDA work goes to the Lenovo.
- Keep Ollama running in the background before starting the orchestrator.
