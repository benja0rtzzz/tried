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

## Google AI Studio (judge)

Get a free API key at [aistudio.google.com](https://aistudio.google.com/). Set environment variables (do not commit these):

```bash
export GEMINI_API_KEY="..."
# Optional: pin to a specific model version (default: gemini-2.5-flash)
export GEMINI_MODEL="gemini-2.5-flash"
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
