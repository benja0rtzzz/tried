# Dev Setup — Lenovo LOQ (RTX 4060)

## Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- NVIDIA drivers + CUDA 12.x
- [UV](https://docs.astral.sh/uv/) installed
- Verify GPU is visible: `nvidia-smi`

## Clone and install workspace

```bash
git clone <repo-url> tried
cd tried
uv sync --package tried-verification
uv sync --package tried-shared
```

This installs only the Lenovo-relevant packages. The `tried-orchestrator` package (Ollama, MLX) is **not** installed here.

## PyTorch + CUDA

UV will install the CPU PyTorch by default. Override for CUDA:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Confirm CUDA is working:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Pin to dGPU

The Ryzen iGPU will silently steal CUDA work if not pinned. Always export before running:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Add this to `.bashrc` or the systemd service file for the FastAPI server.

## Running the verification server

```bash
CUDA_VISIBLE_DEVICES=0 uv run uvicorn verification.server:app --host 0.0.0.0 --port 8000
```

The MacBook orchestrator should be able to reach this at the Lenovo's LAN IP.

## API key

Set the shared secret so only the orchestrator can call the server:

```bash
export VERIFICATION_API_KEY="..."
```

Use the same value on the MacBook side.

## Notes

- Do not run other GPU workloads while the server is handling benchmark requests.
- Do not benchmark on battery power.
- Verify with `nvidia-smi` that only the expected CUDA process is running during benchmarks.
