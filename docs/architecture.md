# Architecture

## Two-machine setup

| Machine | Role | Key software |
|---|---|---|
| M4 MacBook Pro (48 GB) | Orchestrator | Ollama, MLX, UV, httpx |
| Lenovo LOQ RTX 4060 8 GB | Verification server | Triton, PyTorch+CUDA, FastAPI, uvicorn |

## Agent loop

```
PyTorch input
    │
    ▼
Orchestrator (MacBook)
    ├─ calls Ollama → local LLM generates Triton candidate
    ├─ HTTP POST /compile  ──► Lenovo FastAPI
    ├─ HTTP POST /run      ──► Lenovo FastAPI
    ├─ HTTP POST /benchmark──► Lenovo FastAPI
    │       results ◄──────────────────────
    ├─ if wrong/slow: call Azure judge (o4-mini) → advice
    └─ retry (max N=5), record every attempt
```

## FastAPI endpoint contract (Lenovo)

All endpoints accept and return JSON. Triton/PyTorch source is sent as a string field. Three endpoints: `POST /compile`, `POST /run`, `POST /benchmark`.

<!-- TODO: define request/response schemas once the dataset schema is consolidated. -->

## Communication

- Transport: plain HTTP (LAN). Add a shared API key header (`X-API-Key`) for basic auth.
- The MacBook always initiates; the Lenovo never calls back.
- The Lenovo server runs with `CUDA_VISIBLE_DEVICES=0` to pin to the dGPU.
