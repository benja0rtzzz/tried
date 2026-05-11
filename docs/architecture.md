# Architecture

## Two-machine setup

| Machine | Role | Key software |
|---|---|---|
| M4 MacBook Pro (48 GB) | Orchestrator | Ollama, Codex CLI, MLX, UV, httpx |
| Lenovo LOQ RTX 4060 8 GB | Verification server | Triton, PyTorch+CUDA, FastAPI, uvicorn |

## Dataset Agent Loop

The training dataset pipeline consumes preflight-safe synthetic skeleton rows from `data/preflight_safe.jsonl`. The eager-vs-Inductor preflight is done upstream by `orchestrator.dataset.preflight_driver`; rows reaching `orchestrator.dataset.main` are assumed clean.

```
PyTorch input
    │
    ▼
Orchestrator (MacBook)
    ├─ calls Ollama → local LLM generates Triton candidate
    ├─ HTTP POST /compile   ──► Lenovo FastAPI  (synchronous)
    ├─ HTTP POST /run       ──► Lenovo FastAPI  (synchronous)
    ├─ if failed: call Codex CLI judge (`gpt-5-3-codex`) → advice
    └─ retry (max N=5), record every attempt
```

Dataset rows record compile, correctness, judge classification, and retry advice. They do **not** record benchmark timings or speedups; benchmarking is eval-only.

## Eval Run Loop

The eval pipeline consumes the locked holdout at `eval/holdout/synthetic_fusions.jsonl` and measures raw generator capability: one attempt per row, no judge, no retry. A live `/preflight` is run first to catch regressions in accepted holdout rows.

```
EvalCorpusRecord
    │
    ▼
Orchestrator (MacBook)
    ├─ HTTP POST /preflight ──► Lenovo FastAPI
    ├─ calls Ollama → one Triton candidate
    ├─ HTTP POST /compile   ──► Lenovo FastAPI
    ├─ HTTP POST /run       ──► Lenovo FastAPI
    ├─ if correct: HTTP POST /benchmark ──► Lenovo FastAPI (async)
    │              GET /jobs/{id} every 5s until done
    └─ derive final_outcome, write EvalRecord
```

## FastAPI endpoint contract (Lenovo)

All endpoints accept and return JSON. Triton/PyTorch source is sent as a string field. Full request/response shapes are defined in `packages/shared/src/shared/schema/verification_api.json` and enforced as Pydantic models in `packages/shared/src/shared/verification/api.py`.

| Endpoint | Mode | Returns |
|---|---|---|
| `POST /preflight` | Synchronous | `preflight_response` |
| `POST /compile` | Synchronous | `compile_response` |
| `POST /run` | Synchronous | `run_response` |
| `POST /benchmark` | Async | `job_accepted` (HTTP 202) |
| `GET /jobs/{job_id}` | Synchronous | `job_status` |

## Async benchmark pattern

`/benchmark` is the only async endpoint. It runs 10 warmup + 100 timed iterations across three implementations (Triton, eager, Inductor) — execution time is hardware-dependent and can't be bounded by a fixed timeout.

Flow:
1. Orchestrator POSTs to `/benchmark` → receives `{"job_id": "<uuid>"}` immediately (HTTP 202).
2. Lenovo executes the benchmark in a background thread.
3. Orchestrator polls `GET /jobs/{job_id}` every 5 seconds.
4. `job_status.status` transitions: `pending` → `running` → `done` (or `error`).
5. On `done`, `job_status.result` contains the full `benchmark_response`.
6. On `error`, `job_status.error_message` describes what failed.

The polling loop lives inside `VerificationClient.benchmark()` in `packages/orchestrator/src/orchestrator/clients/verification_client.py`. Eval code calls `benchmark()` as if it were synchronous — the async machinery is fully hidden. Poll timeout is controlled by the `VERIFICATION_POLL_TIMEOUT_S` env var (default 600s).

## Communication

- Transport: plain HTTP (LAN). Add a shared API key header (`X-API-Key`) for basic auth.
- The MacBook always initiates; the Lenovo never calls back.
- The Lenovo server runs with `CUDA_VISIBLE_DEVICES=0` to pin to the dGPU.
