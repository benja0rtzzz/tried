# Architecture

## Two-machine setup

| Machine | Role | Key software |
|---|---|---|
| M4 MacBook Pro (48 GB) | Orchestrator | Ollama, Codex CLI, MLX, UV, httpx |
| Lenovo LOQ RTX 4060 8 GB | Verification server | Triton, PyTorch+CUDA, FastAPI, uvicorn |

## Dataset Agent Loop

The training dataset pipeline consumes preflight-safe synthetic skeleton rows from `data/preflight_safe.jsonl`. The eager-vs-Inductor preflight is done upstream by `orchestrator.train.dataset.preflight_driver`; rows reaching `orchestrator.train.dataset.main` are assumed clean. The `tolerance_policy` selected before preflight is carried in the row and reused by the dataset loop; it is not recomputed there.

```
PyTorch input
    │
    ▼
Orchestrator (MacBook)
    ├─ calls Ollama → local LLM generates Triton candidate
    ├─ HTTP POST /compile   ──► Lenovo FastAPI  (static Triton validation)
    ├─ HTTP POST /run       ──► Lenovo FastAPI  (synchronous)
    ├─ if verification failed: call Codex CLI judge (`gpt-5-3-codex`) → labels/advice
    └─ retry (max N=3), record every attempt
```

Dataset rows record `/compile`, `/run` errors, compact correctness outcome, deterministic failure symptoms/patches, and judge classification/root-cause/repair-action/advice for failed attempts. A dataset job is `compiled_correct` only when `/run` reports `correctness_status=passed` against both eager and Inductor under the row's tolerance policy. They do **not** record benchmark timings, speedups, or detailed correctness stats; those are eval-only.

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

`/preflight`, `/compile`, and `/run` execute inside a worker subprocess that is terminated after each request. `/compile` is static validation: it imports the candidate, requires a `@triton.jit` kernel, and finds the callable wrapper. Shape-aware launch compilation happens in `/run`, so launch-time JIT failures are runtime failures. This keeps bad Triton imports, launch-time JIT failures, CUDA illegal-address failures, and timeouts from corrupting the next attempt's CUDA context or the FastAPI server process.

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
