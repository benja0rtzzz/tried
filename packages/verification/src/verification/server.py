"""
Verification FastAPI server — runs on the Lenovo LOQ (CUDA required).
See docs/architecture.md for endpoint contract.

Required env var: VERIFICATION_API_KEY
Launch: CUDA_VISIBLE_DEVICES=0 uv run uvicorn verification.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import Pool as MpPool
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import triton
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from shared.enums import CompileStatus
from shared.logging import get_logger
from shared.verification.api import (
    BenchmarkRequest,
    BenchmarkResponse,
    CompileRequest,
    CompileResponse,
    JobAccepted,
    JobStatus,
    JobStatusValue,
    PreflightRequest,
    PreflightResponse,
    RunRequest,
    RunResponse,
)

from verification.harness import (
    compile_check,
    preflight,
    run_benchmark,
    run_verification,
)

_log = get_logger(__name__)

app = FastAPI(title="tried-verification")

_API_KEY: str = os.environ.get("VERIFICATION_API_KEY", "")
if not _API_KEY:
    raise RuntimeError("VERIFICATION_API_KEY environment variable is not set")

# Server-side cap on how long a single GPU task may run in the worker subprocess.
# Must be LESS than the orchestrator's _INDUCTOR_TIMEOUT (300 s) so the server kills
# a stuck subprocess and returns an error response before the HTTP connection drops.
_WORKER_TIMEOUT = 240.0
_COMPILE_WORKER_TIMEOUT = 25.0


class _HealthResponse(BaseModel):
    cuda_available: bool
    device_name: Optional[str] = None
    device_count: int = 0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_total_mb: float = 0.0
    torch_version: str = ""
    triton_version: str = ""


# Single-worker pool serialises GPU work; benchmark runs are long and must not overlap.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Persistent subprocess for CUDA isolation. Spawned once; respawned automatically
# after a CUDA crash. spawn context required — CUDA contexts are not fork-safe.
_mp_ctx      = mp.get_context("spawn")
_worker:     MpPool | None = None
_worker_lock = threading.Lock()


def _get_worker() -> MpPool:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _mp_ctx.Pool(1)
    return _worker


def _kill_worker() -> None:
    """Terminate the current GPU worker process. The next _get_worker() spawns a fresh one."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.terminate()
            _worker.join()
            _worker = None


def _isolated(fn, *, worker_timeout: float = _WORKER_TIMEOUT, **kwargs):
    """Call fn(**kwargs) in the persistent worker subprocess.

    The worker is ALWAYS terminated after each call (in the finally block) so every
    GPU operation starts with a fresh CUDA context. This prevents a cudaErrorIllegalAddress
    from a bad Triton kernel from corrupting subsequent calls.

    The ~2 s spawn overhead per call is acceptable: preflight/run calls take 30-300 s
    and the spawn cost is <1% of that.

    Raises TimeoutError on timeout; re-raises any other exception from the worker as-is.
    No retry on ProcessError — the kernel is suspect, and the next call gets a fresh worker.
    """
    try:
        return _get_worker().apply_async(fn, kwds=kwargs).get(timeout=worker_timeout)
    except mp.TimeoutError:
        _log.error("worker timed out after %.0fs", worker_timeout)
        raise TimeoutError(f"GPU task timed out after {worker_timeout:.0f}s")
    except mp.ProcessError as exc:
        _log.warning("worker process died unexpectedly: %s", exc)
        raise RuntimeError(f"worker process died: {exc}") from exc
    finally:
        _kill_worker()


@app.middleware("http")
async def _auth(request: Request, call_next):
    if request.headers.get("X-API-Key", "") != _API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )
    return await call_next(request)


@app.get("/health")
async def health_endpoint():
    cuda = torch.cuda.is_available()
    if cuda:
        props = torch.cuda.get_device_properties(0)
        return _HealthResponse(
            cuda_available=True,
            device_name=torch.cuda.get_device_name(0),
            device_count=torch.cuda.device_count(),
            memory_allocated_mb=round(torch.cuda.memory_allocated(0) / 1024**2, 2),
            memory_reserved_mb=round(torch.cuda.memory_reserved(0) / 1024**2, 2),
            memory_total_mb=round(props.total_memory / 1024**2, 2),
            torch_version=torch.__version__,
            triton_version=triton.__version__,
        )
    return _HealthResponse(
        cuda_available=False,
        torch_version=torch.__version__,
        triton_version=triton.__version__,
    )


@app.post("/preflight")
def preflight_endpoint(req: PreflightRequest):
    _log.info(
        "preflight  shapes=%s  policy=%s",
        req.input_shapes,
        req.tolerance_policy.value,
    )
    try:
        return _isolated(
            preflight,
            pytorch_code=req.pytorch_code,
            input_shapes=req.input_shapes,
            input_dtypes=req.input_dtypes,
            rng_seed=req.rng_seed,
            tolerance_policy=req.tolerance_policy,
        )
    except (TimeoutError, RuntimeError) as e:
        return PreflightResponse(passed=False, error_message=str(e))


@app.post("/compile")
def compile_endpoint(req: CompileRequest):
    _log.info("compile")
    try:
        return _isolated(
            compile_check,
            worker_timeout=_COMPILE_WORKER_TIMEOUT,
            triton_code=req.triton_code,
        )
    except (TimeoutError, RuntimeError) as e:
        return CompileResponse(
            status=CompileStatus.FAILED,
            error_message=str(e),
        )


@app.post("/run")
def run_endpoint(req: RunRequest):
    _log.info(
        "run  shapes=%s  policy=%s",
        req.input_shapes,
        req.tolerance_policy.value,
    )
    try:
        return _isolated(
            run_verification,
            triton_code=req.triton_code,
            pytorch_code=req.pytorch_code,
            input_shapes=req.input_shapes,
            input_dtypes=req.input_dtypes,
            rng_seed=req.rng_seed,
            tolerance_policy=req.tolerance_policy,
        )
    except (TimeoutError, RuntimeError) as e:
        return RunResponse(
            error_message=str(e),
        )


@app.post("/benchmark", status_code=202)
async def benchmark_endpoint(req: BenchmarkRequest):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": JobStatusValue.PENDING,
            "result": None,
            "error": None,
        }

    def _run() -> None:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatusValue.RUNNING
        _log.info("benchmark job %s running", job_id)
        try:
            result: BenchmarkResponse = _isolated(
                run_benchmark,
                triton_code=req.triton_code,
                pytorch_code=req.pytorch_code,
                input_shapes=req.input_shapes,
                input_dtypes=req.input_dtypes,
                rng_seed=req.rng_seed,
            )
            with _jobs_lock:
                _jobs[job_id]["status"] = JobStatusValue.DONE
                _jobs[job_id]["result"] = result
            _log.info("benchmark job %s done", job_id)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            _log.error("benchmark job %s failed: %s", job_id, msg)
            with _jobs_lock:
                _jobs[job_id]["status"] = JobStatusValue.ERROR
                _jobs[job_id]["error"] = msg

    _executor.submit(_run)
    _log.info("benchmark job %s accepted", job_id)
    return JobAccepted(job_id=job_id)


@app.get("/jobs/{job_id}")
async def job_status_endpoint(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(
        status=job["status"],
        result=job["result"],
        error_message=job["error"],
    )


@app.on_event("shutdown")
async def _shutdown_worker():
    _kill_worker()
