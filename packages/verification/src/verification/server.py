"""
Verification FastAPI server — runs on the Lenovo LOQ (CUDA required).
See docs/architecture.md for endpoint contract.

Required env var: VERIFICATION_API_KEY
Launch: CUDA_VISIBLE_DEVICES=0 uv run uvicorn verification.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from shared.logging import get_logger
from shared.verification.api import (
    BenchmarkRequest,
    BenchmarkResponse,
    CompileRequest,
    JobAccepted,
    JobStatus,
    JobStatusValue,
    PreflightRequest,
    RunRequest,
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

# Single-worker pool serialises GPU work; benchmark runs are long and must not overlap.
_executor    = ThreadPoolExecutor(max_workers=1)
_jobs:       dict[str, dict[str, Any]] = {}
_jobs_lock   = threading.Lock()


@app.middleware("http")
async def _auth(request: Request, call_next):
    if _API_KEY:
        if request.headers.get("X-API-Key", "") != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)


@app.post("/preflight")
async def preflight_endpoint(req: PreflightRequest):
    _log.info(
        "preflight  shapes=%s  policy=%s",
        req.input_shapes, req.tolerance_policy.value,
    )
    return preflight(
        pytorch_code=req.pytorch_code,
        input_shapes=req.input_shapes,
        input_dtypes=req.input_dtypes,
        rng_seed=req.rng_seed,
        tolerance_policy=req.tolerance_policy,
    )


@app.post("/compile")
async def compile_endpoint(req: CompileRequest):
    _log.info("compile")
    return compile_check(req.triton_code)


@app.post("/run")
async def run_endpoint(req: RunRequest):
    _log.info(
        "run  shapes=%s  policy=%s",
        req.input_shapes, req.tolerance_policy.value,
    )
    return run_verification(
        triton_code=req.triton_code,
        pytorch_code=req.pytorch_code,
        input_shapes=req.input_shapes,
        input_dtypes=req.input_dtypes,
        rng_seed=req.rng_seed,
        tolerance_policy=req.tolerance_policy,
    )


@app.post("/benchmark", status_code=202)
async def benchmark_endpoint(req: BenchmarkRequest):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": JobStatusValue.PENDING, "result": None, "error": None}

    def _run() -> None:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatusValue.RUNNING
        _log.info("benchmark job %s running", job_id)
        try:
            result: BenchmarkResponse = run_benchmark(
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
                _jobs[job_id]["error"]  = msg

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
