"""
Mock of the Lenovo verification FastAPI server for local pipeline tests.

Call POST /admin/configure with a script before each test run.
The actual triton/pytorch code sent in requests is ignored — only the
scripted responses matter.

Script format — a list of attempt entries, one per expected attempt:
  [
    {
      "compile": {"status": "success"|"failed", "error_message": null|"..."},
      "run": {                           # omit for compile-failure attempts
        "correctness_status": "passed"|"failed",
        "tolerance_policy_used": "default_fp32",
        "vs_eager":    {5-stat block},
        "vs_inductor": {5-stat block},
        "error_message": null
      },
      "benchmark": {                     # omit unless run passes
        "triton_ms": ..., "eager_ms": ..., "inductor_ms": ...,
        "speedup_vs_eager": ..., "speedup_vs_inductor": ...,
        "triton_std_ms": ..., "eager_std_ms": ..., "inductor_std_ms": ...
      }
    },
    ...
  ]

State machine:
  _idx always points to the current attempt's script entry.
  After compile FAILED  → advance _idx (no run will follow).
  After run FAILED      → advance _idx (no benchmark will follow).
  After /benchmark      → advance _idx (attempt complete).
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request

app = FastAPI()

_scripts: list[dict[str, Any]] = []
_idx: int = 0
_jobs: dict[str, dict[str, Any]] = {}


@app.post("/admin/configure")
async def configure(request: Request) -> dict[str, Any]:
    global _scripts, _idx, _jobs
    body = await request.json()
    _scripts = body["scripts"]
    _idx = 0
    _jobs = {}
    return {"ok": True, "n_scripts": len(_scripts)}


@app.post("/preflight")
async def preflight(_: Request) -> dict[str, Any]:
    return {"passed": True, "vs_eager_inductor": None, "error_message": None}


@app.post("/compile")
async def compile_endpoint(_: Request) -> dict[str, Any]:
    global _idx
    entry = _scripts[_idx]
    resp = entry["compile"]
    if resp["status"] == "failed":
        _idx += 1
    return resp


@app.post("/run")
async def run_endpoint(_: Request) -> dict[str, Any]:
    global _idx
    entry = _scripts[_idx]
    resp = entry.get("run", {"error_message": "mock: no run script for this attempt"})
    if resp.get("correctness_status") != "passed":
        _idx += 1
    return resp


@app.post("/benchmark")
async def benchmark_endpoint(_: Request) -> dict[str, Any]:
    global _idx
    entry = _scripts[_idx]
    job_id = str(uuid.uuid4())
    _jobs[job_id] = entry.get("benchmark", {})
    _idx += 1
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        return {"status": "error", "result": None, "error_message": f"unknown job {job_id}"}
    return {"status": "done", "result": _jobs[job_id], "error_message": None}
