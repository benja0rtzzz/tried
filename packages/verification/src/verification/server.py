"""
Verification FastAPI server — runs on the Lenovo LOQ (CUDA required).
See docs/architecture.md for endpoint contract.
"""
from fastapi import FastAPI

app = FastAPI(title="tried-verification")


@app.post("/compile")
async def compile_kernel():
    raise NotImplementedError


@app.post("/run")
async def run_kernel():
    raise NotImplementedError


@app.post("/benchmark")
async def benchmark_kernel():
    raise NotImplementedError
