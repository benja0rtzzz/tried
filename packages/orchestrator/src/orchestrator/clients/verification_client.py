"""
HTTP client for the Lenovo verification server.
Wraps the four endpoints defined in shared.verification.api.
Errors outside the protocol (transport failures, non-2xx responses) are logged
and re-raised as native exceptions — the agent loop decides what to do with them.
"""
from __future__ import annotations

import os
import time
from typing import TypeVar, Type

import httpx
from pydantic import BaseModel

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

_log = get_logger(__name__)

_TIMEOUT        = 30.0   # for all synchronous endpoints
_POLL_INTERVAL  = 5.0    # seconds between job status checks
_POLL_TIMEOUT   = float(os.getenv("VERIFICATION_POLL_TIMEOUT_S", "600"))

_T = TypeVar("_T", bound=BaseModel)


class VerificationClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    def preflight(self, request: PreflightRequest) -> PreflightResponse:
        return self._post("/preflight", request, PreflightResponse)

    def compile(self, request: CompileRequest) -> CompileResponse:
        return self._post("/compile", request, CompileResponse)

    def run(self, request: RunRequest) -> RunResponse:
        return self._post("/run", request, RunResponse)

    def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        job: JobAccepted = self._post("/benchmark", request, JobAccepted)
        _log.info("Benchmark job %s accepted — polling every %.0fs", job.job_id, _POLL_INTERVAL)
        return self._poll(job.job_id)

    def _poll(self, job_id: str) -> BenchmarkResponse:
        url = f"{self._base_url}/jobs/{job_id}"
        deadline = time.monotonic() + _POLL_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL)
            try:
                response = httpx.get(url, headers=self._headers, timeout=_TIMEOUT)
                response.raise_for_status()
            except httpx.ConnectError:
                _log.error("Verification server unreachable while polling %s", url)
                raise
            except httpx.HTTPStatusError as e:
                _log.error("Poll returned HTTP %d for job %s", e.response.status_code, job_id)
                raise
            status = JobStatus.model_validate(response.json())
            if status.status == JobStatusValue.DONE:
                _log.info("Benchmark job %s done", job_id)
                return status.result  # type: ignore[return-value]
            if status.status == JobStatusValue.ERROR:
                _log.error("Benchmark job %s failed: %s", job_id, status.error_message)
                raise RuntimeError(f"Benchmark job {job_id} failed: {status.error_message}")
            _log.info("Benchmark job %s — %s", job_id, status.status.value)
        raise TimeoutError(f"Benchmark job {job_id} did not complete within {_POLL_TIMEOUT:.0f}s")

    def _post(self, path: str, request: BaseModel, response_model: Type[_T]) -> _T:
        url = self._base_url + path
        try:
            response = httpx.post(
                url,
                content=request.model_dump_json(),
                headers=self._headers,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return response_model.model_validate(response.json())
        except httpx.ConnectError:
            _log.error("Verification server unreachable at %s", url)
            raise
        except httpx.TimeoutException:
            _log.error("Request to %s timed out after %.0fs", url, _TIMEOUT)
            raise
        except httpx.HTTPStatusError as e:
            _log.error(
                "Verification server returned HTTP %d for %s",
                e.response.status_code, url,
            )
            raise


def make_client() -> VerificationClient:
    """Construct a VerificationClient from environment variables.

    Required env vars: VERIFICATION_SERVER_URL, VERIFICATION_API_KEY.
    Raises KeyError at startup if either is missing.
    """
    return VerificationClient(
        base_url=os.environ["VERIFICATION_SERVER_URL"],
        api_key=os.environ["VERIFICATION_API_KEY"],
    )
