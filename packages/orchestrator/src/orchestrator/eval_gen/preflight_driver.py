"""Stage 4 — Preflight integration.

Consumes data/eval_gen/with_code.jsonl (Block C output: spec + accepted
pytorch_code) and runs each candidate through the Lenovo verification
server's /preflight endpoint. Accepted rows become EvalCorpusRecord
entries in eval/holdout/synthetic_fusions.jsonl; rejections continue to
data/eval_gen/rejected.jsonl with stage='preflight'.

Resumable: on restart, example_ids already present in either output
file are skipped.

Why /preflight specifically:
  - It already runs eager + torch.compile(inductor) on the candidate
    PyTorch code with the same inputs and compares them at the spec's
    tolerance — exactly what we want eval rows to satisfy.
  - We extended the response with eager_first_call_ms /
    inductor_first_call_ms (CUDA-event timed) so EvalCorpusRecord's
    acceptance block can record cold compile cost without a second
    round-trip.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from orchestrator.clients.verification_client import make_client
from shared.eval.forms import FORMS
from shared.logging import get_logger
from shared.models import (
    AcceptanceMeta,
    EvalCorpusRecord,
    EvalSpec,
)
from shared.verification.api import PreflightRequest, PreflightResponse

logger = get_logger(__name__)

DEFAULT_WITH_CODE = Path("data/eval_gen/with_code.jsonl")
DEFAULT_OUT = Path("eval/holdout/synthetic_fusions.jsonl")
DEFAULT_REJECTED = Path("data/eval_gen/rejected.jsonl")

# Stable namespace for example_id UUIDv5. Distinct from spec_id namespace
# so a spec with two different accepted pytorch_codes (shouldn't happen
# in practice — Codex is run once per spec — but guards against it)
# would yield distinct example_ids.
_EXAMPLE_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000eab")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _example_id_for(pytorch_code: str) -> str:
    """UUIDv5 derived from the SHA-256 of pytorch_code. Stable across
    re-runs; used as the join key for paired tests in stats/eval."""
    h = hashlib.sha256(pytorch_code.encode()).hexdigest()
    return str(uuid.uuid5(_EXAMPLE_NAMESPACE, h))


def _load_with_code(path: Path) -> list[tuple[EvalSpec, str]]:
    out: list[tuple[EvalSpec, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append((EvalSpec.model_validate(row["spec"]), row["pytorch_code"]))
    return out


def _collect_seen_example_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "example_id" in row:
                    seen.add(row["example_id"])
                elif "spec_id" in row:
                    # rejected.jsonl rows from Block C use spec_id, not
                    # example_id. Block D writes with example_id; for
                    # resume purposes, also remember spec_ids of Block-C
                    # rejections so we don't reprocess specs that never
                    # produced code.
                    seen.add(row["spec_id"])
    return seen


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line)
        f.flush()


# ---------------------------------------------------------------------------
# Server-version probe (one-shot at startup)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerVersions:
    torch_version: str
    triton_version: str


def _fetch_server_versions(base_url: str, api_key: str) -> ServerVersions:
    response = httpx.get(
        f"{base_url.rstrip('/')}/health",
        headers={"X-API-Key": api_key},
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    return ServerVersions(
        torch_version=str(payload.get("torch_version", "")),
        triton_version=str(payload.get("triton_version", "")),
    )


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PreflightRejection:
    spec_id: str
    example_id: str
    reason: str

    def to_json_line(self) -> str:
        return json.dumps({
            "spec_id":   self.spec_id,
            "example_id": self.example_id,
            "stage":     "preflight",
            "reason":    self.reason,
        }) + "\n"


def _build_request(spec: EvalSpec, pytorch_code: str) -> PreflightRequest:
    return PreflightRequest(
        pytorch_code=pytorch_code,
        input_shapes=spec.input_shapes,
        input_dtypes=spec.input_dtypes,
        rng_seed=spec.rng_seed,
        tolerance_policy=spec.tolerance_policy,
    )


def _build_record(
    spec: EvalSpec,
    pytorch_code: str,
    response: PreflightResponse,
    server_versions: ServerVersions,
    example_id: str,
) -> EvalCorpusRecord:
    """Assumes response.passed and timings non-null. Caller checks."""
    assert response.eager_first_call_ms is not None
    assert response.inductor_first_call_ms is not None
    return EvalCorpusRecord(
        example_id=example_id,
        spec=spec,
        pytorch_code=pytorch_code,
        op_category=FORMS[spec.form].op_category,
        difficulty=spec.tier,
        acceptance=AcceptanceMeta(
            accepted_at=dt.datetime.now(dt.timezone.utc),
            torch_version=server_versions.torch_version,
            triton_version=server_versions.triton_version,
            preflight_eager_first_call_ms=response.eager_first_call_ms,
            preflight_inductor_first_call_ms=response.inductor_first_call_ms,
        ),
    )


def _classify(
    response: PreflightResponse,
) -> tuple[bool, str]:
    """Returns (accept, reason). reason is the rejection text when not accepted."""
    if response.error_message is not None:
        return False, f"harness error: {response.error_message}"
    if not response.passed:
        if response.vs_eager_inductor is not None:
            stats = response.vs_eager_inductor
            return False, (
                f"eager-vs-Inductor disagreement: max_abs={stats.max_abs_diff:.3e} "
                f"max_rel={stats.max_rel_diff:.3e} "
                f"pct_exceeding={stats.pct_elements_exceeding_tol:.2f}"
            )
        return False, "preflight failed without diagnostic stats"
    if response.eager_first_call_ms is None or response.inductor_first_call_ms is None:
        return False, "preflight passed but timings missing — server bug"
    return True, ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    with_code_path: Path,
    out_path: Path,
    rejected_path: Path,
    limit: int | None,
) -> None:
    rows = _load_with_code(with_code_path)
    logger.info(f"loaded {len(rows)} (spec, code) rows from {with_code_path}")

    if limit is not None:
        rows = rows[:limit]
        logger.info(f"limit={limit} applied")

    # Resume: drop any (spec, code) whose example_id (or spec_id, for
    # Block-C rejections) is already in either output file.
    already = _collect_seen_example_ids(out_path, rejected_path)
    if already:
        before = len(rows)
        rows = [
            (s, c) for s, c in rows
            if _example_id_for(c) not in already and s.spec_id not in already
        ]
        logger.info(f"resume: skipping {before - len(rows)} already-processed rows")

    client = make_client()
    base_url = os.environ["VERIFICATION_SERVER_URL"]
    api_key = os.environ["VERIFICATION_API_KEY"]
    versions = _fetch_server_versions(base_url, api_key)
    logger.info(
        f"server versions: torch={versions.torch_version} triton={versions.triton_version}"
    )

    n_accepted = n_rejected = 0
    for i, (spec, code) in enumerate(rows):
        example_id = _example_id_for(code)
        try:
            response = client.preflight(_build_request(spec, code))
        except Exception as e:
            rej = _PreflightRejection(
                spec_id=spec.spec_id, example_id=example_id,
                reason=f"transport error: {type(e).__name__}: {e}",
            )
            _append_jsonl(rejected_path, rej.to_json_line())
            n_rejected += 1
            logger.error(f"transport error on {spec.spec_id[:8]}: {e}")
            continue

        accept, reason = _classify(response)
        if accept:
            record = _build_record(spec, code, response, versions, example_id)
            _append_jsonl(out_path, record.model_dump_json() + "\n")
            n_accepted += 1
        else:
            rej = _PreflightRejection(
                spec_id=spec.spec_id, example_id=example_id, reason=reason,
            )
            _append_jsonl(rejected_path, rej.to_json_line())
            n_rejected += 1
            logger.info(
                f"reject [{i+1}/{len(rows)}] {spec.spec_id[:8]} {reason}"
            )

        if (i + 1) % 10 == 0:
            logger.info(
                f"progress: {i+1}/{len(rows)} accepted={n_accepted} rejected={n_rejected}"
            )

    logger.info(
        f"done: accepted={n_accepted} rejected={n_rejected} total={len(rows)}"
    )


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="orchestrator.eval_gen.preflight_driver",
        description="Run /preflight on Block-C output; produce EvalCorpusRecord rows.",
    )
    parser.add_argument("--with-code", type=Path, default=DEFAULT_WITH_CODE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.with_code, args.out, args.rejected, args.limit)


if __name__ == "__main__":
    main()
