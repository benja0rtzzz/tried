"""
Pre-screen training corpus examples through /preflight before the agent loop.

Run this once (resumable) before starting the agent loop. Only examples that
pass the eager-vs-Inductor sanity check are written to preflight_safe.jsonl;
the agent loop reads from there so no inline preflight is needed.

Reads : data/corpus_gen/with_code.jsonl  (corpus_gen Block output)
Writes:
  data/preflight_safe.jsonl     — CorpusRecord rows that passed
  data/preflight_rejected.jsonl — rejection details for genuine failures

Resumable: rows already present in either output file are skipped.

Usage (from project root):
  TRIED_ROLE=orchestrator uv run python -m orchestrator.dataset.preflight_driver
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from shared.enums import Dtype, OpCategory, Split, TolerancePolicy
from shared.logging import get_logger
from shared.models import CorpusRecord
from shared.verification.api import PreflightRequest

from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)

_REQUIRED_ENV = ["VERIFICATION_SERVER_URL", "VERIFICATION_API_KEY"]

DEFAULT_WITH_CODE = Path("data/corpus_gen/with_code.jsonl")
DEFAULT_SAFE     = Path("data/preflight_safe.jsonl")
DEFAULT_REJECTED = Path("data/preflight_rejected.jsonl")


# ---------------------------------------------------------------------------
# Policy selection (mirrors agent._select_policy)
# ---------------------------------------------------------------------------

def _select_policy(dtypes: list[Dtype], op_category: OpCategory) -> TolerancePolicy:
    fp16_dtypes = {Dtype.FLOAT16, Dtype.BFLOAT16}
    int_dtypes  = {Dtype.INT8, Dtype.INT16, Dtype.INT32, Dtype.INT64, Dtype.BOOL}
    dtype_set   = set(dtypes)

    if dtype_set <= int_dtypes:
        return TolerancePolicy.EXACT_INTEGER

    has_fp16 = bool(dtype_set & fp16_dtypes)

    if op_category == OpCategory.QUANTIZATION:
        return TolerancePolicy.LOW_PRECISION_DEQUANT
    if op_category == OpCategory.FUSED_ATTENTION and has_fp16:
        return TolerancePolicy.ATTENTION_SOFTMAX_FP16
    if op_category == OpCategory.REDUCTION:
        return TolerancePolicy.REDUCTION_FP16 if has_fp16 else TolerancePolicy.REDUCTION_FP32
    return TolerancePolicy.DEFAULT_FP16 if has_fp16 else TolerancePolicy.DEFAULT_FP32


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_corpus(path: Path) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                records.append(CorpusRecord.model_validate(row["candidate"]))
            except Exception as exc:
                raise ValueError(f"{path}:{i} — {exc}") from exc
    return records


def _collect_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line)
        f.flush()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    with_code_path: Path,
    safe_path: Path,
    rejected_path: Path,
    limit: int | None,
) -> None:
    all_records = _load_corpus(with_code_path)
    records = [r for r in all_records if r.split == Split.TRAIN]
    _log.info(
        "loaded %d train records (total corpus: %d)",
        len(records), len(all_records),
    )

    if limit is not None:
        records = records[:limit]
        _log.info("limit=%d applied", limit)

    already = _collect_seen(safe_path) | _collect_seen(rejected_path)
    if already:
        before = len(records)
        records = [r for r in records if r.example_id not in already]
        _log.info("resume: skipping %d already-processed", before - len(records))

    _log.info("%d records to preflight", len(records))

    client = make_client()
    n_passed = n_failed = n_errors = 0

    for i, record in enumerate(records, start=1):
        policy = _select_policy(record.input_dtypes, record.op_category)
        try:
            resp = client.preflight(PreflightRequest(
                pytorch_code=record.pytorch_code,
                input_shapes=record.input_shapes,
                input_dtypes=record.input_dtypes,
                rng_seed=record.rng_seed,
                tolerance_policy=policy,
            ))
        except Exception as exc:
            _log.error(
                "[%d/%d] transport error  example_id=%s  %s: %s — will retry on restart",
                i, len(records), record.example_id, type(exc).__name__, exc,
            )
            n_errors += 1
            continue

        if resp.passed:
            _append(safe_path, record.model_dump_json() + "\n")
            n_passed += 1
            _log.info("[%d/%d] pass    example_id=%s", i, len(records), record.example_id)
        else:
            reason = resp.error_message or "eager-vs-inductor disagreement"
            _append(rejected_path, json.dumps({
                "example_id": record.example_id,
                "reason": reason,
            }) + "\n")
            n_failed += 1
            _log.info(
                "[%d/%d] reject  example_id=%s  reason=%s",
                i, len(records), record.example_id, reason,
            )

        if i % 50 == 0:
            _log.info(
                "progress: %d/%d  passed=%d  rejected=%d  transport_errors=%d",
                i, len(records), n_passed, n_failed, n_errors,
            )

    _log.info("=== preflight complete ===")
    _log.info("passed:           %d  → %s", n_passed, safe_path)
    _log.info("rejected:         %d  → %s", n_failed, rejected_path)
    _log.info("transport errors: %d  (not recorded — will retry on restart)", n_errors)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        _log.error("missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    parser = argparse.ArgumentParser(
        prog="orchestrator.dataset.preflight_driver",
        description="Pre-screen training corpus through /preflight before the agent loop.",
    )
    parser.add_argument(
        "--with-code", type=Path, default=DEFAULT_WITH_CODE,
        help="Input: with_code.jsonl from corpus_gen",
    )
    parser.add_argument(
        "--safe", type=Path, default=DEFAULT_SAFE,
        help="Output: CorpusRecord rows that passed preflight",
    )
    parser.add_argument(
        "--rejected", type=Path, default=DEFAULT_REJECTED,
        help="Output: rejection details for genuine failures",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (for testing)",
    )
    args = parser.parse_args()
    run(args.with_code, args.safe, args.rejected, args.limit)


if __name__ == "__main__":
    main()
