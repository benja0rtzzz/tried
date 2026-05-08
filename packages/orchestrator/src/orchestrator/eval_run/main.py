"""Eval-run entry point.

Reads eval/holdout/synthetic_fusions.jsonl (the locked eval set) and
runs a single raw generator attempt on each row, writing EvalRecord
rows to eval/results/<model_label>/eval_rows.jsonl. No judge calls;
classification is derived from compile/correctness/benchmark.

Usage:
    TRIED_ROLE=orchestrator uv run python -m orchestrator.eval_run.main \\
        --model-label qwen2.5-coder:14b-vanilla

Required env vars:
    VERIFICATION_SERVER_URL
    VERIFICATION_API_KEY

Resume: on restart, example_ids already in eval_rows.jsonl are skipped.
The run_id is read from the first existing row (so all rows in one
output file share one run_id) or freshly generated if the file is empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from shared.logging import get_logger
from shared.models import EvalCorpusRecord

from orchestrator.clients.verification_client import make_client
from orchestrator.eval_run.agent import run_eval_job

_log = get_logger(__name__)

DEFAULT_HOLDOUT = Path("eval/holdout/synthetic_fusions.jsonl")
DEFAULT_RESULTS_ROOT = Path("eval/results")

_REQUIRED_ENV = [
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]


def _check_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        _log.error(f"missing env vars: {', '.join(missing)}")
        sys.exit(1)


def _load_holdout(path: Path) -> list[EvalCorpusRecord]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(EvalCorpusRecord.model_validate_json(line))
    return out


def _read_existing(out_path: Path) -> tuple[set[str], str | None]:
    """Read out_path (if it exists). Returns (already-processed example_ids,
    existing run_id). run_id is the value found on the first row; if the
    file is empty or missing, returns None and the caller generates a fresh
    run_id."""
    if not out_path.exists():
        return set(), None
    seen: set[str] = set()
    run_id: str | None = None
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seen.add(row["example_id"])
            if run_id is None:
                run_id = row.get("run_id")
    return seen, run_id


def _append_record(out_path: Path, record_json: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(record_json)
        f.write("\n")
        f.flush()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="orchestrator.eval_run.main",
        description="Run the agent loop on the locked eval set; produce EvalRecord rows.",
    )
    parser.add_argument(
        "--model-label", required=True,
        help="generator condition tag (e.g. 'qwen2.5-coder:14b-vanilla'). "
             "Drives the output directory under eval/results/.",
    )
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N rows (after resume filtering)",
    )
    args = parser.parse_args()

    _check_env()

    out_dir = args.results_root / args.model_label
    out_path = out_dir / "eval_rows.jsonl"

    rows = _load_holdout(args.holdout)
    _log.info(f"loaded {len(rows)} eval rows from {args.holdout}")

    seen, existing_run_id = _read_existing(out_path)
    run_id = existing_run_id or str(uuid.uuid4())
    if seen:
        before = len(rows)
        rows = [r for r in rows if r.example_id not in seen]
        _log.info(f"resume: skipping {before - len(rows)} already-processed rows")
        _log.info(f"continuing run_id={run_id}")
    else:
        _log.info(f"fresh run_id={run_id}")

    if args.limit is not None:
        rows = rows[:args.limit]
        _log.info(f"limit={args.limit} applied")

    client = make_client()
    n_done = n_skipped = 0
    try:
        for i, record in enumerate(rows):
            try:
                eval_record = run_eval_job(record, client, args.model_label, run_id)
            except Exception as e:
                _log.error(
                    f"transport / unexpected error on {record.example_id[:8]}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            if eval_record is None:
                n_skipped += 1
                continue

            _append_record(out_path, eval_record.model_dump_json())
            n_done += 1
            if (i + 1) % 5 == 0:
                _log.info(f"progress: {i+1}/{len(rows)} done={n_done} skipped={n_skipped}")
    finally:
        _log.info(f"shutdown: done={n_done} skipped={n_skipped} run_id={run_id}")


if __name__ == "__main__":
    main()
