"""
Entry point for the TRIED orchestrator dataset-generation pipeline.

Usage (from the project root):
    TRIED_ROLE=orchestrator uv run python -m orchestrator.train.dataset.main
    TRIED_ROLE=orchestrator uv run python -m orchestrator.train.dataset.main --allowed-ops embedding

Required env vars: see packages/orchestrator/.env
Optional env vars:
    TRIED_DATASET_DIR      — dataset output directory (overrides TRIED_DATA_DIR)
    TRIED_DATA_DIR         — legacy dataset output override; ignored when "data"
    TRIED_CORPUS_PATH      — path to the preflight-safe corpus JSONL
                             (default: data/preflight_safe.jsonl)
                             Run orchestrator.train.dataset.preflight_driver first to
                             produce this file.
    TRIED_MAX_PER_CATEGORY — max total unique source example_ids per op
                             category (default: 180). Rows already in
                             dataset.jsonl count toward this cap.

Run targeting: edit the ALLOWED_OPS constant below to restrict which op
categories a run loads (None = all), or pass --allowed-ops to override it
for a run.

Corpus/resume behaviour: the corpus is deduped to one row per source
example_id — shape/dtype/seed variants of the same source never enter. On
startup, every source example_id already present in dataset.jsonl is skipped,
so no source is processed twice across runs. Each category is then filled up
to TRIED_MAX_PER_CATEGORY total unique example_ids. Preflight is handled
upstream by the preflight_driver — this pipeline assumes every row in
TRIED_CORPUS_PATH already passed the eager-vs-Inductor sanity check.
Hitting a Codex CLI rate limit stops the run cleanly with exit code 0; restart
the process once the rate-limit window clears.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from shared.dataset import load_dataset_index
from shared.enums import DatasetOutcome, OpCategory
from shared.logging import get_logger
from shared.models import CorpusRecord, PreflightSafeRecord

from orchestrator.train.dataset.agent import run_job
from orchestrator.clients.dataset.judge_client import RateLimitError
from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)


_REQUIRED_ENV = [
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]

# --- Run-targeting policy (edit ALLOWED_OPS to retarget a night's run) -------
#
# ALLOWED_OPS restricts which op categories this run will load. Set it to a
# list of shared.enums.OpCategory *values* (validated at startup — a typo
# aborts the run rather than silently loading nothing). Set it to None to
# load every category.
#
#   ALLOWED_OPS = ["embedding"]
#   ALLOWED_OPS = None   # load all categories
#
# The corpus is deduped to one row per source example_id (variants — same
# source, different shapes/dtypes/seed — never enter). Each category is
# capped at TRIED_MAX_PER_CATEGORY total unique example_ids; rows already in
# dataset.jsonl count toward that cap, so a night fills the lagging
# categories rather than re-saturating the full ones.
ALLOWED_OPS: list[str] | None = ["embedding"]

_DEFAULT_MAX_PER_CATEGORY = 250


def _split_allowed_ops(raw_values: list[str]) -> list[str]:
    return [
        value.strip()
        for raw in raw_values
        for value in raw.split(",")
        if value.strip()
    ]


def _validated_allowed_ops(configured: list[str] | None) -> set[str] | None:
    """Resolve ALLOWED_OPS to a set of valid OpCategory values, or None."""
    if configured is None:
        return None
    valid = {c.value for c in OpCategory}
    invalid = [c for c in configured if c not in valid]
    if invalid:
        _log.error(
            "ALLOWED_OPS contains invalid op categories: %s — valid values: %s",
            ", ".join(invalid),
            ", ".join(sorted(valid)),
        )
        sys.exit(1)
    if not configured:
        _log.error("ALLOWED_OPS is an empty list — set it to None to load all categories")
        sys.exit(1)
    return set(configured)


def _append_error(
    errors_path: Path,
    *,
    record: CorpusRecord | None = None,
    dataset_id: str | None = None,
    example_id: str | None = None,
    corpus_path: Path | None = None,
    line_no: int | None = None,
    error_type: str,
    message: str,
) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "dataset_id": record.dataset_id if record is not None else dataset_id,
        "example_id": record.example_id if record is not None else example_id,
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if corpus_path is not None:
        entry["corpus_path"] = str(corpus_path)
    if line_no is not None:
        entry["line_no"] = line_no
    with errors_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _extract_row_ids(line: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    dataset_id = payload.get("dataset_id")
    example_id = payload.get("example_id")
    return (
        dataset_id if isinstance(dataset_id, str) else None,
        example_id if isinstance(example_id, str) else None,
    )


def _load_corpus(corpus_path: Path, errors_path: Path) -> list[CorpusRecord]:
    """Load preflight_safe.jsonl and rehydrate fixed training metadata in code."""
    records: list[CorpusRecord] = []
    seen: dict[str, int] = {}
    invalid_rows = 0
    variant_rows = 0
    with corpus_path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = PreflightSafeRecord.model_validate_json(line).to_corpus_record()
            except Exception as exc:
                dataset_id, example_id = _extract_row_ids(line)
                message = f"{corpus_path}:{i} — {exc}"
                _log.error("invalid corpus row  %s — skipping", message)
                _append_error(
                    errors_path,
                    dataset_id=dataset_id,
                    example_id=example_id,
                    corpus_path=corpus_path,
                    line_no=i,
                    error_type=type(exc).__name__,
                    message=message,
                )
                invalid_rows += 1
                continue
            if record.dataset_id is None:
                message = f"{corpus_path}:{i} — dataset_id was not hydrated"
                _log.error("invalid corpus row  %s — skipping", message)
                _append_error(
                    errors_path,
                    record=record,
                    corpus_path=corpus_path,
                    line_no=i,
                    error_type="MissingDatasetId",
                    message=message,
                )
                invalid_rows += 1
                continue
            if record.example_id in seen:
                # Expected, by-design: the corpus carries shape/dtype/seed
                # variants of the same source. Collapsing to one row per
                # example_id is policy, not an error — count it, do not write
                # to errors.jsonl (that file is for retryable failures only).
                variant_rows += 1
                continue
            seen[record.example_id] = i
            records.append(record)
    if invalid_rows:
        _log.warning(
            "corpus load skipped %d invalid row(s); details in %s",
            invalid_rows,
            errors_path,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.train.dataset.main")
    parser.add_argument(
        "--allowed-ops",
        nargs="+",
        default=None,
        metavar="OP_CATEGORY",
        help="only train rows from these op categories; accepts spaces or commas",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        _log.error("missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    dataset_dir_env = os.getenv("TRIED_DATASET_DIR")
    legacy_data_dir = os.getenv("TRIED_DATA_DIR")
    if dataset_dir_env:
        output_dir = Path(dataset_dir_env)
    elif legacy_data_dir and Path(legacy_data_dir) != Path("data"):
        output_dir = Path(legacy_data_dir)
    else:
        output_dir = Path("data/dataset")
    corpus_path = Path(os.getenv("TRIED_CORPUS_PATH", Path("data") / "preflight_safe.jsonl"))
    dataset_path = output_dir / "dataset.jsonl"
    errors_path = output_dir / "errors.jsonl"

    if not corpus_path.exists():
        _log.error(
            "corpus not found: %s — run orchestrator.train.dataset.preflight_driver first",
            corpus_path,
        )
        sys.exit(1)

    records = _load_corpus(corpus_path, errors_path)

    # --- Resume filter (skip any source example_id already in the dataset) ---
    dataset_index = load_dataset_index(dataset_path)
    already_done = dataset_index.completed_example_ids
    if dataset_index.duplicate_row_count:
        _log.warning(
            "dataset output contains %d duplicate completed row(s) across %d dataset_id(s)",
            dataset_index.duplicate_row_count,
            len(dataset_index.duplicate_id_counts),
        )
    records = [r for r in records if r.example_id not in already_done]

    # --- ALLOWED_OPS category filter ---
    configured_allowed_ops = (
        _split_allowed_ops(args.allowed_ops)
        if args.allowed_ops is not None
        else ALLOWED_OPS
    )
    allowed = _validated_allowed_ops(configured_allowed_ops)
    if allowed is not None:
        records = [r for r in records if r.op_category.value in allowed]

    # --- Per-category cap (total unique example_ids; already-collected count toward it) ---
    try:
        cap = int(os.getenv("TRIED_MAX_PER_CATEGORY", _DEFAULT_MAX_PER_CATEGORY))
    except ValueError:
        _log.error(
            "TRIED_MAX_PER_CATEGORY=%r is not an integer",
            os.getenv("TRIED_MAX_PER_CATEGORY"),
        )
        sys.exit(1)
    if cap <= 0:
        _log.error("TRIED_MAX_PER_CATEGORY must be a positive integer (got %d)", cap)
        sys.exit(1)

    existing_per_cat: Counter[str] = Counter()
    seen_eids: set[str] = set()
    for row in dataset_index.rows:
        if row.example_id in seen_eids:
            continue
        seen_eids.add(row.example_id)
        existing_per_cat[row.source.op_category.value] += 1

    capped: list[CorpusRecord] = []
    added_per_cat: Counter[str] = Counter()
    cap_skipped: Counter[str] = Counter()
    for r in records:
        cat = r.op_category.value
        if existing_per_cat[cat] + added_per_cat[cat] >= cap:
            cap_skipped[cat] += 1
            continue
        added_per_cat[cat] += 1
        capped.append(r)
    records = capped

    _log.info("ready: %d record(s) to process (corpus=%s, cap=%d)", len(records), corpus_path, cap)

    client = make_client()

    completed = 0
    transport_errors = 0
    passed = 0
    failed = 0

    for i, record in enumerate(records):
        try:
            outcome = run_job(record, client, output_dir)
        except RateLimitError as exc:
            _log.error("Codex CLI rate limit hit: %s", exc)
            _log.error(
                "stopping cleanly — %d example(s) completed this run; "
                "restart once the rate-limit window clears",
                completed,
            )
            sys.exit(0)
        except ValidationError as exc:
            _log.error(
                "data validation error  dataset_id=%s  source_id=%s  %s: %s — skipping without recording; "
                "fix the underlying bug then restart to retry this example",
                record.dataset_id,
                record.example_id,
                type(exc).__name__,
                exc,
            )
            _append_error(
                errors_path,
                record=record,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            transport_errors += 1
            continue
        except Exception as exc:
            _log.error(
                "transport error  dataset_id=%s  source_id=%s  %s: %s — will retry on restart",
                record.dataset_id,
                record.example_id,
                type(exc).__name__,
                exc,
            )
            _append_error(
                errors_path,
                record=record,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            transport_errors += 1
            continue

        completed += 1
        if outcome == DatasetOutcome.COMPILED_CORRECT:
            passed += 1
        else:
            failed += 1
        _log.info(
            "progress: %d/%d passed=%d failed=%d outcome=%s",
            i + 1, len(records), passed, failed, outcome.value,
        )

    # --- Summary ---
    _log.info("=== run complete ===")
    _log.info("total records:      %d", len(records))
    _log.info("completed:          %d", completed)
    _log.info("transport errors:   %d", transport_errors)


if __name__ == "__main__":
    main()
