"""
Entry point for the TRIED orchestrator dataset-generation pipeline.

Usage (from the project root):
    TRIED_ROLE=orchestrator uv run python -m orchestrator.dataset.main

Required env vars: see packages/orchestrator/.env
Optional env vars:
    TRIED_DATASET_DIR — path to the dataset output directory (overrides TRIED_DATA_DIR)
    TRIED_DATA_DIR    — legacy dataset output override; ignored when set to "data"
    TRIED_CORPUS_PATH — path to the preflight-safe corpus JSONL
                        (default: data/preflight_safe.jsonl)
                        Run orchestrator.dataset.preflight_driver first to
                        produce this file.

Resume behaviour: on startup, already-completed dataset_ids are loaded from
dataset.jsonl and filtered out so no exact dataset task is processed twice. Preflight is
handled upstream by the preflight_driver — this pipeline assumes every row in
TRIED_CORPUS_PATH already passed the eager-vs-Inductor sanity check.
Hitting a Codex CLI rate limit stops the run cleanly with exit code 0; restart
the process once the rate-limit window clears.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from shared.dataset import load_dataset
from shared.logging import get_logger
from shared.models import CorpusRecord, PreflightSafeRecord

from orchestrator.dataset.agent import run_job
from orchestrator.clients.judge_client import RateLimitError
from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)


_REQUIRED_ENV = [
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]


def _load_completed_ids(dataset_path: Path) -> set[str]:
    return {row.dataset_id for row in load_dataset(dataset_path)}


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
    duplicate_rows = 0
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
            if record.dataset_id in seen:
                message = (
                    f"{corpus_path}:{i} — duplicate dataset_id={record.dataset_id} "
                    f"(first seen on line {seen[record.dataset_id]})"
                )
                _log.warning("duplicate corpus row  %s — skipping", message)
                _append_error(
                    errors_path,
                    record=record,
                    corpus_path=corpus_path,
                    line_no=i,
                    error_type="DuplicateDatasetId",
                    message=message,
                )
                duplicate_rows += 1
                continue
            seen[record.dataset_id] = i
            records.append(record)
    if invalid_rows or duplicate_rows:
        _log.warning(
            "corpus load skipped %d invalid row(s) and %d duplicate row(s); details in %s",
            invalid_rows,
            duplicate_rows,
            errors_path,
        )
    return records


def main() -> None:
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
            "corpus not found: %s — run orchestrator.dataset.preflight_driver first",
            corpus_path,
        )
        sys.exit(1)

    _log.info("loading corpus from %s", corpus_path)
    _log.info("dataset output: %s", dataset_path)
    _log.info("error output:   %s", errors_path)
    records = _load_corpus(corpus_path, errors_path)
    _log.info("%d preflight-safe train record(s) loaded", len(records))

    # --- Resume filter (completed exact dataset tasks only — preflight is pre-done) ---
    already_done = _load_completed_ids(dataset_path)
    if already_done:
        _log.info(
            "resuming: %d dataset task(s) already completed — skipping", len(already_done)
        )
    records = [r for r in records if r.dataset_id not in already_done]
    _log.info("%d record(s) remaining", len(records))

    client = make_client()

    completed = 0
    transport_errors = 0

    for i, record in enumerate(records):
        _log.info(
            "--- record %d/%d  dataset_id=%s  source_id=%s ---",
            i + 1,
            len(records),
            record.dataset_id,
            record.example_id,
        )
        try:
            run_job(record, client, output_dir)
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

    # --- Summary ---
    _log.info("=== run complete ===")
    _log.info("total records:      %d", len(records))
    _log.info("completed:          %d", completed)
    _log.info("transport errors:   %d", transport_errors)


if __name__ == "__main__":
    main()
