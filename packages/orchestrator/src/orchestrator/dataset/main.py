"""
Entry point for the TRIED orchestrator dataset-generation pipeline.

Usage (from the project root):
    TRIED_ROLE=orchestrator uv run python -m orchestrator.dataset.main

Required env vars: see packages/orchestrator/.env
Optional env vars:
    TRIED_DATA_DIR    — path to the data directory (default: data/)
    TRIED_CORPUS_PATH — path to the preflight-safe corpus JSONL
                        (default: data/preflight_safe.jsonl)
                        Run orchestrator.dataset.preflight_driver first to
                        produce this file.

Resume behaviour: on startup, already-completed example_ids are loaded from
dataset.jsonl and filtered out so no example is processed twice. Preflight is
handled upstream by the preflight_driver — this pipeline assumes every row in
TRIED_CORPUS_PATH already passed the eager-vs-Inductor sanity check.
Hitting a Codex CLI rate limit stops the run cleanly with exit code 0; restart
the process once the rate-limit window clears.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from pydantic import ValidationError
from shared.dataset import load_dataset
from shared.logging import get_logger
from shared.models import CorpusRecord, PreflightSafeRecord

from orchestrator.dataset.agent import run_job
from orchestrator.clients.judge_client import RateLimitError
from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)

_stop_requested = False


def _handle_sigint(signum: int, frame: object) -> None:
    global _stop_requested
    if not _stop_requested:
        _stop_requested = True
        _log.info("interrupt received — finishing current example then stopping cleanly")


_REQUIRED_ENV = [
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]


def _load_completed_ids(dataset_path: Path) -> set[str]:
    return {row.example_id for row in load_dataset(dataset_path)}


def _load_corpus(corpus_path: Path) -> list[CorpusRecord]:
    """Load preflight_safe.jsonl and rehydrate fixed training metadata in code."""
    records: list[CorpusRecord] = []
    with corpus_path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(PreflightSafeRecord.model_validate_json(line).to_corpus_record())
            except Exception as exc:
                raise ValueError(f"{corpus_path}:{i} — {exc}") from exc
    return records


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        _log.error("missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    data_dir = Path(os.getenv("TRIED_DATA_DIR", "data"))
    corpus_path = Path(os.getenv("TRIED_CORPUS_PATH", data_dir / "preflight_safe.jsonl"))
    dataset_path = data_dir / "dataset.jsonl"

    if not corpus_path.exists():
        _log.error(
            "corpus not found: %s — run orchestrator.dataset.preflight_driver first",
            corpus_path,
        )
        sys.exit(1)

    _log.info("loading corpus from %s", corpus_path)
    records = _load_corpus(corpus_path)
    _log.info("%d preflight-safe train record(s) loaded", len(records))

    # --- Resume filter (completed examples only — preflight is pre-done) ---
    already_done = _load_completed_ids(dataset_path)
    if already_done:
        _log.info(
            "resuming: %d example(s) already completed — skipping", len(already_done)
        )
    records = [r for r in records if r.example_id not in already_done]
    _log.info("%d record(s) remaining", len(records))

    signal.signal(signal.SIGINT, _handle_sigint)

    client = make_client()

    completed = 0
    transport_errors = 0

    for i, record in enumerate(records):
        _log.info(
            "--- record %d/%d  example_id=%s ---",
            i + 1,
            len(records),
            record.example_id,
        )
        try:
            run_job(record, client, data_dir)
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
                "data validation error  example_id=%s  %s: %s — skipping without recording; "
                "fix the underlying bug then restart to retry this example",
                record.example_id,
                type(exc).__name__,
                exc,
            )
            transport_errors += 1
            continue
        except Exception as exc:
            _log.error(
                "transport error  example_id=%s  %s: %s — will retry on restart",
                record.example_id,
                type(exc).__name__,
                exc,
            )
            transport_errors += 1
            continue

        completed += 1

        if _stop_requested:
            _log.info(
                "stopping after interrupt — %d example(s) completed this run; "
                "restart to continue from here",
                completed,
            )
            sys.exit(0)

    # --- Summary ---
    _log.info("=== run complete ===")
    _log.info("total records:      %d", len(records))
    _log.info("completed:          %d", completed)
    _log.info("transport errors:   %d", transport_errors)


if __name__ == "__main__":
    main()
