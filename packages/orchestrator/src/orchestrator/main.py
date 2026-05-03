"""
Entry point for the TRIED orchestrator.

Usage (from the project root):
    TRIED_ROLE=orchestrator uv run python -m orchestrator.main

Required env vars: see packages/orchestrator/.env
Optional env vars:
    TRIED_DATA_DIR   — path to the data directory (default: data/)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from shared.dataset import append_skipped, load_corpus_train
from shared.enums import FinalOutcome, Split
from shared.logging import get_logger

from orchestrator.agent import run_job
from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)

_REQUIRED_ENV = [
    "GEMINI_API_KEY",
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]


def main() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        _log.error("missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    data_dir = Path(os.getenv("TRIED_DATA_DIR", "data"))
    corpus_path = data_dir / "corpus_train.jsonl"

    _log.info("loading corpus from %s", corpus_path)
    all_records = load_corpus_train(corpus_path)
    records = [r for r in all_records if r.split == Split.TRAIN]
    _log.info(
        "%d train records loaded  (total corpus: %d)",
        len(records),
        len(all_records),
    )

    client = make_client()

    outcome_counts: Counter[str] = Counter()
    preflight_skipped = 0
    transport_errors = 0

    for i, record in enumerate(records):
        _log.info(
            "--- record %d/%d  example_id=%s ---",
            i + 1,
            len(records),
            record.example_id,
        )
        try:
            outcome = run_job(record, client, data_dir)
        except Exception as exc:
            _log.error(
                "transport error  example_id=%s  %s: %s",
                record.example_id,
                type(exc).__name__,
                exc,
            )
            append_skipped(
                data_dir / "skipped.jsonl",
                record.example_id,
                f"{type(exc).__name__}: {exc}",
            )
            transport_errors += 1
            continue

        if outcome is None:
            preflight_skipped += 1
        else:
            outcome_counts[outcome.value] += 1

    # --- Summary ---
    total = len(records)
    completed = total - preflight_skipped - transport_errors
    _log.info("=== run complete ===")
    _log.info("total records:      %d", total)
    _log.info("completed:          %d", completed)
    _log.info("preflight skipped:  %d", preflight_skipped)
    _log.info("transport errors:   %d", transport_errors)
    if outcome_counts:
        _log.info("outcomes:")
        for outcome in FinalOutcome:
            count = outcome_counts.get(outcome.value, 0)
            if count:
                _log.info("  %-45s %d", outcome.value, count)


if __name__ == "__main__":
    main()
