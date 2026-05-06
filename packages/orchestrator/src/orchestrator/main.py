"""
Entry point for the TRIED orchestrator.

Usage (from the project root):
    TRIED_ROLE=orchestrator uv run python -m orchestrator.main

Required env vars: see packages/orchestrator/.env
Optional env vars:
    TRIED_DATA_DIR   — path to the data directory (default: data/)

Resume behaviour: on startup, already-completed and preflight-skipped
example_ids are loaded from dataset.jsonl / skipped.jsonl and filtered out so
no example is processed twice. Hitting the Gemini daily quota stops the run
cleanly with exit code 0; restart the process after the quota resets.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from shared.dataset import append_skipped, load_corpus_train, load_dataset
from shared.enums import FinalOutcome, Split
from shared.logging import get_logger

from orchestrator.agent import run_job
from orchestrator.clients.judge_client import RateLimitError
from orchestrator.clients.verification_client import make_client

_log = get_logger(__name__)

_REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "VERIFICATION_SERVER_URL",
    "VERIFICATION_API_KEY",
]


def _load_completed_ids(dataset_path: Path) -> set[str]:
    return {row.example_id for row in load_dataset(dataset_path)}


def _load_skipped_ids(skipped_path: Path) -> set[str]:
    if not skipped_path.exists():
        return set()
    ids: set[str] = set()
    with skipped_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def main() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        _log.error("missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    data_dir = Path(os.getenv("TRIED_DATA_DIR", "data"))
    corpus_path = data_dir / "corpus_train.jsonl"
    dataset_path = data_dir / "dataset.jsonl"
    skipped_path = data_dir / "skipped.jsonl"

    _log.info("loading corpus from %s", corpus_path)
    all_records = load_corpus_train(corpus_path)
    records = [r for r in all_records if r.split == Split.TRAIN]
    _log.info(
        "%d train records loaded  (total corpus: %d)",
        len(records),
        len(all_records),
    )

    # --- Resume filter ---
    already_done = _load_completed_ids(dataset_path) | _load_skipped_ids(skipped_path)
    if already_done:
        _log.info("resuming: %d example(s) already processed — skipping", len(already_done))
    records = [r for r in records if r.example_id not in already_done]
    _log.info("%d record(s) remaining", len(records))

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
        except RateLimitError as exc:
            completed = sum(outcome_counts.values())
            _log.error("Gemini rate limit hit: %s", exc)
            _log.error(
                "stopping cleanly — %d example(s) completed this run; "
                "restart after the daily quota resets (midnight US/Pacific)",
                completed,
            )
            sys.exit(0)
        except Exception as exc:
            _log.error(
                "transport error  example_id=%s  %s: %s",
                record.example_id,
                type(exc).__name__,
                exc,
            )
            append_skipped(
                skipped_path,
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
