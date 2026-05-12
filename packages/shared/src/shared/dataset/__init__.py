"""
Dataset I/O for the TRIED experiment.

Public API
----------
load_corpus_train   — read one or more corpus JSONL files into CorpusRecord list
merge_corpus        — merge + deduplicate multiple corpus files by dataset_id and write to data/
load_dataset        — read the dataset JSONL into DatasetRow list
append_dataset_row  — validate and append one completed row to the dataset
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from shared.logging import get_logger
from shared.models import CorpusRecord, DatasetRow

_log = get_logger(__name__)

__all__ = [
    "load_corpus_train",
    "merge_corpus",
    "load_dataset",
    "append_dataset_row",
]

# ---------------------------------------------------------------------------
# Corpus (read-only from the agent loop's perspective)
# ---------------------------------------------------------------------------

def load_corpus_train(path: str | Path) -> list[CorpusRecord]:
    """Read a corpus JSONL file and return validated CorpusRecord objects.

    Raises ValidationError on the first invalid row (fail fast — corpus rows
    are fixed before data collection and should never be malformed).
    """
    path = Path(path)
    records: list[CorpusRecord] = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CorpusRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{i} — {exc}") from exc
    return records


def merge_corpus(
    *source_paths: str | Path,
    output_path: str | Path,
) -> list[CorpusRecord]:
    """Merge multiple corpus JSONL files, deduplicate by dataset_id, write output.

    Dedup strategy: first-seen wins (earlier paths take precedence).
    Skipped duplicates are reported to stdout so the scraper operator knows.

    Returns the deduplicated list that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: dict[str, CorpusRecord] = {}
    for src in source_paths:
        for record in load_corpus_train(src):
            if record.dataset_id is None:
                raise ValueError(f"{src} record {record.example_id} did not hydrate dataset_id")
            if record.dataset_id in seen:
                _log.warning(
                    "duplicate dataset_id=%s source_id=%s origin=%s (from %s) — skipped",
                    record.dataset_id, record.example_id, record.origin, src,
                )
            else:
                seen[record.dataset_id] = record

    merged = list(seen.values())
    with output_path.open("w") as f:
        for record in merged:
            f.write(record.model_dump_json() + "\n")

    _log.info("merge_corpus wrote %d rows → %s", len(merged), output_path)
    return merged


# ---------------------------------------------------------------------------
# Dataset (append-only during the experiment)
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[DatasetRow]:
    """Read the dataset JSONL and return validated DatasetRow objects."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[DatasetRow] = []
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(DatasetRow.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"{path}:{i} — {exc}") from exc
    return rows


def append_dataset_row(path: str | Path, row: DatasetRow) -> None:
    """Validate and append one completed DatasetRow to the dataset JSONL.

    The file is created if it does not exist. Validation runs even if the
    caller already holds a DatasetRow object — model_dump/model_validate
    round-trips to catch any in-memory mutation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Round-trip to catch any post-construction mutation
    validated = DatasetRow.model_validate(row.model_dump())
    with path.open("a") as f:
        f.write(validated.model_dump_json() + "\n")
