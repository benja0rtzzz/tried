"""Load EvalRecord JSONL for one or two model_labels and join by example_id.

Layout per docs/eval-stats.md:
    eval/results/<model_label>/eval_rows.jsonl
"""

from __future__ import annotations

from pathlib import Path

from shared.models import EvalRecord


def results_path(model_label: str, results_root: Path = Path("eval/results")) -> Path:
    return results_root / model_label / "eval_rows.jsonl"


def load_label(
    model_label: str,
    results_root: Path = Path("eval/results"),
) -> list[EvalRecord]:
    path = results_path(model_label, results_root)
    rows: list[EvalRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(EvalRecord.model_validate_json(line))
    return rows


def join_labels(
    label_a: str,
    label_b: str,
    results_root: Path = Path("eval/results"),
) -> list[tuple[EvalRecord, EvalRecord]]:
    """Inner-join two label runs on example_id. Order follows label_a's file."""
    rows_a = load_label(label_a, results_root)
    by_id_b = {r.example_id: r for r in load_label(label_b, results_root)}
    return [(a, by_id_b[a.example_id]) for a in rows_a if a.example_id in by_id_b]
