"""Quick summary of a dataset.jsonl run — counts only, no stat tables."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from tabulate import tabulate

from shared.dataset import load_dataset
from shared.enums import CorrectnessStatus

DATASET_PATH = Path(os.getenv("TRIED_DATASET", "data/dataset/dataset.jsonl"))


def _print_report(rows) -> None:
    n_rows = len(rows)
    all_attempts = [a for r in rows for a in r.attempts]
    n_attempts = len(all_attempts)
    if n_rows == 0 or n_attempts == 0:
        print(f"No dataset rows found at {DATASET_PATH}")
        return

    correct = sum(
        1 for r in rows
        if any(
            a.correctness and a.correctness.status == CorrectnessStatus.PASSED
            for a in r.attempts
        )
    )

    class_counts = Counter(
        a.judge_classification.value
        if a.judge_classification is not None
        else "judge_skipped"
        for a in all_attempts
    )
    class_table = [
        (cls, count, f"{100 * count / n_attempts:.1f}%")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    cat_counts = Counter(r.source.op_category.value for r in rows)
    cat_table = [
        (cat, count)
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]
    outcome_counts = Counter(r.final_outcome.value for r in rows)
    outcome_table = [
        (outcome, count, f"{100 * count / n_rows:.1f}%")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1])
    ]
    duplicate_dataset_ids = n_rows - len({r.dataset_id for r in rows})
    duplicate_source_ids = n_rows - len({r.example_id for r in rows})

    print(f"\n{'=' * 50}")
    print(f"  Dataset — {n_rows} rows, {n_attempts} attempts")
    print(f"  Rows with a passing attempt: {correct} / {n_rows}")
    print(f"  Duplicate dataset_id rows: {duplicate_dataset_ids}")
    print(f"  Reused source example_id rows: {duplicate_source_ids}")
    print(f"{'=' * 50}")

    print("\n## Final Outcome Distribution\n")
    print(tabulate(outcome_table, headers=["Outcome", "Count", "%"], tablefmt="simple"))

    print("\n## Judge Classification Distribution\n")
    print(tabulate(class_table, headers=["Classification", "Count", "%"], tablefmt="simple"))

    print("\n## Rows by Op Category\n")
    print(tabulate(cat_table, headers=["Category", "Count"], tablefmt="simple"))
    print()


def main() -> None:
    os.environ.setdefault("TRIED_ROLE", "stats")
    rows = load_dataset(DATASET_PATH)
    _print_report(rows)


if __name__ == "__main__":
    main()
