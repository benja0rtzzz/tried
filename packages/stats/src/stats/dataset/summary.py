"""Quick summary of a dataset.jsonl run — counts only, no stat tables."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from tabulate import tabulate

from shared.dataset import load_dataset
from shared.enums import CorrectnessStatus

DATASET_PATH = Path(os.getenv("TRIED_DATASET", "data/dataset.jsonl"))


def _print_report(rows) -> None:
    n_rows = len(rows)
    all_attempts = [a for r in rows for a in r.attempts]
    n_attempts = len(all_attempts)

    correct = sum(
        1 for r in rows
        if any(
            a.correctness and a.correctness.status == CorrectnessStatus.PASSED
            for a in r.attempts
        )
    )

    class_counts = Counter(a.judge_classification.value for a in all_attempts)
    class_table = [
        (cls, count, f"{100 * count / n_attempts:.1f}%")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    cat_counts = Counter(r.source.op_category.value for r in rows)
    cat_table = [
        (cat, count)
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    print(f"\n{'=' * 50}")
    print(f"  Dataset — {n_rows} rows, {n_attempts} attempts")
    print(f"  Rows with a passing attempt: {correct} / {n_rows}")
    print(f"{'=' * 50}")

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
