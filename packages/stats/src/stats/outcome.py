from collections import Counter

from shared.enums import FinalOutcome
from shared.models import DatasetRow

_SUCCESS_OUTCOMES = {
    FinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    FinalOutcome.COMPILED_CORRECT_PARITY,
    FinalOutcome.COMPILED_CORRECT_SLOW,
}


def compute(rows: list[DatasetRow]) -> dict:
    n = len(rows)
    all_attempts = [a for r in rows for a in r.attempts]
    n_attempts = len(all_attempts)

    # Final outcome distribution
    outcome_counts = Counter(r.final_outcome.value for r in rows)
    outcome_table = [
        (outcome, count, f"{100 * count / n:.1f}%")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1])
    ]

    # Judge classification across all attempts
    class_counts = Counter(a.judge_classification.value for a in all_attempts)
    class_table = [
        (cls, count, f"{100 * count / n_attempts:.1f}%")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    # Success rate by op_category
    by_cat: dict[str, dict[str, int]] = {}
    for r in rows:
        cat = r.source.op_category.value
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "success": 0}
        by_cat[cat]["total"] += 1
        if r.final_outcome in _SUCCESS_OUTCOMES:
            by_cat[cat]["success"] += 1

    cat_table = [
        (cat, v["total"], v["success"], f"{100 * v['success'] / v['total']:.0f}%")
        for cat, v in sorted(by_cat.items(), key=lambda x: -x[1]["total"])
    ]

    return {
        "n_rows": n,
        "n_attempts": n_attempts,
        "outcome_table": outcome_table,
        "class_table": class_table,
        "cat_table": cat_table,
    }
