"""Dataset-run summary printer. Consolidates the prior outcome / benchmark
/ report scripts into one entry point. Output is byte-for-byte the same as
before the consolidation."""

from __future__ import annotations

import os
import statistics
from collections import Counter
from pathlib import Path

from tabulate import tabulate

from shared.dataset import load_dataset
from shared.enums import FinalOutcome
from shared.models import Benchmark, DatasetRow

DATASET_PATH = Path(os.getenv("TRIED_DATASET", "data/dataset.jsonl"))

_SUCCESS_OUTCOMES = {
    FinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    FinalOutcome.COMPILED_CORRECT_PARITY,
    FinalOutcome.COMPILED_CORRECT_SLOW,
}


def _describe(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _outcome_stats(rows: list[DatasetRow]) -> dict:
    n = len(rows)
    all_attempts = [a for r in rows for a in r.attempts]
    n_attempts = len(all_attempts)

    outcome_counts = Counter(r.final_outcome.value for r in rows)
    outcome_table = [
        (outcome, count, f"{100 * count / n:.1f}%")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1])
    ]

    class_counts = Counter(a.judge_classification.value for a in all_attempts)
    class_table = [
        (cls, count, f"{100 * count / n_attempts:.1f}%")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

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


def _benchmark_stats(rows: list[DatasetRow]) -> dict:
    benchmarks: list[tuple[str, Benchmark]] = []
    for r in rows:
        win_n = r.final_winning_attempt_n
        if win_n is not None:
            attempt = r.attempts[win_n]
            if attempt.benchmark is not None:
                benchmarks.append((r.final_outcome.value, attempt.benchmark))

    if not benchmarks:
        return {"n_benchmarked": 0}

    speedup_eager    = [b.speedup_vs_eager    for _, b in benchmarks]
    speedup_inductor = [b.speedup_vs_inductor for _, b in benchmarks]
    triton_ms        = [b.triton_ms           for _, b in benchmarks]
    eager_ms         = [b.eager_ms            for _, b in benchmarks]
    inductor_ms      = [b.inductor_ms         for _, b in benchmarks]

    speedup_table = [
        (
            outcome,
            f"{b.speedup_vs_eager:.2f}x",
            f"{b.speedup_vs_inductor:.2f}x",
            f"{b.triton_ms:.3f}",
            f"{b.eager_ms:.3f}",
            f"{b.inductor_ms:.3f}",
        )
        for outcome, b in benchmarks
    ]

    return {
        "n_benchmarked":      len(benchmarks),
        "speedup_vs_eager":   _describe(speedup_eager),
        "speedup_vs_inductor":_describe(speedup_inductor),
        "triton_ms":          _describe(triton_ms),
        "eager_ms":           _describe(eager_ms),
        "inductor_ms":        _describe(inductor_ms),
        "speedup_table":      speedup_table,
    }


def _print_report(rows: list[DatasetRow]) -> None:
    o = _outcome_stats(rows)
    b = _benchmark_stats(rows)

    print(f"\n{'=' * 62}")
    print(f"  TRIED Run — {o['n_rows']} examples, {o['n_attempts']} attempts")
    print(f"{'=' * 62}")

    print("\n## Final Outcome Distribution\n")
    print(tabulate(o["outcome_table"], headers=["Outcome", "Count", "%"], tablefmt="simple"))

    print("\n## Success Rate by Op Category\n")
    print(tabulate(o["cat_table"], headers=["Category", "Total", "Success", "Rate"], tablefmt="simple"))

    print("\n## Judge Classification Distribution (all attempts)\n")
    print(tabulate(o["class_table"], headers=["Classification", "Count", "%"], tablefmt="simple"))

    n_bench = b.get("n_benchmarked", 0)
    if n_bench == 0:
        print("\n(no benchmarked attempts in dataset)\n")
        return

    def _speedup_row(label: str, d: dict) -> list:
        return [label, d["n"], f"{d['min']:.2f}x", f"{d['median']:.2f}x",
                f"{d['mean']:.2f}x", f"{d['max']:.2f}x", f"{d['std']:.2f}x"]

    print(f"\n## Speedup Summary ({n_bench} benchmarked winning attempts)\n")
    print(tabulate(
        [
            _speedup_row("vs eager",    b["speedup_vs_eager"]),
            _speedup_row("vs inductor", b["speedup_vs_inductor"]),
        ],
        headers=["Metric", "n", "min", "median", "mean", "max", "std"],
        tablefmt="simple",
    ))

    def _ms_row(label: str, d: dict) -> list:
        return [label, f"{d['min']:.3f}", f"{d['median']:.3f}",
                f"{d['mean']:.3f}", f"{d['max']:.3f}"]

    print("\n## Absolute Timing Summary (ms)\n")
    print(tabulate(
        [
            _ms_row("triton",   b["triton_ms"]),
            _ms_row("eager",    b["eager_ms"]),
            _ms_row("inductor", b["inductor_ms"]),
        ],
        headers=["Implementation", "min", "median", "mean", "max"],
        tablefmt="simple",
    ))

    print("\n## Per-Example Speedup Detail\n")
    print(tabulate(
        b["speedup_table"],
        headers=["Outcome", "vs eager", "vs inductor", "triton_ms", "eager_ms", "inductor_ms"],
        tablefmt="simple",
    ))
    print()


def main() -> None:
    os.environ.setdefault("TRIED_ROLE", "stats")
    rows = load_dataset(DATASET_PATH)
    _print_report(rows)


if __name__ == "__main__":
    main()
