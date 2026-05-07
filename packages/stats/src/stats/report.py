from tabulate import tabulate

from shared.models import DatasetRow
from stats.benchmark import compute as bench_compute
from stats.outcome import compute as outcome_compute


def report(rows: list[DatasetRow]) -> None:
    o = outcome_compute(rows)
    b = bench_compute(rows)

    print(f"\n{'=' * 62}")
    print(f"  TRIED Run — {o['n_rows']} examples, {o['n_attempts']} attempts")
    print(f"{'=' * 62}")

    # ── Outcome distribution ────────────────────────────────────────
    print("\n## Final Outcome Distribution\n")
    print(tabulate(o["outcome_table"], headers=["Outcome", "Count", "%"], tablefmt="simple"))

    # ── Op category success rate ────────────────────────────────────
    print("\n## Success Rate by Op Category\n")
    print(tabulate(o["cat_table"], headers=["Category", "Total", "Success", "Rate"], tablefmt="simple"))

    # ── Judge classification frequencies ───────────────────────────
    print("\n## Judge Classification Distribution (all attempts)\n")
    print(tabulate(o["class_table"], headers=["Classification", "Count", "%"], tablefmt="simple"))

    # ── Benchmark section ───────────────────────────────────────────
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
