import statistics

from shared.models import Benchmark, DatasetRow


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


def compute(rows: list[DatasetRow]) -> dict:
    # Only winning attempts that have benchmark data
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
