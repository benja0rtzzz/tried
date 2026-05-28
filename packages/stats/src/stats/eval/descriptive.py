"""Descriptive statistics over one EvalRecord run.

Covers the single-label descriptive analyses called out in the report:
outcome distribution (raw and the 5-bucket report mapping), Wilson CIs on
per-tier pass rate, per-method timing IQR, speedup summaries, and
Triton static-validation latency descriptives.

Paired tests are in hypothesis.py; power/MDE in power.py.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Iterable

from shared.enums import EvalFinalOutcome
from shared.models import EvalRecord


# ---------------------------------------------------------------------------
# Outcome bucketing
# ---------------------------------------------------------------------------

SUCCESS_OUTCOMES: frozenset[EvalFinalOutcome] = frozenset({
    EvalFinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    EvalFinalOutcome.COMPILED_CORRECT_PARITY,
    EvalFinalOutcome.COMPILED_CORRECT_SLOW,
})


REPORT_BUCKETS: list[str] = [
    "correct_faster",
    "correct_parity",
    "correct_slow",
    "correctness_failed",
    "compile_or_runtime_fail",
]


def _report_bucket(outcome: EvalFinalOutcome) -> str:
    """Map a raw EvalFinalOutcome to the 5-bucket vocabulary the report uses
    in §2.1 (collapses compile_fail and runtime_fail into one bucket)."""
    if outcome == EvalFinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR:
        return "correct_faster"
    if outcome == EvalFinalOutcome.COMPILED_CORRECT_PARITY:
        return "correct_parity"
    if outcome == EvalFinalOutcome.COMPILED_CORRECT_SLOW:
        return "correct_slow"
    if outcome == EvalFinalOutcome.CORRECTNESS_FAILED:
        return "correctness_failed"
    return "compile_or_runtime_fail"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Defaults to 95%."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def wilson_diff_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe (1998) hybrid score interval for p1 - p2 with independent
    samples. Used as the IC 95% Δp column in the report's McNemar table
    (a robust unpaired CI on the difference of proportions; the paired
    McNemar χ² supplies significance separately)."""
    lo1, hi1 = wilson_ci(k1, n1, z=z)
    lo2, hi2 = wilson_ci(k2, n2, z=z)
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2)
    hi = diff + math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    if len(xs) == 1:
        return (xs[0], xs[0], xs[0])
    qs = statistics.quantiles(xs, n=4, method="inclusive")
    return (qs[0], qs[1], qs[2])


def describe_series(xs: list[float]) -> dict:
    """Standard descriptive summary used in every report table."""
    if not xs:
        return {"n": 0}
    q1, med, q3 = _quartiles(xs)
    return {
        "n": len(xs),
        "min": min(xs),
        "q1": q1,
        "median": med,
        "mean": statistics.fmean(xs),
        "q3": q3,
        "max": max(xs),
        "iqr": q3 - q1,
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def winners(rows: Iterable[EvalRecord]) -> list[EvalRecord]:
    return [r for r in rows if r.final_winning_attempt_n is not None]


# ---------------------------------------------------------------------------
# Public analyses
# ---------------------------------------------------------------------------


def outcome_distribution(rows: list[EvalRecord]) -> dict:
    """Raw + report-bucketed counts, total and per-tier.

    `total` and `by_tier` are keyed by the raw EvalFinalOutcome.value;
    `report_total` and `report_by_tier` are keyed by REPORT_BUCKETS — that
    is the shape consumed by the §2.1 outcome distribution table and the
    stacked bar in §2.2 (Figure 4).
    """
    n = len(rows)
    raw_total = Counter(r.final_outcome.value for r in rows)
    raw_by_tier: dict[str, Counter] = {}
    rep_total = Counter(_report_bucket(r.final_outcome) for r in rows)
    rep_by_tier: dict[str, Counter] = {}
    for r in rows:
        raw_by_tier.setdefault(r.spec.tier.value, Counter())[r.final_outcome.value] += 1
        rep_by_tier.setdefault(r.spec.tier.value, Counter())[_report_bucket(r.final_outcome)] += 1
    return {
        "n": n,
        "total": dict(raw_total),
        "by_tier": {tier: dict(c) for tier, c in raw_by_tier.items()},
        "report_total": dict(rep_total),
        "report_by_tier": {tier: dict(c) for tier, c in rep_by_tier.items()},
    }


def pass_rate_by_tier(rows: list[EvalRecord]) -> dict:
    """Per-tier pass rate (any compiled-correct outcome) with Wilson 95% CI,
    plus an `__overall__` entry. Matches the §2.1 "Tabla resumen" + the
    per-tier subtable."""
    by_tier: dict[str, list[EvalRecord]] = {}
    for r in rows:
        by_tier.setdefault(r.spec.tier.value, []).append(r)

    out: dict[str, dict] = {}
    overall_k = sum(1 for r in rows if r.final_outcome in SUCCESS_OUTCOMES)
    overall_n = len(rows)
    lo, hi = wilson_ci(overall_k, overall_n)
    out["__overall__"] = {
        "k": overall_k, "n": overall_n,
        "rate": overall_k / overall_n if overall_n else 0.0,
        "wilson_lo": lo, "wilson_hi": hi,
    }
    for tier, group in sorted(by_tier.items()):
        k = sum(1 for r in group if r.final_outcome in SUCCESS_OUTCOMES)
        n = len(group)
        lo, hi = wilson_ci(k, n)
        out[tier] = {
            "k": k, "n": n, "rate": k / n if n else 0.0,
            "wilson_lo": lo, "wilson_hi": hi,
        }
    return out


def speedup_summary(rows: list[EvalRecord]) -> dict:
    """Per-method speedup summary (median/IQR/min/max) over the winning
    attempts. Selection-biased — see report §1.10 constructo (2)."""
    ws = winners(rows)
    vs_eager = [
        r.attempts[r.final_winning_attempt_n].benchmark.speedup_vs_eager  # type: ignore[union-attr]
        for r in ws
    ]
    vs_inductor = [
        r.attempts[r.final_winning_attempt_n].benchmark.speedup_vs_inductor  # type: ignore[union-attr]
        for r in ws
    ]
    return {
        "n_winners": len(ws),
        "vs_eager": describe_series(vs_eager),
        "vs_inductor": describe_series(vs_inductor),
        "log_vs_eager": describe_series([math.log(x) for x in vs_eager]),
        "log_vs_inductor": describe_series([math.log(x) for x in vs_inductor]),
    }


def timing_iqr(rows: list[EvalRecord]) -> dict:
    """Per-method median-timing distribution across winning rows.
    Raw 100-iter samples are consumed by hypothesis.wilcoxon_speedup_samples,
    not here."""
    ws = winners(rows)
    triton = [r.attempts[r.final_winning_attempt_n].benchmark.triton_ms for r in ws]       # type: ignore[union-attr]
    eager = [r.attempts[r.final_winning_attempt_n].benchmark.eager_ms for r in ws]         # type: ignore[union-attr]
    inductor = [r.attempts[r.final_winning_attempt_n].benchmark.inductor_ms for r in ws]   # type: ignore[union-attr]
    return {
        "n_winners": len(ws),
        "triton_ms": describe_series(triton),
        "eager_ms": describe_series(eager),
        "inductor_ms": describe_series(inductor),
    }


def triton_compile_stats(rows: list[EvalRecord]) -> dict:
    """Descriptive stats on /compile latency (winning attempts only).
    `/compile` is static validation — not shape-aware launch JIT — so this
    is presented as descriptive context, not as model-comparison evidence."""
    ws = winners(rows)
    triton_compile_ms = [
        float(r.attempts[r.final_winning_attempt_n].latency.compile_ms)  # type: ignore[union-attr]
        for r in ws
    ]
    return {"triton_compile_ms": describe_series(triton_compile_ms)}


# ---------------------------------------------------------------------------
# Joined helpers used by hypothesis/power
# ---------------------------------------------------------------------------


def pass_array(rows: list[EvalRecord]) -> dict[str, int]:
    """example_id -> 1 if final_outcome is a success bucket, else 0.
    The map shape is the join primitive used by Cochran's Q and McNemar."""
    return {
        r.example_id: 1 if r.final_outcome in SUCCESS_OUTCOMES else 0
        for r in rows
    }


def speedup_map(
    rows: list[EvalRecord], *, against: str = "inductor"
) -> dict[str, float]:
    """example_id -> speedup_vs_<against> on the winning attempt. Skips rows
    without a winning attempt. Used to align speedup vectors across labels
    for Friedman / Wilcoxon."""
    if against not in {"eager", "inductor"}:
        raise ValueError(f"against must be 'eager' or 'inductor', got {against!r}")
    out: dict[str, float] = {}
    for r in rows:
        if r.final_winning_attempt_n is None:
            continue
        bench = r.attempts[r.final_winning_attempt_n].benchmark
        if bench is None:
            continue
        out[r.example_id] = getattr(bench, f"speedup_vs_{against}")
    return out
