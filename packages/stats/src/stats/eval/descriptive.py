"""Descriptive statistics over one EvalRecord run.

Covers single-label descriptive eval analyses that don't need scipy: outcome
distribution, Wilson CIs on per-tier pass rate, timing IQR from raw samples,
speedup summaries, and Triton static-validation latency descriptives. Paired tests are
scaffolded in hypothesis.py for the second eval run.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Iterable

from shared.enums import EvalFinalOutcome
from shared.models import EvalRecord


SUCCESS_OUTCOMES: frozenset[EvalFinalOutcome] = frozenset({
    EvalFinalOutcome.COMPILED_CORRECT_FASTER_THAN_INDUCTOR,
    EvalFinalOutcome.COMPILED_CORRECT_PARITY,
    EvalFinalOutcome.COMPILED_CORRECT_SLOW,
})


# ---------------------------------------------------------------------------
# Helpers


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    """(Q1, median, Q3) using the inclusive method to match numpy default."""
    if not xs:
        return (0.0, 0.0, 0.0)
    if len(xs) == 1:
        return (xs[0], xs[0], xs[0])
    qs = statistics.quantiles(xs, n=4, method="inclusive")
    return (qs[0], qs[1], qs[2])


def _describe(xs: list[float]) -> dict:
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


def _winning(rows: Iterable[EvalRecord]) -> list[EvalRecord]:
    return [r for r in rows if r.final_winning_attempt_n is not None]


# ---------------------------------------------------------------------------
# Public analyses


def outcome_distribution(rows: list[EvalRecord]) -> dict:
    """Total + per-tier counts for each EvalFinalOutcome."""
    n = len(rows)
    total = Counter(r.final_outcome.value for r in rows)
    by_tier: dict[str, Counter] = {}
    for r in rows:
        by_tier.setdefault(r.spec.tier.value, Counter())[r.final_outcome.value] += 1
    return {
        "n": n,
        "total": dict(total),
        "by_tier": {tier: dict(c) for tier, c in by_tier.items()},
    }


def pass_rate_by_tier(rows: list[EvalRecord]) -> dict:
    """Per-tier pass rate (any compiled-correct outcome) with Wilson 95% CI."""
    by_tier: dict[str, list[EvalRecord]] = {}
    for r in rows:
        by_tier.setdefault(r.spec.tier.value, []).append(r)

    out: dict[str, dict] = {}
    overall_k = sum(1 for r in rows if r.final_outcome in SUCCESS_OUTCOMES)
    overall_n = len(rows)
    lo, hi = _wilson_ci(overall_k, overall_n)
    out["__overall__"] = {"k": overall_k, "n": overall_n,
                         "rate": overall_k / overall_n if overall_n else 0.0,
                         "wilson_lo": lo, "wilson_hi": hi}

    for tier, group in sorted(by_tier.items()):
        k = sum(1 for r in group if r.final_outcome in SUCCESS_OUTCOMES)
        n = len(group)
        lo, hi = _wilson_ci(k, n)
        out[tier] = {"k": k, "n": n, "rate": k / n if n else 0.0,
                     "wilson_lo": lo, "wilson_hi": hi}
    return out


def speedup_summary(rows: list[EvalRecord]) -> dict:
    """Mean / median / std / IQR over winning-attempt speedup vs eager and inductor."""
    winners = _winning(rows)
    vs_eager = [
        r.attempts[r.final_winning_attempt_n].benchmark.speedup_vs_eager  # type: ignore[union-attr]
        for r in winners
    ]
    vs_inductor = [
        r.attempts[r.final_winning_attempt_n].benchmark.speedup_vs_inductor  # type: ignore[union-attr]
        for r in winners
    ]
    return {
        "n_winners": len(winners),
        "vs_eager": _describe(vs_eager),
        "vs_inductor": _describe(vs_inductor),
        "log_vs_eager": _describe([math.log(x) for x in vs_eager]),
        "log_vs_inductor": _describe([math.log(x) for x in vs_inductor]),
    }


def timing_iqr(rows: list[EvalRecord]) -> dict:
    """Per-method timing distribution (Q1/median/Q3/IQR) over the per-row median
    of each method, computed across winning rows. The raw 100-iter samples are
    consumed by hypothesis.wilcoxon_log_speedup, not here."""
    winners = _winning(rows)
    triton = [r.attempts[r.final_winning_attempt_n].benchmark.triton_ms for r in winners]      # type: ignore[union-attr]
    eager = [r.attempts[r.final_winning_attempt_n].benchmark.eager_ms for r in winners]        # type: ignore[union-attr]
    inductor = [r.attempts[r.final_winning_attempt_n].benchmark.inductor_ms for r in winners]  # type: ignore[union-attr]
    return {
        "n_winners": len(winners),
        "triton_ms": _describe(triton),
        "eager_ms": _describe(eager),
        "inductor_ms": _describe(inductor),
    }


def triton_compile_stats(rows: list[EvalRecord]) -> dict:
    """Descriptive stats on /compile latency (winning attempts only).

    The current /compile endpoint performs static candidate validation: import,
    @triton.jit kernel detection, and wrapper detection. Shape-aware Triton
    launch compilation happens later in /run, so this is not compile-time
    evidence for model comparison."""
    winners = _winning(rows)
    triton_compile_ms = [
        float(r.attempts[r.final_winning_attempt_n].latency.compile_ms)  # type: ignore[union-attr]
        for r in winners
    ]
    return {"triton_compile_ms": _describe(triton_compile_ms)}
