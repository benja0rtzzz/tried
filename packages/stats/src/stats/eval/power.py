"""Power-analysis utilities (report §1.7 / §2.3 power column).

Cohen's h, post-hoc power on a paired-proportion contrast under McNemar,
and the pre-experiment MDE table at the locked n = 437. The MDE table is
derived from the two-proportion z-test approximation the report uses; it
is the conservative lower bound vs the true paired-McNemar power, which
depends on the discordance rate.
"""

from __future__ import annotations

import math

import scipy.stats as sps


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h on two proportions. Sign matches p1 − p2.

    Convention: h ≈ 0.2 small, 0.5 medium, 0.8 large.
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def post_hoc_power(
    p1: float, p2: float, n: int, alpha: float = 0.05, two_sided: bool = True
) -> float:
    """Approximate power for detecting p1 vs p2 at sample size n.

    Uses the two-proportion arcsine z-approximation (same formula the
    report uses in §1.7 for its a-priori n calculation). For paired
    McNemar this is conservative — the true power is somewhat higher when
    within-pair correlation is positive.
    """
    h = abs(cohens_h(p1, p2))
    if h == 0 or n <= 0:
        return alpha
    z_alpha = sps.norm.ppf(1 - alpha / 2) if two_sided else sps.norm.ppf(1 - alpha)
    z_beta = h * math.sqrt(n) - z_alpha
    return float(sps.norm.cdf(z_beta))


def mde_table(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_rates: list[float] | None = None,
) -> dict:
    """Minimum detectable effect at the locked n.

    For each baseline p, returns the smallest h (and the smallest absolute
    Δp at that baseline) that achieves the target power under the same
    two-proportion arcsine approximation. Defaults to the four baselines
    the report cares about (0.05, 0.10, 0.20, 0.30).

    `n` for the per-tier rows: easy=103, medium=217, hard=117 → call this
    function once per tier to fill the §1.7 / §2.4 per-tier MDE table.
    """
    if baseline_rates is None:
        baseline_rates = [0.05, 0.10, 0.20, 0.30]
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    mde_h = (z_alpha + z_beta) / math.sqrt(n)

    rows = []
    for p in baseline_rates:
        delta_p = _h_to_delta_p(p, mde_h)
        rows.append({"baseline_p": p, "mde_h": mde_h, "mde_delta_p": delta_p})
    return {
        "n": n,
        "alpha": alpha,
        "power": power,
        "mde_h": mde_h,
        "rows": rows,
    }


def _h_to_delta_p(p1: float, h: float) -> float:
    """Smallest p2 > p1 satisfying Cohen's h = h, then return p2 - p1.

    Inverts h = 2(asin√p2 − asin√p1) for p2 ∈ [0, 1]. Returns NaN if the
    target arcsin exceeds π/2."""
    target = math.asin(math.sqrt(p1)) + h / 2
    if target > math.pi / 2:
        return float("nan")
    p2 = math.sin(target) ** 2
    return p2 - p1
