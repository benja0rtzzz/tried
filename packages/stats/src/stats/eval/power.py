"""Power-analysis utilities (Week 4 material).

Per docs/eval-stats.md, n is locked at 437 with a 103/217/117 easy/medium/hard
tier split. The functions here are scaffolded for Cohen's h on pass-rate lift,
post-hoc power on the observed effect, and the pre-experiment MDE table at the
locked n.
"""

from __future__ import annotations


def cohens_h(p1: float, p2: float) -> float:
    raise NotImplementedError


def post_hoc_power(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
    raise NotImplementedError


def mde_table(n: int, alpha: float = 0.05, rho: float = 0.7) -> dict:
    """Minimum detectable effect at the locked n. Defaults match the eval-stats
    note: α=0.05, within-pair correlation ρ=0.7."""
    raise NotImplementedError
