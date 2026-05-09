"""Paired hypothesis tests across two model_labels (Week 3 material).

Headline tests per docs/eval-stats.md:
- paired McNemar on pass rate (final_outcome ∈ success set, joined by example_id)
- paired Wilcoxon signed-rank on log-speedup (winning-attempt speedup_vs_inductor)
- paired t-test on log Triton compile time (attempts[winning].latency.compile_ms)

Inputs are the joined pairs returned by stats.eval.load.join_labels.
"""

from __future__ import annotations

from shared.models import EvalRecord


def mcnemar_pass_rate(pairs: list[tuple[EvalRecord, EvalRecord]]) -> dict:
    """Paired McNemar on success vs non-success final_outcome."""
    raise NotImplementedError


def wilcoxon_log_speedup(
    pairs: list[tuple[EvalRecord, EvalRecord]],
    against: str = "inductor",
) -> dict:
    """Paired Wilcoxon signed-rank on log(speedup_vs_<against>) for rows where
    both sides have a winning attempt."""
    raise NotImplementedError


def paired_t_log_triton_compile(pairs: list[tuple[EvalRecord, EvalRecord]]) -> dict:
    """Paired t-test on log(latency.compile_ms) of the winning attempt on each
    side. Triton compile is genuinely cold per attempt (no shared cache like
    the dropped Inductor baseline), so the test measures real compile-cost
    differences between vanilla and fine-tuned generators."""
    raise NotImplementedError
