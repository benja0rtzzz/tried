"""Inferential tests (report §2.3).

Headline tests:
- Omnibus pass-rate: Cochran's Q over N labels on the binary pass matrix
  joined by example_id.
- Pairwise pass-rate: McNemar (exact binomial when b+c is small, otherwise
  the asymptotic χ² with continuity correction) with Holm step-down on the
  user-supplied set of planned contrasts.
- Paired speedup: Wilcoxon signed-rank on log(speedup_vs_*) over the rows
  where both labels produced a winning attempt.
- Speedup ANOVA-like: Friedman over the K-way intersection of winners, with
  Dunn's post-hoc (Holm).
- Normality: Shapiro-Wilk per condition on the per-row median speedup.

Every function returns a plain dict so the report renderer doesn't need
scipy types in scope.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import scipy.stats as sps
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

from shared.models import EvalRecord

from stats.eval.descriptive import (
    SUCCESS_OUTCOMES,
    pass_array,
    speedup_map,
    wilson_diff_ci,
)


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm step-down correction.

    Returns one entry per input p-value (in input order) with:
        - p_raw:    the input value
        - p_holm:   the Holm-adjusted p-value
        - threshold: the per-rank Holm threshold (α / (m - i)) that would be
                     compared against the sorted p-value at this rank
        - reject:   True iff the adjusted p-value is below alpha
    """
    m = len(p_values)
    if m == 0:
        return {"alpha": alpha, "m": 0, "results": []}
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        candidate = min(1.0, factor * p_values[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    results = []
    rank_of = {idx: rank for rank, idx in enumerate(order)}
    for i, p in enumerate(p_values):
        rank = rank_of[i]
        threshold = alpha / (m - rank)
        results.append({
            "p_raw": float(p),
            "p_holm": float(adjusted[i]),
            "threshold": threshold,
            "reject": adjusted[i] < alpha,
        })
    return {"alpha": alpha, "m": m, "results": results}


# ---------------------------------------------------------------------------
# Joined pass/speedup data preparation
# ---------------------------------------------------------------------------


def build_pass_matrix(
    labels_to_rows: Mapping[str, list[EvalRecord]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Inner-join the labels' pass arrays on example_id.

    Returns (label_order, example_ids, matrix) where matrix is shape
    (n_examples, n_labels) of 0/1. label_order preserves dict iteration
    order; example_ids is the sorted intersection."""
    label_order = list(labels_to_rows.keys())
    pass_maps = {label: pass_array(rows) for label, rows in labels_to_rows.items()}
    common = sorted(set.intersection(*(set(m.keys()) for m in pass_maps.values())))
    matrix = np.array(
        [[pass_maps[label][eid] for label in label_order] for eid in common],
        dtype=int,
    )
    return label_order, common, matrix


def build_speedup_matrix(
    labels_to_rows: Mapping[str, list[EvalRecord]],
    *,
    against: str = "inductor",
) -> tuple[list[str], list[str], np.ndarray]:
    """Same as build_pass_matrix but for speedup, restricted to the example
    ids where every label produced a winning attempt."""
    label_order = list(labels_to_rows.keys())
    sp_maps = {
        label: speedup_map(rows, against=against)
        for label, rows in labels_to_rows.items()
    }
    common = sorted(set.intersection(*(set(m.keys()) for m in sp_maps.values())))
    matrix = np.array(
        [[sp_maps[label][eid] for label in label_order] for eid in common],
        dtype=float,
    )
    return label_order, common, matrix


# ---------------------------------------------------------------------------
# Omnibus + pairwise pass-rate
# ---------------------------------------------------------------------------


def cochran_q_pass_rate(
    labels_to_rows: Mapping[str, list[EvalRecord]],
) -> dict:
    """Cochran's Q on the binary pass matrix.

    The 6-condition report § 2.3 omnibus. df = K - 1 where K is the number
    of labels. ε² ≈ (Q - df) / (n(K - 1) - Q) is reported as an effect-size
    proxy."""
    label_order, common, matrix = build_pass_matrix(labels_to_rows)
    if matrix.shape[0] == 0:
        return {
            "labels": label_order, "n": 0, "k": matrix.shape[1],
            "statistic": float("nan"), "df": matrix.shape[1] - 1,
            "p_value": float("nan"), "epsilon_squared": float("nan"),
            "skipped_reason": "empty intersection of example_ids",
        }
    n, k = matrix.shape
    col_sums = matrix.sum(axis=0)
    if np.all(col_sums == col_sums[0]):
        return {
            "labels": label_order, "n": n, "k": k,
            "statistic": 0.0, "df": k - 1,
            "p_value": 1.0, "epsilon_squared": 0.0,
            "skipped_reason": "all columns have identical pass totals (degenerate)",
        }
    result = cochrans_q(matrix, return_object=True)
    denom = n * (k - 1) - result.statistic
    eps2 = (result.statistic - (k - 1)) / denom if denom > 0 else 0.0
    return {
        "labels": label_order,
        "n": n,
        "k": k,
        "statistic": float(result.statistic),
        "df": int(result.df),
        "p_value": float(result.pvalue),
        "epsilon_squared": float(eps2),
        "skipped_reason": None,
    }


def pairwise_mcnemar(
    rows_a: list[EvalRecord],
    rows_b: list[EvalRecord],
) -> dict:
    """Paired McNemar on success vs non-success, joined by example_id.

    Reports:
        - b (a-success, b-failure), c (a-failure, b-success)
        - p_a (a's pass rate over the common set), p_b
        - statistic (χ² or exact stat), p_value
        - exact: True iff the exact binomial branch was taken (b+c < 25)
        - cohen_h: 2(asin√p_a − asin√p_b)  — directional, sign matches p_a − p_b
        - delta_p_ci_95: Newcombe Δp CI (descriptive; the paired test is the
          source of significance)
    """
    pa = pass_array(rows_a)
    pb = pass_array(rows_b)
    common = sorted(set(pa) & set(pb))
    if not common:
        return {"n_common": 0}
    a_arr = np.array([pa[e] for e in common], dtype=int)
    b_arr = np.array([pb[e] for e in common], dtype=int)
    n11 = int(np.sum((a_arr == 1) & (b_arr == 1)))
    n10 = int(np.sum((a_arr == 1) & (b_arr == 0)))
    n01 = int(np.sum((a_arr == 0) & (b_arr == 1)))
    n00 = int(np.sum((a_arr == 0) & (b_arr == 0)))
    table = np.array([[n11, n10], [n01, n00]])
    exact = (n10 + n01) < 25
    result = mcnemar(table, exact=exact, correction=True)
    k_a, k_b = int(a_arr.sum()), int(b_arr.sum())
    n = len(common)
    p_a, p_b = k_a / n, k_b / n
    cohen_h = 2 * (math.asin(math.sqrt(p_a)) - math.asin(math.sqrt(p_b)))
    lo, hi = wilson_diff_ci(k_a, n, k_b, n)
    return {
        "n_common": n,
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "p_a": p_a, "p_b": p_b,
        "k_a": k_a, "k_b": k_b,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "exact": bool(exact),
        "cohen_h": float(cohen_h),
        "delta_p": p_a - p_b,
        "delta_p_ci_95": (lo, hi),
    }


def planned_contrasts_mcnemar(
    labels_to_rows: Mapping[str, list[EvalRecord]],
    contrasts: Sequence[tuple[str, str, str]],
    alpha: float = 0.05,
) -> dict:
    """Run McNemar for each (name, label_a, label_b) contrast and apply
    Holm step-down across the set.

    Contrasts whose labels are absent from labels_to_rows are skipped with
    a 'missing' entry — the Holm correction is computed over the labels
    that did run.
    """
    runnable = []
    missing = []
    for name, la, lb in contrasts:
        if la not in labels_to_rows or lb not in labels_to_rows:
            missing.append({"name": name, "label_a": la, "label_b": lb})
            continue
        result = pairwise_mcnemar(labels_to_rows[la], labels_to_rows[lb])
        runnable.append({"name": name, "label_a": la, "label_b": lb, **result})

    p_values = [r["p_value"] for r in runnable]
    holm = holm_correction(p_values, alpha=alpha)
    for r, adj in zip(runnable, holm["results"]):
        r["p_holm"] = adj["p_holm"]
        r["holm_threshold"] = adj["threshold"]
        r["reject_holm"] = adj["reject"]

    return {
        "alpha": alpha,
        "n_contrasts": len(runnable),
        "results": runnable,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Speedup distributional tests
# ---------------------------------------------------------------------------


def shapiro_speedup(
    labels_to_rows: Mapping[str, list[EvalRecord]],
    *,
    against: str = "inductor",
) -> dict:
    """Shapiro-Wilk on each label's per-winner speedup distribution.

    Used in §2.3 to justify the non-parametric path. n < 3 returns NaN; n
    > 5000 is downsampled by scipy automatically."""
    out: dict[str, dict] = {}
    for label, rows in labels_to_rows.items():
        sp = list(speedup_map(rows, against=against).values())
        if len(sp) < 3:
            out[label] = {"n": len(sp), "statistic": float("nan"), "p_value": float("nan")}
            continue
        stat, p = sps.shapiro(sp)
        out[label] = {"n": len(sp), "statistic": float(stat), "p_value": float(p)}
    return out


def friedman_speedup(
    labels_to_rows: Mapping[str, list[EvalRecord]],
    *,
    against: str = "inductor",
    min_n: int = 30,
) -> dict:
    """Friedman test on the K-way intersection of winning rows.

    The report's contingency in §1.8: if the intersection has n < min_n the
    test is reported as descriptive only and the Friedman statistic is not
    computed."""
    label_order, common, matrix = build_speedup_matrix(labels_to_rows, against=against)
    n = matrix.shape[0]
    if n < min_n:
        return {
            "labels": label_order, "n": n, "k": matrix.shape[1],
            "statistic": None, "df": matrix.shape[1] - 1,
            "p_value": None, "epsilon_squared": None,
            "skipped_reason": f"intersection n={n} < min_n={min_n}",
        }
    stat, p = sps.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    k = matrix.shape[1]
    eps2 = (stat - (k - 1)) / (n * (k - 1) - stat) if n * (k - 1) - stat > 0 else 0.0
    return {
        "labels": label_order, "n": n, "k": k,
        "statistic": float(stat), "df": k - 1, "p_value": float(p),
        "epsilon_squared": float(eps2),
        "skipped_reason": None,
    }


def dunns_speedup(
    labels_to_rows: Mapping[str, list[EvalRecord]],
    *,
    against: str = "inductor",
) -> dict:
    """Dunn's post-hoc with Holm correction over the full per-label speedup
    distributions (NOT the intersection — Dunn is an independent-samples
    rank test, paired structure is handled by Wilcoxon). Returns a square
    p-value table indexed by label."""
    import scikit_posthocs as sp_posthoc

    label_order = list(labels_to_rows.keys())
    arrays = [
        list(speedup_map(rows, against=against).values())
        for rows in labels_to_rows.values()
    ]
    if any(len(a) < 2 for a in arrays):
        return {"labels": label_order, "p_value_table": None,
                "skipped_reason": "at least one label has < 2 winners"}
    df = sp_posthoc.posthoc_dunn(arrays, p_adjust="holm")
    df.index = label_order
    df.columns = label_order
    return {
        "labels": label_order,
        "p_value_table": df.to_dict(),
        "skipped_reason": None,
    }


def pairwise_wilcoxon_log_speedup(
    rows_a: list[EvalRecord],
    rows_b: list[EvalRecord],
    *,
    against: str = "inductor",
) -> dict:
    """Paired Wilcoxon signed-rank on log(speedup_vs_<against>) over the
    rows where both sides produced a winning attempt.

    Returned `effect_r` is the rank-biserial r = 1 − 2·(W− / W) where W is
    the sum of ranks. Sign convention: positive means label_a has larger
    log-speedup than label_b.
    """
    sa = speedup_map(rows_a, against=against)
    sb = speedup_map(rows_b, against=against)
    common = sorted(set(sa) & set(sb))
    if len(common) < 5:
        return {"n_pairs": len(common), "skipped_reason": "n < 5"}
    a = np.array([math.log(sa[e]) for e in common])
    b = np.array([math.log(sb[e]) for e in common])
    diff = a - b
    if np.allclose(diff, 0):
        return {"n_pairs": len(common), "skipped_reason": "all pairs identical"}
    stat, p = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    ranks = sps.rankdata(np.abs(diff[diff != 0]))
    pos = float(np.sum(ranks[diff[diff != 0] > 0]))
    neg = float(np.sum(ranks[diff[diff != 0] < 0]))
    total = pos + neg
    effect_r = (pos - neg) / total if total > 0 else 0.0
    return {
        "n_pairs": len(common),
        "statistic": float(stat),
        "p_value": float(p),
        "effect_r": float(effect_r),
        "median_log_diff": float(np.median(diff)),
        "median_speedup_ratio": float(math.exp(np.median(diff))),
        "skipped_reason": None,
    }


def _success_set(rows: list[EvalRecord]) -> set[str]:
    return {r.example_id for r in rows if r.final_outcome in SUCCESS_OUTCOMES}
