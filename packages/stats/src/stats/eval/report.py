"""Render eval describe / compare reports as markdown or JSON."""

from __future__ import annotations

import json as _json
import math
from io import StringIO

from tabulate import tabulate


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if pct:
        return f"{100 * x:.1f}%"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.3e}"
    return f"{x:.3f}"


def _fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def _describe_row(label: str, d: dict) -> list:
    if d.get("n", 0) == 0:
        return [label, 0, "-", "-", "-", "-", "-", "-", "-"]
    return [
        label, d["n"],
        _fmt(d["min"]), _fmt(d["q1"]), _fmt(d["median"]),
        _fmt(d["mean"]), _fmt(d["q3"]), _fmt(d["max"]),
        _fmt(d["iqr"]),
    ]


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def render_describe_markdown(label: str, sections: dict) -> str:
    out = StringIO()
    print(f"# Eval describe — `{label}`", file=out)
    print(f"\nLoaded **{sections['n_rows']}** rows.\n", file=out)

    _render_outcome_distribution(out, sections["outcome_distribution"])
    _render_pass_rate(out, sections["pass_rate_by_tier"])
    _render_speedup_summary(out, sections["speedup_summary"])
    _render_timing_iqr(out, sections["timing_iqr"])
    _render_triton_compile(out, sections["triton_compile_stats"])
    return out.getvalue()


def render_describe_json(label: str, sections: dict) -> str:
    return _json.dumps({"model_label": label, **sections}, indent=2, default=str)


_TIER_ORDER = ["easy", "medium", "hard"]


def _ordered_tiers(present: list[str]) -> list[str]:
    seen = set(present)
    return [t for t in _TIER_ORDER if t in seen] + sorted(seen - set(_TIER_ORDER))


def _render_outcome_distribution(out, od: dict) -> None:
    print("## Outcome distribution (raw)", file=out)
    outcomes = sorted(od["total"].keys())
    tiers = _ordered_tiers(list(od["by_tier"].keys()))
    rows = []
    for o in outcomes:
        total = od["total"][o]
        per_tier = [od["by_tier"].get(t, {}).get(o, 0) for t in tiers]
        rows.append([o, total, f"{100 * total / od['n']:.1f}%", *per_tier])
    print(tabulate(rows, headers=["outcome", "total", "%", *tiers], tablefmt="pipe"), file=out)
    print(file=out)

    print("## Outcome distribution (report 5-bucket §2.1)", file=out)
    rep = od["report_total"]
    rep_tiers = _ordered_tiers(list(od["report_by_tier"].keys()))
    rows = []
    for bucket in [
        "correct_faster", "correct_parity", "correct_slow",
        "correctness_failed", "compile_or_runtime_fail",
    ]:
        total = rep.get(bucket, 0)
        per_tier = [od["report_by_tier"].get(t, {}).get(bucket, 0) for t in rep_tiers]
        rows.append([bucket, total, f"{100 * total / od['n']:.1f}%", *per_tier])
    print(tabulate(rows, headers=["bucket", "total", "%", *rep_tiers], tablefmt="pipe"), file=out)
    print(file=out)


def _render_pass_rate(out, pr: dict) -> None:
    print("## Pass rate by tier (Wilson 95% CI)", file=out)
    rows = []
    for tier in ["__overall__", "easy", "medium", "hard"]:
        if tier not in pr:
            continue
        d = pr[tier]
        rows.append([
            tier, d["k"], d["n"], _fmt(d["rate"], pct=True),
            f"[{_fmt(d['wilson_lo'], pct=True)}, {_fmt(d['wilson_hi'], pct=True)}]",
        ])
    print(tabulate(rows, headers=["tier", "k", "n", "rate", "Wilson 95% CI"], tablefmt="pipe"), file=out)
    print(file=out)


def _render_speedup_summary(out, ss: dict) -> None:
    print("## Speedup on winning attempts", file=out)
    print(f"n_winners = {ss['n_winners']}\n", file=out)
    headers = ["metric", "n", "min", "q1", "median", "mean", "q3", "max", "iqr"]
    rows = [
        _describe_row("speedup_vs_eager", ss["vs_eager"]),
        _describe_row("speedup_vs_inductor", ss["vs_inductor"]),
        _describe_row("log(speedup_vs_eager)", ss["log_vs_eager"]),
        _describe_row("log(speedup_vs_inductor)", ss["log_vs_inductor"]),
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)


def _render_timing_iqr(out, ti: dict) -> None:
    print("## Per-method median timing (winning rows)", file=out)
    headers = ["metric", "n", "min", "q1", "median", "mean", "q3", "max", "iqr"]
    rows = [
        _describe_row("triton_ms", ti["triton_ms"]),
        _describe_row("eager_ms", ti["eager_ms"]),
        _describe_row("inductor_ms", ti["inductor_ms"]),
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)


def _render_triton_compile(out, tc: dict) -> None:
    print("## Triton static validation latency (winning rows)", file=out)
    headers = ["metric", "n", "min", "q1", "median", "mean", "q3", "max", "iqr"]
    rows = [_describe_row("triton_compile_ms", tc["triton_compile_ms"])]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def render_compare_markdown(sections: dict) -> str:
    out = StringIO()
    labels = sections["labels"]
    against = sections["against"]
    alpha = sections["alpha"]
    print(f"# Eval compare — {', '.join(f'`{l}`' for l in labels)}", file=out)
    print(f"\nSpeedup reference: **{against}**. α = {alpha}.\n", file=out)

    _render_per_label_summary(out, sections["describe"], against=against)
    _render_shapiro(out, sections["shapiro"])
    _render_cochran_q(out, sections["cochran_q"])
    _render_mcnemar(out, sections["mcnemar"])
    _render_friedman(out, sections["friedman"])
    _render_dunns(out, sections["dunns"])
    _render_wilcoxon(out, sections["wilcoxon"], against=against)
    _render_mde(out, sections["mde"])
    return out.getvalue()


def render_compare_json(sections: dict) -> str:
    return _json.dumps(sections, indent=2, default=str)


def _render_per_label_summary(out, describe: dict, against: str = "inductor") -> None:
    print("## Per-label headline", file=out)
    summary_key = f"vs_{against}"
    rows = []
    for label, d in describe.items():
        pr = d["pass_rate_by_tier"]["__overall__"]
        ss = d["speedup_summary"]
        med = ss[summary_key].get("median", None)
        rows.append([
            label,
            pr["n"], pr["k"], _fmt(pr["rate"], pct=True),
            f"[{_fmt(pr['wilson_lo'], pct=True)}, {_fmt(pr['wilson_hi'], pct=True)}]",
            ss["n_winners"],
            _fmt(med),
        ])
    print(tabulate(
        rows,
        headers=["label", "n", "k_passed", "pass_rate", "Wilson 95% CI",
                 "n_winners", f"speedup_vs_{against} median"],
        tablefmt="pipe",
    ), file=out)
    print(file=out)


def _render_shapiro(out, sh: dict) -> None:
    print("## Shapiro-Wilk on per-row speedup", file=out)
    rows = [
        [label, d["n"], _fmt(d["statistic"]), _fmt_p(d["p_value"]),
         "non-normal" if (d["p_value"] is not None
                          and not math.isnan(d["p_value"])
                          and d["p_value"] < 0.05) else "—"]
        for label, d in sh.items()
    ]
    print(tabulate(rows, headers=["label", "n", "W", "p", "verdict"], tablefmt="pipe"), file=out)
    print(file=out)


def _render_cochran_q(out, q: dict) -> None:
    print("## Cochran's Q (omnibus on pass rate)", file=out)
    if q.get("skipped_reason"):
        print(f"_skipped: {q['skipped_reason']}_  (n={q['n']}, k={q['k']})\n", file=out)
        return
    print(
        f"Q({q['df']}) = {_fmt(q['statistic'])}, "
        f"p = {_fmt_p(q['p_value'])}, "
        f"ε² = {_fmt(q['epsilon_squared'])}, "
        f"n_common = {q['n']}, k = {q['k']}\n",
        file=out,
    )


def _render_mcnemar(out, mc: dict) -> None:
    print(f"## Pairwise McNemar (Holm, α = {mc['alpha']})", file=out)
    if not mc["results"]:
        print("_no runnable contrasts_\n", file=out)
    rows = []
    for r in mc["results"]:
        if r.get("n_common", 0) == 0:
            rows.append([r["name"], "—", "—", "—", "—", "—", "—", "—", "—", "—"])
            continue
        lo, hi = r["delta_p_ci_95"]
        rows.append([
            r["name"],
            r["n_common"],
            f"{r['k_a']}/{r['k_b']}",
            f"{r['n10']}/{r['n01']}",
            _fmt(r["statistic"]),
            _fmt_p(r["p_value"]),
            _fmt_p(r.get("p_holm")),
            _fmt(r.get("cohen_h")),
            f"[{_fmt(lo)}, {_fmt(hi)}]",
            "reject" if r.get("reject_holm") else "—",
        ])
    print(tabulate(
        rows,
        headers=["contrast", "n", "k_a/k_b", "b/c", "stat", "p", "p_holm",
                 "h", "Δp 95% CI", "decision"],
        tablefmt="pipe",
    ), file=out)
    if mc["missing"]:
        print("\n_Missing labels for these contrasts (omitted from Holm):_", file=out)
        for m in mc["missing"]:
            print(f"- {m['name']}  ({m['label_a']} ⟂ {m['label_b']})", file=out)
    print(file=out)


def _render_friedman(out, fr: dict) -> None:
    print("## Friedman on speedup (intersection of winners)", file=out)
    if fr["skipped_reason"]:
        print(f"_skipped: {fr['skipped_reason']}_  (n={fr['n']}, k={fr['k']})\n", file=out)
        return
    print(
        f"χ²({fr['df']}) = {_fmt(fr['statistic'])}, "
        f"p = {_fmt_p(fr['p_value'])}, "
        f"ε² = {_fmt(fr['epsilon_squared'])}, "
        f"n_common = {fr['n']}\n",
        file=out,
    )


def _render_dunns(out, dn: dict) -> None:
    print("## Dunn's post-hoc on speedup (Holm-adjusted)", file=out)
    if dn["skipped_reason"]:
        print(f"_skipped: {dn['skipped_reason']}_\n", file=out)
        return
    labels = dn["labels"]
    table = dn["p_value_table"]
    headers = [""] + labels
    rows = [[la] + [_fmt_p(table[la][lb]) for lb in labels] for la in labels]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)


def _render_wilcoxon(out, wx: list[dict], against: str) -> None:
    print(f"## Pairwise Wilcoxon on log(speedup_vs_{against}) (winners only)", file=out)
    rows = []
    for r in wx:
        if r.get("skipped_reason"):
            rows.append([r["name"], r.get("n_pairs", 0), "—", "—", "—", "—", r["skipped_reason"]])
            continue
        rows.append([
            r["name"], r["n_pairs"], _fmt(r["statistic"]),
            _fmt_p(r["p_value"]), _fmt(r["effect_r"]),
            _fmt(r["median_speedup_ratio"]),
            "—",
        ])
    print(tabulate(
        rows,
        headers=["contrast", "n_pairs", "W", "p", "rank_biserial_r",
                 "median ratio a/b", "notes"],
        tablefmt="pipe",
    ), file=out)
    print(file=out)


def _render_mde(out, mde: dict) -> None:
    print("## MDE table (Cohen's h required at α=0.05, power=0.80)", file=out)
    headers = ["tier", "n", "MDE h"]
    # Pull the per-baseline Δp into separate columns.
    baselines = [r["baseline_p"] for r in next(iter(mde.values()))["rows"]]
    headers += [f"Δp @ p={p:.2f}" for p in baselines]
    rows = []
    for tier, t in mde.items():
        row = [tier, t["n"], _fmt(t["mde_h"])]
        for r in t["rows"]:
            row.append(_fmt(r["mde_delta_p"]))
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)
