"""Render single-label and paired-comparison reports as markdown / JSON."""

from __future__ import annotations

import json as _json
from io import StringIO

from tabulate import tabulate


def _fmt(x: float, pct: bool = False) -> str:
    if pct:
        return f"{100 * x:.1f}%"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return f"{x:.3e}"
    return f"{x:.3f}"


def _describe_row(label: str, d: dict) -> list:
    if d.get("n", 0) == 0:
        return [label, 0, "-", "-", "-", "-", "-", "-", "-"]
    return [
        label, d["n"],
        _fmt(d["min"]), _fmt(d["q1"]), _fmt(d["median"]),
        _fmt(d["mean"]), _fmt(d["q3"]), _fmt(d["max"]),
        _fmt(d["iqr"]),
    ]


def render_describe_markdown(label: str, sections: dict) -> str:
    out = StringIO()
    print(f"# Eval describe — `{label}`", file=out)
    print(file=out)
    print(f"Loaded **{sections['n_rows']}** rows.", file=out)
    print(file=out)

    print("## Outcome distribution", file=out)
    od = sections["outcome_distribution"]
    rows = []
    outcomes = sorted(od["total"].keys())
    tiers = sorted(od["by_tier"].keys())
    for o in outcomes:
        total = od["total"][o]
        per_tier = [od["by_tier"].get(t, {}).get(o, 0) for t in tiers]
        rows.append([o, total, f"{100 * total / od['n']:.1f}%", *per_tier])
    print(tabulate(rows, headers=["outcome", "total", "%", *tiers], tablefmt="pipe"), file=out)
    print(file=out)

    print("## Pass rate by tier (Wilson 95% CI)", file=out)
    pr = sections["pass_rate_by_tier"]
    rows = []
    for tier in ["__overall__", "easy", "medium", "hard"]:
        if tier not in pr: continue
        d = pr[tier]
        rows.append([
            tier, d["k"], d["n"], _fmt(d["rate"], pct=True),
            f"[{_fmt(d['wilson_lo'], pct=True)}, {_fmt(d['wilson_hi'], pct=True)}]",
        ])
    print(tabulate(rows, headers=["tier", "k", "n", "rate", "Wilson 95% CI"], tablefmt="pipe"), file=out)
    print(file=out)

    print("## Speedup on winning attempts", file=out)
    ss = sections["speedup_summary"]
    print(f"n_winners = {ss['n_winners']}", file=out)
    print(file=out)
    headers = ["metric", "n", "min", "q1", "median", "mean", "q3", "max", "iqr"]
    rows = [
        _describe_row("speedup_vs_eager", ss["vs_eager"]),
        _describe_row("speedup_vs_inductor", ss["vs_inductor"]),
        _describe_row("log(speedup_vs_eager)", ss["log_vs_eager"]),
        _describe_row("log(speedup_vs_inductor)", ss["log_vs_inductor"]),
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)

    print("## Per-method median timing (winning rows)", file=out)
    ti = sections["timing_iqr"]
    rows = [
        _describe_row("triton_ms", ti["triton_ms"]),
        _describe_row("eager_ms", ti["eager_ms"]),
        _describe_row("inductor_ms", ti["inductor_ms"]),
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)

    print("## Triton compile time (winning rows)", file=out)
    tc = sections["triton_compile_stats"]
    rows = [
        _describe_row("triton_compile_ms", tc["triton_compile_ms"]),
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"), file=out)
    print(file=out)

    print("## Judge classification distribution (all attempts)", file=out)
    jc = sections["judge_classification_dist"]
    total = sum(jc.values()) or 1
    rows = [
        [k, v, f"{100 * v / total:.1f}%"]
        for k, v in sorted(jc.items(), key=lambda kv: -kv[1])
    ]
    print(tabulate(rows, headers=["classification", "count", "%"], tablefmt="pipe"), file=out)
    print(file=out)

    return out.getvalue()


def render_describe_json(label: str, sections: dict) -> str:
    return _json.dumps({"model_label": label, **sections}, indent=2)
