"""CLI entry: `python -m stats.eval {describe,compare}`.

`describe <label>` prints the single-label descriptive report against
`eval/results/<label>/eval_rows.jsonl`.

`compare <label_a> <label_b> [<label_c> ...]` runs the full §2.3 protocol:
  - Per-label describe section
  - Shapiro-Wilk per condition on speedup
  - Cochran's Q omnibus on pass rate
  - Pairwise McNemar (Holm-corrected) on the supplied contrast set
  - Friedman + Dunn's on speedup intersection
  - Pairwise Wilcoxon on log-speedup over winners
  - MDE/power table at locked n

Contrasts default to "first label vs every other label". The exact
contrasts from the report (e.g. SFT+DPO vs SFT) can be passed with
`--contrasts a:b,c:d`. Contrast names are auto-generated from the labels
(`<label_a>_vs_<label_b>`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from stats.eval import descriptive, hypothesis, power
from stats.eval.load import load_label
from stats.eval.report import (
    render_compare_json,
    render_compare_markdown,
    render_describe_json,
    render_describe_markdown,
)


DEFAULT_RESULTS_ROOT = Path("eval/results")
DEFAULT_MDE_TIERS: dict[str, int] = {"overall": 437, "easy": 103, "medium": 217, "hard": 117}


def _describe_sections(rows) -> dict:
    return {
        "n_rows": len(rows),
        "outcome_distribution": descriptive.outcome_distribution(rows),
        "pass_rate_by_tier":    descriptive.pass_rate_by_tier(rows),
        "speedup_summary":      descriptive.speedup_summary(rows),
        "timing_iqr":           descriptive.timing_iqr(rows),
        "triton_compile_stats": descriptive.triton_compile_stats(rows),
    }


def _parse_contrasts(arg: str | None, labels: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return [(name, label_a, label_b), ...]. If arg is None, default to
    label[0] vs each subsequent label."""
    if arg is None:
        baseline = labels[0]
        return [(f"{label}_vs_{baseline}", label, baseline) for label in labels[1:]]
    out = []
    for pair in arg.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise SystemExit(f"--contrasts entry {pair!r} must be 'label_a:label_b'")
        a, b = pair.split(":", 1)
        a, b = a.strip(), b.strip()
        if a not in labels or b not in labels:
            raise SystemExit(
                f"--contrasts entry {pair!r} references unknown labels; available: {labels}"
            )
        out.append((f"{a}_vs_{b}", a, b))
    return out


def _compare_sections(
    labels_to_rows: dict[str, list],
    contrasts: list[tuple[str, str, str]],
    alpha: float,
    against: str,
) -> dict:
    describe = {label: _describe_sections(rows) for label, rows in labels_to_rows.items()}
    return {
        "labels": list(labels_to_rows.keys()),
        "against": against,
        "alpha": alpha,
        "describe": describe,
        "shapiro": hypothesis.shapiro_speedup(labels_to_rows, against=against),
        "cochran_q": hypothesis.cochran_q_pass_rate(labels_to_rows),
        "mcnemar": hypothesis.planned_contrasts_mcnemar(
            labels_to_rows, contrasts, alpha=alpha
        ),
        "friedman": hypothesis.friedman_speedup(labels_to_rows, against=against),
        "dunns": hypothesis.dunns_speedup(labels_to_rows, against=against),
        "wilcoxon": [
            {
                "name": name,
                "label_a": la,
                "label_b": lb,
                **hypothesis.pairwise_wilcoxon_log_speedup(
                    labels_to_rows[la], labels_to_rows[lb], against=against,
                ),
            }
            for name, la, lb in contrasts
            if la in labels_to_rows and lb in labels_to_rows
        ],
        "mde": {
            tier: power.mde_table(n, alpha=alpha)
            for tier, n in DEFAULT_MDE_TIERS.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stats-eval")
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing eval/results/<label>/eval_rows.jsonl",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_describe = sub.add_parser("describe", help="single-label descriptive report")
    p_describe.add_argument("label")
    p_describe.add_argument("--json", action="store_true")

    p_compare = sub.add_parser("compare", help="paired comparison across N labels")
    p_compare.add_argument("labels", nargs="+", help="≥2 model labels; first is the baseline")
    p_compare.add_argument(
        "--contrasts", default=None,
        help="Comma-separated label_a:label_b pairs (e.g. 'sft:vanilla,sft_dpo:sft'). "
             "Defaults to 'label_2..N vs label_1'.",
    )
    p_compare.add_argument(
        "--against", choices=["eager", "inductor"], default="inductor",
        help="Reference for the speedup tests (default: inductor).",
    )
    p_compare.add_argument(
        "--alpha", type=float, default=0.05,
        help="Family-wise α applied to Holm step-down (default: 0.05).",
    )
    p_compare.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "describe":
        rows = load_label(args.label, args.results_root)
        sections = _describe_sections(rows)
        if args.json:
            print(render_describe_json(args.label, sections))
        else:
            print(render_describe_markdown(args.label, sections))
        return 0

    if args.cmd == "compare":
        if len(args.labels) < 2:
            raise SystemExit("compare requires ≥2 labels")
        labels_to_rows = {
            label: load_label(label, args.results_root) for label in args.labels
        }
        contrasts = _parse_contrasts(args.contrasts, args.labels)
        sections = _compare_sections(
            labels_to_rows, contrasts, alpha=args.alpha, against=args.against,
        )
        if args.json:
            print(render_compare_json(sections))
        else:
            print(render_compare_markdown(sections))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
