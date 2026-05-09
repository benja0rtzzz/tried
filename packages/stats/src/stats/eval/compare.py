"""CLI entry: `python -m stats.eval {describe,compare}`.

`describe <model_label>` runs the single-label descriptive report against
`eval/results/<model_label>/eval_rows.jsonl`. `compare` is wired as a stub
until the second eval run lands.
"""

from __future__ import annotations

import argparse
import sys

from stats.eval import descriptive
from stats.eval.load import load_label
from stats.eval.report import render_describe_json, render_describe_markdown


def _describe_sections(rows) -> dict:
    return {
        "n_rows": len(rows),
        "outcome_distribution":     descriptive.outcome_distribution(rows),
        "pass_rate_by_tier":        descriptive.pass_rate_by_tier(rows),
        "speedup_summary":          descriptive.speedup_summary(rows),
        "timing_iqr":               descriptive.timing_iqr(rows),
        "triton_compile_stats":     descriptive.triton_compile_stats(rows),
        "judge_classification_dist": descriptive.judge_classification_dist(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stats-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="paired comparison of two model_labels")
    p_compare.add_argument("label_a")
    p_compare.add_argument("label_b")
    p_compare.add_argument("--json", action="store_true")

    p_describe = sub.add_parser("describe", help="single-label descriptive report")
    p_describe.add_argument("label")
    p_describe.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "describe":
        rows = load_label(args.label)
        sections = _describe_sections(rows)
        if args.json:
            print(render_describe_json(args.label, sections))
        else:
            print(render_describe_markdown(args.label, sections))
        return 0

    if args.cmd == "compare":
        raise NotImplementedError(
            "compare: scaffold — implement once the fine-tuned eval lands"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
