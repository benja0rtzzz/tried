# Eval Stats Plan

Lists every statistical analysis we plan to run on eval data, mapped to the course-week material it derives from and split into two groups: analyses that **need new eval-schema fields** vs analyses that **work from fields already in `DatasetRow` / `EvalRecord`**. The split is what determined the additions in `schema/eval/record.json`.

> Implementation lives in `packages/stats/eval/` (see "Package layout" at the bottom). One CLI entry point: `stats-eval compare <model_label_A> <model_label_B>` — reads `eval/results/<label>/eval_rows.jsonl` for each side, joins on `example_id`, emits a comparison report.

## Inputs

Two JSONL files, both per-row `EvalRecord`, joined by `example_id`:

- `eval/results/<model_label_A>/eval_rows.jsonl`
- `eval/results/<model_label_B>/eval_rows.jsonl`

Each row carries the full `EvalSpec` (so `tier`, `form`, `op_category` are local) and the full agent-loop result (so `final_outcome`, `attempts[*].benchmark`, `attempts[*].latency`, `baseline_compile` are local).

## Course → analysis map

| Course week | Source material | Analyses produced |
|---|---|---|
| 1 — Fundamentos de Probabilidad | Sample spaces, conditional probability, Bayes, expected value | Outcome distribution, conditional `P(faster | tier)`, expected speedup |
| 2 — Distribuciones y Estadística Descriptiva | Bernoulli, Binomial, CLT, IQR, descriptive stats | Per-tier pass rate w/ Binomial CI, per-method timing distributions w/ Q1/median/Q3/IQR, outlier flagging, log-speedup stats, cold compile-time stats |
| 3 — Pruebas de Hipótesis | H₀/H₁, p-values, paired t-test | **Paired McNemar** on pass rate (vanilla vs fine-tuned), **paired Wilcoxon signed-rank** on log-speedup, paired t-test on log Inductor cold-compile time |
| 4 — Power Analysis | Cohen's d/h, sample-size calc, post-hoc power | Pre-experiment MDE at locked n=300, post-hoc power on observed effect, Cohen's h on pass-rate lift |

## Group A — analyses that need new eval-schema fields

These drove the additions in `schema/eval/record.json` beyond what `dataset_record.json` already carries.

| Analysis | Eval-schema field(s) consumed | Why it can't be done from `DatasetRow` alone |
|---|---|---|
| Per-method timing IQR / quartiles / outlier flagging (Week 2) | `attempts[*].benchmark.{triton,eager,inductor}_samples_ms` | `DatasetRow` records median + std only. Quartiles, IQR, and 1.5×IQR outlier detection require the raw 100-iter samples. |
| Bootstrap CI on median speedup (Week 2) | same `*_samples_ms` arrays | Bootstrap resamples from the per-iteration distribution. Median + std doesn't carry enough information. |
| Paired Wilcoxon signed-rank on per-iter speedup (Week 3) | same `*_samples_ms` arrays | The non-parametric paired test wants per-pair raw values, not summary statistics. |
| Cold compile-time descriptive stats (Week 2) | `baseline_compile.{eager,inductor}_first_call_ms` | `DatasetRow.attempts[*].latency.compile_ms` is **Triton compile**, not the eager / Inductor baselines. The "Inductor compiles in 60-120 s" headline number isn't recoverable from the dataset schema. |
| Compile-vs-runtime breakeven analysis (Week 2) | `baseline_compile.*` + `attempts[*].latency.compile_ms` + `attempts[*].benchmark.*_ms` | Combines the new compile-time fields with existing latency / benchmark fields. |
| Joining two model conditions for paired tests (Week 3) | `model_label`, `run_id`, `example_id` | `DatasetRow` has no concept of a "condition". `model_label` and `run_id` identify which run each `EvalRecord` came from; `example_id` is the join key. |
| Per-row tier stratification (Weeks 2-4) | `spec.tier` (embedded) | `DatasetRow.source` doesn't carry difficulty. The locked synthetic-fusion eval is the only place tier exists. |

## Group B — analyses computable from fields already in `DatasetRow` / `EvalRecord`

No new schema fields needed. Everything below uses existing dataset fields plus the eval-schema fields that just identify the row (`example_id`, `model_label`, `spec.tier`).

| Analysis | Fields consumed |
|---|---|
| Outcome distribution per tier (Week 1) | `final_outcome` + `spec.tier` |
| Conditional probabilities, e.g. `P(faster_than_inductor | tier=Hard)` (Week 1) | `final_outcome` + `spec.tier` |
| Per-tier pass rate with Binomial / Wilson CI (Week 2) | `final_outcome` + `spec.tier` |
| Judge-classification distribution (Week 2) | `attempts[*].judge_classification` |
| Speedup mean / median / std per method (Week 2) | `attempts[winning].benchmark.speedup_vs_{eager,inductor}` |
| Log-speedup distribution stats (Week 2) | derived from speedup fields above |
| **Paired McNemar test on pass rate** (Week 3) | `final_outcome` from both `model_label`s, joined on `example_id` |
| Cohen's h on pass-rate lift (Week 4) | derived from McNemar inputs |
| Post-hoc power on observed effect (Week 4) | derived from observed effect size + n |
| Pre-experiment MDE at locked n (Week 4) | n only (300, or per-tier 105 / 115 / 80) |
| Paired t-test on Triton compile time (Week 3) | `attempts[winning].latency.compile_ms` (already in `DatasetRow`) — but the Inductor / eager baseline equivalent needs Group A fields |
| Cross-tab outcome × `op_category` × tier (Week 1) | `final_outcome` + `op_category` + `spec.tier` |

## Sample-size note (recorded once, not recomputed per run)

Locked at **n = 300** for the overall paired McNemar, with proportional tier split **105 / 115 / 80**. At α=0.05 and assuming within-pair correlation ρ=0.7 (realistic when both models share most failure modes), this n detects:

- +5pp lift: ~61% power (inconclusive results expected)
- +8pp lift: ~94-98% power
- +10pp lift: ~99% power

Per-tier inference is underpowered — tier results are reported as **descriptive only**, never as standalone significance tests. This is a deliberate choice (see decision log entry when n was locked).

## Package layout (preview of task #5)

```
packages/stats/src/stats/
├── dataset/                       # existing dataset stats, consolidated
│   └── summary.py                 # one main printer (replaces the spread-out scripts)
└── eval/
    ├── load.py                    # load EvalRecord JSONL, join two labels by example_id
    ├── descriptive.py             # Group A descriptive (timing IQR, compile times)
    │                              # + Group B descriptive (pass rate, conditionals)
    ├── hypothesis.py              # Paired McNemar, paired Wilcoxon, paired t-test
    ├── power.py                   # Cohen's h, post-hoc power, pre-experiment MDE
    ├── compare.py                 # CLI entry: stats-eval compare <A> <B>
    └── report.py                  # Tabulate the markdown / JSON comparison report
```

## What this doc does NOT cover

- Visualization / plot generation. Out of scope for v1; tables only.
- Cross-experiment comparisons (different prompts, different judges, etc.). Locked-experiment design — comparisons across experiments require a separate doc.
- Multiple-comparison corrections beyond the headline McNemar + Wilcoxon. If we report per-tier descriptive stats only, no Bonferroni / Holm needed.
