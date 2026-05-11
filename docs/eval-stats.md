# Eval Stats Plan

Lists every statistical analysis we plan to run on eval data, mapped to the course-week material it derives from and split into two groups: analyses that **need eval-only schema fields** vs analyses that **work from fields already in `EvalRecord`**. The split is what determined the additions in `packages/shared/src/shared/schema/eval/eval_result.json`.

> Implementation lives in `packages/stats/eval/` (see "Package layout" at the bottom). Current implemented CLI: `stats-eval describe <model_label>` — reads `eval/results/<label>/eval_rows.jsonl` and emits a single-label descriptive report. `compare` is scaffolded only until the fine-tuned eval lands.

## Inputs

Current descriptive input is one JSONL file of per-row `EvalRecord`s:

- `eval/results/<model_label>/eval_rows.jsonl`

Each row carries the full `EvalSpec` (so `tier`, `form`, and ops are local) and one raw eval attempt (so `final_outcome`, `attempts[0].benchmark`, and `attempts[0].latency` are local). Future paired analyses will join two label folders by `example_id`.

## Course → analysis map

| Course week | Source material | Analyses produced |
|---|---|---|
| 1 — Fundamentos de Probabilidad | Sample spaces, conditional probability, Bayes, expected value | Outcome distribution, conditional `P(faster | tier)`, expected speedup |
| 2 — Distribuciones y Estadística Descriptiva | Bernoulli, Binomial, CLT, IQR, descriptive stats | Per-tier pass rate w/ Binomial CI, per-method timing distributions w/ Q1/median/Q3/IQR, outlier flagging, log-speedup stats, Triton compile-time stats |
| 3 — Pruebas de Hipótesis | H₀/H₁, p-values, paired t-test | Planned after second eval: **paired McNemar** on pass rate, **paired Wilcoxon signed-rank** on log-speedup, paired t-test on log Triton compile time |
| 4 — Power Analysis | Cohen's d/h, sample-size calc, post-hoc power | Planned after second eval: pre-experiment MDE at locked n=437, post-hoc power on observed effect, Cohen's h on pass-rate lift |

## Group A — analyses that need new eval-schema fields

These drove the additions in `schema/eval/eval_result.json` beyond what `dataset_record.json` carries. Training dataset rows intentionally do not include benchmark blocks.

| Analysis | Eval-schema field(s) consumed | Why it can't be done from `DatasetRow` alone |
|---|---|---|
| Per-method timing IQR / quartiles / outlier flagging (Week 2) | `attempts[*].benchmark.{triton,eager,inductor}_samples_ms` | `DatasetRow` records median + std only. Quartiles, IQR, and 1.5×IQR outlier detection require the raw 100-iter samples. |
| Bootstrap CI on median speedup (Week 2) | same `*_samples_ms` arrays | Bootstrap resamples from the per-iteration distribution. Median + std doesn't carry enough information. |
| Paired Wilcoxon signed-rank on per-iter speedup (Week 3) | same `*_samples_ms` arrays | The non-parametric paired test wants per-pair raw values, not summary statistics. |
| Joining two model conditions for paired tests (Week 3, planned) | parent folder `eval/results/<label>/` + `run_id` + `example_id` | The label is the directory each `EvalRecord` lives under (not an in-record field), `run_id` distinguishes re-runs, and `example_id` is the join key. |
| Per-row tier stratification (Weeks 2-4) | `spec.tier` (embedded) | `DatasetRow.source` doesn't carry difficulty. The locked synthetic-fusion eval is the only place tier exists. |

## Group B — analyses computable from fields already in `EvalRecord`

No new schema fields needed beyond the eval record itself.

| Analysis | Fields consumed |
|---|---|
| Outcome distribution per tier (Week 1) | `final_outcome` + `spec.tier` |
| Conditional probabilities, e.g. `P(faster_than_inductor | tier=Hard)` (Week 1) | `final_outcome` + `spec.tier` |
| Per-tier pass rate with Binomial / Wilson CI (Week 2) | `final_outcome` + `spec.tier` |
| Speedup mean / median / std per method (Week 2) | `attempts[winning].benchmark.speedup_vs_{eager,inductor}` |
| Log-speedup distribution stats (Week 2) | derived from speedup fields above |
| **Paired McNemar test on pass rate** (Week 3, planned) | `final_outcome` from both label folders, joined on `example_id` |
| Cohen's h on pass-rate lift (Week 4, planned) | derived from McNemar inputs |
| Post-hoc power on observed effect (Week 4, planned) | derived from observed effect size + n |
| Pre-experiment MDE at locked n (Week 4) | n only (437, or per-tier 103 / 217 / 117) |
| Triton compile-time descriptive stats (Week 2) | `attempts[winning].latency.compile_ms` |
| **Paired t-test on log Triton compile time** (Week 3, planned) | `attempts[winning].latency.compile_ms`, joined on `example_id` |
| Cross-tab outcome × form/category × tier (Week 1) | `final_outcome` + `spec.form` + `spec.tier`; category can be derived from `shared.eval.forms.FORMS` when needed |

## Sample-size note (recorded once, not recomputed per run)

Locked at **n = 437** for the overall paired McNemar, with the empirical tier split **103 easy / 217 medium / 117 hard** that came out of the spec sampler (post-cleanup, see decision log 2026-05-09). At α=0.05 and within-pair correlation ρ=0.7 (realistic when both models share most failure modes), n=437 has more than the original 300-row plan would have given on every effect size, so the +5/+8/+10pp power table from the original plan is a conservative lower bound.

Per-tier inference at the medium / hard counts is borderline rather than underpowered — tier results are still reported as **descriptive only** for v1, with the option to upgrade to per-tier McNemar in a follow-up if the headline test is significant.

## Package layout

```
packages/stats/src/stats/
├── dataset/                       # existing dataset stats, consolidated
│   └── summary.py                 # one main printer (replaces the spread-out scripts)
└── eval/
    ├── load.py                    # load EvalRecord JSONL, join two labels by example_id
    ├── descriptive.py             # Group A descriptive (timing IQR, compile times)
    │                              # + Group B descriptive (pass rate, conditionals)
    ├── hypothesis.py              # Scaffold: paired McNemar, paired Wilcoxon, paired t-test
    ├── power.py                   # Scaffold: Cohen's h, post-hoc power, pre-experiment MDE
    ├── compare.py                 # CLI entry: stats-eval describe <label>; compare scaffold
    └── report.py                  # Tabulate the markdown / JSON describe report
```

## What this doc does NOT cover

- Visualization / plot generation. Out of scope for v1; tables only.
- Cross-experiment comparisons (different prompts, different judges, etc.). Locked-experiment design — comparisons across experiments require a separate doc.
- Multiple-comparison corrections beyond the headline McNemar + Wilcoxon. If we report per-tier descriptive stats only, no Bonferroni / Holm needed.
