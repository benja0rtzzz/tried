# Repeatability

This document is the contract for someone who pulls the repository and wants to reproduce the TRIED experiment from scratch. Read it end-to-end before running anything.

The exact software/hardware environment is recorded in [`docs/specs.yaml`](specs.yaml). Update that file whenever any version changes.

## What "reproducible" means here

**Bit-for-bit equality is not the goal and is not achievable.** Two stages of the pipeline call hosted LLMs (`gpt-5-3-codex` for skeleton synthesis and for the judge), and hosted LLMs cannot be pinned by the user. We aim for **qualitative reproducibility**: a teammate running this end-to-end on comparable hardware should land within sampling noise on:

- Per-`op_category` final-outcome counts (`compiled_correct`, `numeric_fail`, `runtime_fail`, `compile_fail`).
- Judge classification distribution.
- Eval headline numbers (success rate per tier, paired-test conclusions).

What will **not** match across runs:

- Row-by-row `dataset.jsonl` content. The order of rows, the per-attempt `triton_code` strings, and the judge `fix_suggestion` strings depend on stochastic hosted models and on rate-limit timing.
- Benchmark `*_samples_ms` arrays. Even on identical hardware, GPU scheduling jitter shifts the tail of the distribution.

If a future reader gets, say, "elementwise 82% pass, matmul 0.4% pass" within a few percentage points of our recorded numbers on the same Qwen2.5-Coder:14B, the experiment has reproduced. Beyond that is overclaiming.

## Step-by-step rerun procedure

This sequence assumes both machines from `docs/architecture.md` are available.

### 0. One-time setup

On both machines:

```bash
git clone <this-repo>
cd tried
uv sync                       # uses uv.lock — full transitive pin
.repos/clone.sh               # initial clone of the 16 reference repos
.repos/clone.sh --reset       # ensure each repo is at its pinned SHA
```

On the **MacBook (orchestrator)**:

```bash
brew install ollama codex     # versions in docs/specs.yaml
ollama pull qwen2.5-coder:14b
ollama show  qwen2.5-coder:14b   # check the digest matches docs/specs.yaml.orchestrator.generator.model_weights_digest
codex --version                  # check vs docs/specs.yaml.orchestrator.judge.cli_version
codex auth login              # complete the Codex CLI login flow
cp packages/orchestrator/.env.example packages/orchestrator/.env
# fill in VERIFICATION_SERVER_URL and VERIFICATION_API_KEY
```

On the **Lenovo (verification)**:

```bash
sudo apt install gcc g++ python3.12-dev   # required by Inductor at runtime
cp packages/verification/.env.example packages/verification/.env
# fill in VERIFICATION_API_KEY (same value as the orchestrator side)
```

Start the verification server (Lenovo):

```bash
TRIED_ROLE=verification CUDA_VISIBLE_DEVICES=0 \
  uv run uvicorn verification.server:app --host 0.0.0.0 --port 8000
```

### 1. Reuse the locked stages

Everything before `data/preflight_safe.jsonl` involves a hosted LLM (`gpt-5-3-codex` writes the PyTorch skeletons). To stay close to the recorded experiment, **start from the committed corpus_gen artifacts** rather than regenerating them:

```
data/corpus_gen/observations.jsonl   # discovery output
data/corpus_gen/specs.jsonl          # sampler output (seed=1)
data/corpus_gen/with_code.jsonl      # codex skeleton output
data/corpus_gen/rejected.jsonl       # codex rejections
data/preflight_safe.jsonl            # eager-vs-Inductor cleared
```

These files are tracked in git. A from-scratch regeneration is documented in point 4 below for the curious; it is not the recommended path.

### 2. Run the dataset agent loop

From the **MacBook**:

```bash
TRIED_ROLE=orchestrator uv run python -m orchestrator.train.dataset.main
```

The loop:

- Loads `data/preflight_safe.jsonl`, deduplicates one-row-per-source `example_id` (decision-log 2026-05-18).
- Skips any `example_id` already present in `data/dataset/dataset.jsonl` (resume).
- For each remaining row, runs up to 3 attempts: generator → `/compile` → `/run` → judge (on failure) → retry. Records every attempt.
- Stops cleanly with exit code 0 on a Codex CLI rate limit. Restart once the window clears.

Configuration knobs:

- `ALLOWED_OPS` (constant in `dataset/main.py`) restricts which categories a run loads.
- `TRIED_MAX_PER_CATEGORY` env var caps total unique source `example_id`s per category (default 180; rows already in `dataset.jsonl` count toward it).

### 3. Run the held-out eval

After the dataset loop is far enough along to fine-tune (or to compare against a baseline model):

```bash
TRIED_ROLE=orchestrator uv run python -m orchestrator.eval.eval_run.main
```

Eval is **single-attempt, no judge** by design. It writes to `eval/results/<model_label>/eval_rows.jsonl`. Eval rows store the full 10 correctness stats plus the raw 100-iter benchmark sample arrays needed for paired non-parametric tests; this is intentional and asymmetric with the dataset loop (see §"What is recorded" below).

### 4. Optional: regenerate corpus_gen from scratch

Skip unless you specifically want a fresh corpus. Expect a different `with_code.jsonl` than ours — `gpt-5-3-codex` is hosted.

```bash
# Discovery walks the cloned reference repos
TRIED_ROLE=orchestrator uv run python -m orchestrator.train.corpus_gen.discovery

# Stratified sampler — seed=1 is the locked seed; do not change it
TRIED_ROLE=orchestrator uv run python -m orchestrator.train.corpus_gen.sampler --seed 1

# Codex skeleton synthesis (stops cleanly on rate limit)
TRIED_ROLE=orchestrator uv run python -m orchestrator.train.corpus_gen.driver

# Eager-vs-Inductor preflight on the verification server
TRIED_ROLE=orchestrator uv run python -m orchestrator.train.dataset.preflight_driver
```

## What is and is not pinned

### Pinned (deterministic given the same inputs)

| Item                                 | Mechanism                                                                              |
|--------------------------------------|----------------------------------------------------------------------------------------|
| Python deps                          | `uv.lock` at repo root                                                                 |
| Reference source repos               | `.repos/COMMITS.txt` (16 repos by SHA)                                                 |
| Schema (dataset, eval)               | `packages/shared/src/shared/schema/`                                                    |
| Tolerance policy                     | `packages/shared/src/shared/verification/tolerance.py`                                  |
| Generator + judge + skeleton prompts | `packages/orchestrator/src/orchestrator/prompts/` and `corpus_gen/prompt.py`           |
| Sampler seed                         | `corpus_gen/sampler.py` (`--seed 1` default)                                           |
| Held-out eval set                    | `eval/holdout/synthetic_fusions.jsonl`                                                  |
| Per-`example_id` UUIDv5              | `corpus_gen/dedup.py::derive_example_id` (sha256 of `pytorch_code`)                    |
| `dataset_id` derivation              | `shared.models.derive_dataset_id` (UUIDv5 over source + shapes/dtypes/seed/policy)     |

### Not bit-for-bit, qualitatively stable

- **Local generator (Qwen2.5-Coder:14B via Ollama).** Temperature 0 and Ollama's default `seed=0`. With the same model digest on the same machine, generations are reproducible in practice. Across machines or after an Ollama version bump, expect token-level drift. Generator determinism in the dataset loop is "best effort"; the eval pipeline will tighten the seed contract separately when we get there.
- **Hardware-dependent execution.** Triton lowers to GPU-specific code; Inductor compiles C at runtime. Different RTX 4060 driver versions can shift the lower bits of correctness stats. Tolerance policies absorb this for the pass/fail decision.

### Not reproducible (hosted services)

- **Codex CLI profile `gpt-5-3-codex`.** Drives both `corpus_gen/codex.py` (skeleton synthesis) and `clients/judge_client.py` (judge classification + fix advice). The profile is a local Codex CLI config (declared in `~/.codex/config.toml`) that selects model `gpt-5.3-codex` via the `oca` provider with `model_reasoning_effort = "high"`; the full profile is recorded in `docs/specs.yaml`. The model itself is hosted, so its weights and serving stack can change without notice. Two runs from scratch will produce different `with_code.jsonl` and different `judge_fix_suggestion` strings. This is why point 1 recommends starting from the committed `data/corpus_gen/` artifacts.
- **Codex CLI version.** Recorded in `docs/specs.yaml`; subsequent CLI releases can change argument handling or output framing.

## What is recorded (and why) per stage

| Stage                        | Stores correctness stats?              | Stores benchmark?                                  |
|------------------------------|----------------------------------------|----------------------------------------------------|
| `/compile` response          | n/a                                    | n/a                                                |
| `/run` response              | All 10 stats (5 vs eager, 5 vs Inductor) | n/a                                                |
| `/benchmark` response        | n/a                                    | median + std + raw 100-iter samples per backend    |
| Dataset row (`dataset.jsonl`)| Compact: `status` + `tolerance_policy_used` only — the 10 stats are discarded | Never recorded (training is stats-free)            |
| Eval row (`eval_rows.jsonl`) | All 10 stats kept                      | Full `EvalBenchmark` (median + std + raw arrays)   |

This split is deliberate. Training only needs the pass/fail bit and which tolerance produced it; eval needs the full statistical material for Wilcoxon and bootstrap CIs. No code changes are required for this asymmetry — `dataset/agent.py::_build_attempt` already projects `RunResponse` down to `DatasetCorrectnessCheck`, and `eval_run/agent.py` already keeps the wide `CorrectnessCheck` and `EvalBenchmark`.

## Known gaps (record so the next person doesn't fall in)

These are tracked here rather than silently in code so a teammate auditing reproducibility sees them up front. They are out of scope for the current 8-week run but worth listing for a follow-up:

- **Dedup is byte-identical only.** `corpus_gen/dedup.py` checks for byte-identical `pytorch_code` collision against the locked eval set; `dataset/main.py::_load_corpus` collapses exact-source duplicates. There is no near-duplicate filter (no MinHash / SimHash / AST-edit-distance / embedding cosine). Two skeletons differing in a constant or whitespace survive as distinct rows. Adding proximity dedup is a future improvement.
- **NaN/Inf handling at preflight.** The harness substitutes a `_INF_SENTINEL` of `1e38` for non-finite stats and lets the row through if eager and Inductor still agree. A reference function that produces NaN can therefore pass preflight. Stricter rejection is a future improvement.
- **Generator seed is the Ollama default.** Adequate for dataset collection at temperature 0; the eval pipeline will fix an explicit seed when we tighten its contract.
- **Codex CLI auth state is not version-controlled.** A reproducer must run `codex auth login` interactively.

## Honesty disclaimer to include in the paper / report

> Two stages of the dataset pipeline call a hosted reasoning model (OpenAI `gpt-5-3-codex` via Codex CLI). The corresponding artifacts are committed to the repository at the SHAs recorded in `docs/specs.yaml`, but a from-scratch regeneration of the synthetic skeletons or the judge advice will not match bit-for-bit. We claim qualitative reproducibility — comparable per-category outcome distributions and eval headline numbers on equivalent hardware — and not bitwise determinism.
