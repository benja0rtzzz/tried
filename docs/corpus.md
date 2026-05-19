# Corpus

Source of all PyTorch ops used in the experiment. Split into a generated training corpus (fed to the dataset agent loop) and a held-out evaluation set (used only by the eval runner).

## Training Corpus

Current training rows are generated synthetic skeletons.

Pipeline:

1. `orchestrator.train.corpus_gen.discovery` scans cloned Triton-oriented repos and records structural observations.
2. `orchestrator.train.corpus_gen.sampler` samples `SkeletonSpec` rows over op category, shape rank, dtype mix, broadcast pattern, reduction axis, fusion shape, and memory pattern.
3. `orchestrator.train.corpus_gen.driver` asks Codex CLI to synthesize standalone PyTorch functions, validates them with AST checks, and deduplicates against the locked eval holdout.
4. `orchestrator.train.dataset.preflight_driver` runs eager-vs-Inductor preflight and writes accepted rows.
5. `orchestrator.train.dataset.main` consumes the preflight-safe file, rehydrates rows as training `CorpusRecord`s, and writes `data/dataset/dataset.jsonl` plus `data/dataset/errors.jsonl`.

The active training input is `data/preflight_safe.jsonl` by default (`TRIED_CORPUS_PATH` can override it). Persisted preflight-safe rows are intentionally slim: `example_id`, `op_category`, `pytorch_code`, `input_shapes`, `input_dtypes`, `rng_seed`, and `tolerance_policy`. When loaded for the dataset loop, they become `split="train"`, `origin="synthetic/skeleton"`, `difficulty=null`, and a derived `dataset_id` that uniquely identifies the exact task.

## Eval corpus (437 synthetic fusions)

Synthetically constructed — not extracted from any existing repo. Not contaminated by training: a fusion like `layer_norm(gelu(x))` is a different generation task than generating `layer_norm` or `gelu` individually.

**Base ops:** tritonbench `operators/` set + a small set of plain PyTorch built-ins not present in training (`relu`, `sigmoid`, `tanh`, `exp`).

**Difficulty split (post-cleanup, 2026-05-09):**

| Tier | Count | Definition |
|---|---|---|
| Easy | 103 | 2-op elementwise fusions |
| Medium | 217 | 3-op fusions, or any fusion involving a reduction |
| Hard | 117 | 4+ ops, or complex memory patterns (attention-style, fused linear+norm+dropout) |

Difficulty tiers are for eval reporting only. All failures at any tier are recorded in eval results. `origin` for all eval examples: `"synthetic/fusion"`.

The original plan was 130 examples; the spec sampler produced 443 rows in `eval/holdout/synthetic_fusions.jsonl`. After the vanilla qwen run, six rows were dropped (4 duplicate `example_id`s introduced during sampling, and 2 specs that never produced an `EvalRecord`) so that holdout and result files contain the same 437 unique IDs. See `docs/decision-log.md` 2026-05-09 entry.

## Extraction rules

Each training op must be a standalone, executable PyTorch function:
- Executable with `import torch` only (no model state, no module dependencies)
- Accepts tensor arguments matching `source.input_shapes` and `source.input_dtypes`
- Returns a single tensor

Store in `source.pytorch_code`. Generated training rows use `origin="synthetic/skeleton"` after the preflight-safe shape is rehydrated for the dataset loop. All corpus rows have a non-null `origin`.

## What is NOT done

- Triton implementations from source repos are never indexed into RAG and never appear in generator prompts.
- Eval examples are never used as training rows, RAG retrieval, or prompt examples. They are read only by `orchestrator.eval.eval_run`.
