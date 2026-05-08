# Corpus

Source of all PyTorch ops used in the experiment. Split into a training corpus (fed to the agent loop) and a held-out evaluation set (never touched programmatically).

## Training corpus (~200 examples, 50/50 split)

| Source | Count | Origin tag |
|---|---|---|
| Tritonbench `operators/` (extracted from `BenchmarkOperator` wrappers) | ~48 | `"tritonbench/<op_name>"` |
| Submodule reference implementations (Liger-Kernel, Flash-Attention, FBGEMM) | ~20-30 | `"tritonbench/<op_name>"` |
| Select ATen operator loader entries (compute-heavy subset) | ~20-30 | `"tritonbench/aten/<op>"` |
| `torch.fx`-traced subgraphs from diverse model architectures | ~100 | `"hf/<model>"` / `"timm/<model>"` |
| Curated standalone training patterns for underrepresented behavior buckets | as needed | `"curated/train/<name>"` |

The torch.fx tracer is being built as part of the pipeline anyway. Models must be architecturally diverse to avoid deduplication waste — transformers, CNNs, SSMs, and MLPs each contribute distinct compute patterns. Curated training rows are hand-written standalone PyTorch references used to fill behavior gaps found by the quota/fingerprint audit; they are training examples, not held-out eval examples.

## Eval corpus (437 synthetic fusions)

Synthetically constructed — not extracted from any existing repo. Not contaminated by training: a fusion like `layer_norm(gelu(x))` is a different generation task than generating `layer_norm` or `gelu` individually.

**Base ops:** tritonbench `operators/` set + a small set of plain PyTorch built-ins not present in training (`relu`, `sigmoid`, `tanh`, `exp`).

**Difficulty split (post-cleanup, 2026-05-09):**

| Tier | Count | Definition |
|---|---|---|
| Easy | 103 | 2-op elementwise fusions |
| Medium | 217 | 3-op fusions, or any fusion involving a reduction |
| Hard | 117 | 4+ ops, or complex memory patterns (attention-style, fused linear+norm+dropout) |

Difficulty tiers are for eval reporting only. All failures at any tier are recorded in the dataset. `origin` for all eval examples: `"synthetic/fusion"`.

The original plan was 130 examples; the spec sampler produced 443 rows in `eval/holdout/synthetic_fusions.jsonl`. After the vanilla qwen run, six rows were dropped (4 duplicate `example_id`s introduced during sampling, and 2 specs that never produced an `EvalRecord`) so that holdout and result files contain the same 437 unique IDs. See `docs/decision-log.md` 2026-05-09 entry.

## Extraction rules

Each training op must be extracted as a standalone, executable PyTorch function:
- Executable with `import torch` only (no model state, no module dependencies)
- Accepts tensor arguments matching `source.input_shapes` and `source.input_dtypes`
- Returns a single tensor

Store in `source.pytorch_code`. Set `source.origin` per the table above. All corpus rows have a non-null `origin`.

## What is NOT done

- The Triton implementations inside tritonbench (`operators/*/kernels/`) are never indexed into RAG and never appear in any prompt. They exist on the Lenovo disk only as part of the tritonbench install.
- Eval examples are never fed to the agent loop, RAG retrieval, or used as prompt examples.
