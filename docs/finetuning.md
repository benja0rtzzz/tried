# Fine-Tuning Plan

Fine-tuning Qwen2.5-Coder-14B and measure whether it improves over the base model on the locked holdout.

## Goal and non-goals

**Goal.** Fine-tune the same base model that produced the dataset (Qwen2.5-Coder-14B, the source of Ollama's `qwen2.5-coder:14b`) on TRIED-collected attempt trajectories, and report whether eval pass rate on `eval/holdout/synthetic_fusions.jsonl` improves over a same-precision baseline.

**Non-goals.**
- We are not benchmarking the fine-tuned model against `torch.compile` head-to-head as a headline claim. Given that most of the results where wrong, speedup remains a secondary eval metric. The main thing to check is whether fine tuning can help improve the compilation ratio.
- We are not deploying the fine-tuned model through Ollama in this round. Inference for both baseline and fine-tuned eval runs goes through MLX directly.

## Base model and provenance

- **Source.** `mlx-community/Qwen2.5-Coder-14B-4bit` — Apple Silicon MLX-compatible, fits in 48 GB unified RAM alongside training state.
- **Relationship to the dataset.** This is the same parameter set Ollama quantized to Q4_K_M to serve `qwen2.5-coder:14b`. The dataset rows are precision-agnostic `(prompt, response)` records.

The Ollama Q4 vanilla baseline (`eval/results/qwen2.5-coder:14b-vanilla/`) was deleted on 2026-05-20. A new MLX-4bit baseline replaces it; all 4 local Qwen conditions run at 4-bit to satisfy the precision consistency rule.

## Method: QLoRA, rank 64

| Knob | Value | Reason |
|---|---|---|
| Base precision | 4-bit (frozen) | Frees memory for higher rank, longer sequences, larger batches on 48 GB unified RAM. |
| Adapter rank | 64 | Capacity to learn Triton API patterns; rank-8 is too small for a new skill. |
| Adapter scale | 2.0 (= alpha/rank with alpha=128, rank=64) | Standard LoRA scaling. mlx-lm-lora's `lora_parameters.scale` is the alpha/rank ratio, not alpha itself. |
| Adapter dropout | 0.1 | Conservative for ~776 SFT examples. Drop to 0.05 if/when SFT-positive pool reaches 1000. |
| Gradient checkpointing | on | Trades ~25% compute for ~3–4× activation memory savings. |
| Adapters trained in | bf16 | Apple Silicon native format; better numerical stability than fp16 at no memory cost. |

Full-parameter fine-tuning is out (doesn't fit in 48 GB). Plain LoRA-bf16 is a fallback if QLoRA training is unstable in MLX; expected gap is <1pp on coding tasks at equal rank, and QLoRA's saved memory buys more useful capacity (rank, batch, context).

## Data builders

Both builders read `data/dataset/dataset.jsonl` and render the prompt with the **fixed generator template** at `packages/orchestrator/src/orchestrator/prompts/generator/`. Training prompts MUST match the inference prompts character-for-character — if they drift, the fine-tuned model behaves worse, not better.

**Chat-template parity (locked).** Both training and inference wrap the rendered system/user content through the HF tokenizer's `apply_chat_template` for `Qwen/Qwen2.5-Coder-14B-Instruct` (ChatML-style `<|im_start|>…<|im_end|>`). This is the same template Ollama applied internally during dataset collection. Training data is emitted as `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}` for `mlx_lm.lora`'s chat format; the MLX inference client calls `apply_chat_template([system, user], add_generation_prompt=True)` before `mlx_lm.generate`. No hand-rolled wrapping anywhere — the tokenizer is the single source of truth.

### SFT data (`packages/orchestrator/src/orchestrator/improvement/data/sft_builder.py`)

- **Filter:** rows where any attempt has `correctness.status == "passed"`.
- **Prompt:** rendered with empty `prior_attempt_section` (the attempt-0 prompt), regardless of which attempt index actually won. This trains the model to get the answer right on the first try, which is the metric the locked holdout measures (one attempt per row, no retry).
- **Response:** `attempts[i].triton_code` for the first `i` where the correctness check passed.
- **Output:** JSONL with `{"messages": [system, user, assistant]}` (the `mlx_lm.lora` chat format). The tokenizer applies the chat template at load time.
- **Expected count:** 776 today (empirically counted from `data/dataset/dataset.jsonl`, 2500 rows / 6899 attempts).

### DPO data (`packages/orchestrator/src/orchestrator/improvement/data/dpo_builder.py`)

- **Filter:** rows that have both a passing attempt AND at least one failing attempt.
- **Prompt:** the attempt-0 prompt (same rendering as SFT). Both `chosen` and `rejected` are framed as competing responses to the same first-attempt prompt.
- **Chosen:** the winning Triton code (same as SFT).
- **Rejected:** the first failing attempt's `triton_code`. If multiple failures exist, take the earliest (deterministic, easy to reason about). One rejected per chosen.
- **Output:** JSONL with `{"prompt": <str>, "chosen": <str>, "rejected": <str>}`.
- **Expected count:** 628 today (empirically counted).

**Rejected-attempt mix (kept as-is).** Across the 628 pairs, the earliest non-passing attempt is `runtime_fail` 85% (compiles but crashes at launch — the dominant Triton-API-error pattern), `numeric_fail` 15% (compiles and runs but fails tolerance), and `compile_fail` 0% (none observed). The mix is preserved without filtering: runtime failures are exactly the capability gap DPO is meant to address.

Cross-row pair mining from the 1724 pure-failure rows is deferred. The 628 same-row pairs are higher quality (same PyTorch task, same shapes, same dtype) and are enough to start.

## Training recipe: SFT then DPO

The standard recipe. SFT teaches the model what correct Triton looks like; DPO refines preferences against same-row failures using the SFT checkpoint as the reference policy. DPO-only from the base model is known to be unstable when the base has a real capability gap (which Qwen does for Triton — see the 64.6% `triton_api_error` rate).

### SFT

Starting hyperparameters (to be tuned against training loss curves on a held-out slice):

| Hyperparameter | Starting value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | cosine, 5% warmup |
| Epochs | 2 (watch val loss; add epoch 3 only if val loss still declining at epoch 2 end) |
| Effective batch | 16 (micro-batch ≈ 2–4 with gradient accumulation) |
| Max sequence length | 2048 (empirical max across 776 SFT-positive rows is 1218 tokens, p99 1073; 2048 gives ~1.7× headroom, zero truncations) |
| Loss masking | response-only (prompt tokens masked out of the loss) |
| Validation split | 5% stratified by `op_category`, carved once and persisted to `data/improvement/val_split.json`; the SAME split is used for SFT and for DPO chosen/rejected filtering so the validation rows never leak into training |

### DPO

Started from the SFT checkpoint as both policy and reference.

| Hyperparameter | Starting value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-7 (much lower than SFT — DPO destabilizes easily) |
| LR schedule | linear, 10% warmup |
| Epochs | 1–2 |
| Effective batch | 8 |
| Beta (KL strength) | 0.1 |
| Max sequence length | 4096 (prompt + chosen, prompt + rejected; empirical max across 1256 sides is 2506 tokens, p99 993 — 2048 would clip 5 sides / ~0.8% of pairs, 4096 covers all observed rows at near-zero cost since `mlx_lm.lora` pads to the longest in batch, not to the ceiling) |

After each stage: `mlx_lm.merge` produces a transient fp16 checkpoint, then `mlx_lm.convert -q --q-bits 4` re-quantizes it to 4-bit. Only the final 4-bit checkpoint is kept (`sft-merged-4bit/`, `sft-dpo-merged-4bit/`). The intermediate fp16 artifact can be discarded once conversion succeeds.

## Evaluation methodology

Both baseline and fine-tuned runs go through MLX. The existing eval runner at `packages/orchestrator/src/orchestrator/eval/eval_run/` stays — only the generator client swaps.

```
A. Download mlx-community/Qwen2.5-Coder-14B-4bit from HuggingFace   (one-time)
B. Run holdout eval through MLX on the base 4-bit checkpoint
       → eval/results/qwen2.5-coder-14b-mlx-4bit-baseline/
C. SFT → merge → re-quantize → improvement/checkpoints/sft-merged-4bit/
D. Run holdout eval through MLX on the SFT-only 4-bit checkpoint
       → eval/results/qwen2.5-coder-14b-mlx-4bit-sft/
E. From the SFT adapters: DPO → merge → re-quantize
       → improvement/checkpoints/sft-dpo-merged-4bit/
F. Run holdout eval through MLX on the SFT+DPO 4-bit checkpoint
       → eval/results/qwen2.5-coder-14b-mlx-4bit-tried-ft/
G. Compare with paired McNemar (pass rate) and paired Wilcoxon (log-speedup),
   joining the label folders by example_id. Three pairwise comparisons:
   base vs SFT, base vs SFT+DPO, SFT vs SFT+DPO.
```

The three eval result folders are joinable by `example_id` (the locked holdout's join key) — no schema changes needed. The SFT-only eval is what tells us whether DPO helped, hurt, or was a wash; without it, a positive base-vs-(SFT+DPO) result can't be attributed.

A new generator client `packages/orchestrator/src/orchestrator/clients/mlx_generator_client.py` loads a local MLX checkpoint and exposes the same interface as `clients/generator_client.py` (Ollama). The eval runner selects which client to use via env var (e.g., `TRIED_GENERATOR_BACKEND={ollama,mlx}`) plus a checkpoint path. Generation parameters (`temperature=0`, `max_tokens=2048`, markdown-fence stripping) match the Ollama-vanilla config recorded in `config/specs.yaml` exactly — the MLX client reads these from the shared control config so all eval conditions share one knob.

### Pre-registered headline thresholds

"Fine-tuning helped" requires BOTH of the following on the base-vs-(SFT+DPO) comparison:
- **Statistical significance:** two-sided paired McNemar on pass rate, p < 0.05.
- **Practical significance:** pass-rate lift ≥ +3 percentage points absolute.

The +3pp threshold corresponds to Cohen's h ≈ 0.1 (small-but-real effect) and is within the ~80% power that n=437 gives at realistic within-pair correlation ρ=0.7. Report p, lift, and effect size regardless of which way they fall — a p<0.05 result with a +1pp lift is "statistically detectable, practically marginal" and should be reported as such, not as a headline win. The same dual threshold applies to the SFT-only and SFT-vs-SFT+DPO contrasts. Per the Week 7 non-parametric deck, also report median + IQR and the rank-biserial effect size alongside the Wilcoxon p-value on log-speedup.

## File layout

```
packages/orchestrator/src/orchestrator/improvement/
├── __init__.py
├── builders/
│   ├── __init__.py
│   ├── sft_builder.py            # dataset.jsonl → sft.jsonl
│   └── dpo_builder.py            # dataset.jsonl → dpo.jsonl
├── training/
│   ├── __init__.py
│   ├── sft.py                    # wraps mlx_lm.lora for SFT
│   └── dpo.py                    # wraps mlx_lm.lora for DPO
└── merge.py                      # wraps mlx_lm.merge + mlx_lm.convert to produce 4-bit merged checkpoints

packages/orchestrator/src/orchestrator/clients/
└── mlx_generator_client.py       # MLX-backed inference, same interface as Ollama client

data/improvement/
├── sft.jsonl                     # built artifact
├── dpo.jsonl                     # built artifact
├── val_split.json                # persisted held-out example_ids, stratified by op_category
└── checkpoints/                  # gitignored; see .gitignore
    ├── sft-adapters/
    ├── sft-merged-4bit/          # eval target (SFT-only)
    ├── sft-dpo-adapters/
    └── sft-dpo-merged-4bit/      # eval target (SFT+DPO)

eval/results/                     # three eval folders, joined by example_id
├── qwen2.5-coder-14b-mlx-4bit-baseline/
├── qwen2.5-coder-14b-mlx-4bit-sft/
└── qwen2.5-coder-14b-mlx-4bit-tried-ft/
```

Config lives in three files under `config/`:

- **`config/config.yaml`** — authoritative spec. Single source of truth for the random seed, generation params (`temperature`, `max_tokens`), chat-template reference, and training hyperparameters. Read by the orchestrator (eval clients + fine-tuning wrappers).
- **`config/finetuning_sft.yaml`** — technical SFT config consumed directly by `mlx_lm_lora.train`. Uses mlx-lm-lora's exact key names (`learning_rate`, `max_seq_length`, `mask_prompt`, `gradient_accumulation_steps`, `lora_parameters.{rank,dropout,scale}`).
- **`config/finetuning_dpo.yaml`** — same shape, DPO-specific values (plus `beta`, `dpo_cpo_loss_type`, `reference_model_path`).

The two stage YAMLs MUST mirror the values in `config.yaml`. `orchestrator.improvement.preflight.preflight_check(stage)` is called by each trainer wrapper at startup and aborts the run on any mismatch (per-stage hyperparams, the experiment-wide seed, and the SFT `model` field against `training.base_model`). The stage YAMLs additionally carry mlx-lm-lora-specific keys that don't live in `config.yaml` and aren't preflight-enforced: `lr_schedule` (warmup + decay), `load_in_4bits: false` (uniform for both stages — every base we load is already 4-bit on disk, so we let mlx-lm-lora preserve the existing quantization; setting it true would raise "Cannot quantize already quantized model"), `grad_checkpoint`, `num_layers`, `fuse: false`, and `dpo_cpo_loss_type: sigmoid`.

Shape of `config.yaml`:

```yaml
schema_version: 1

inference:                         # read by MLX client + OpenRouter client + eval runner
  temperature:    0
  max_tokens:     2048
  chat_template:  "Qwen/Qwen2.5-Coder-14B-Instruct"   # tokenizer ID whose apply_chat_template is used
  seed:           20260522                             # one seed across the whole experiment

openrouter:                        # reference-baseline conditions
  base_url:       "https://openrouter.ai/api/v1"
  temperature:    0
  max_completion_tokens: 2048
  seed:           20260522
  models:
    llama:    { model_id: "meta-llama/llama-3.3-70b-instruct" }
    deepseek: { model_id: "deepseek/deepseek-v4-flash" }

training:                          # mirrored into finetuning_{sft,dpo}.yaml
  base_model:     "mlx-community/Qwen2.5-Coder-14B-4bit"   # SFT base; DPO uses the SFT-merged fp16 staging dir
  val_split_path: data/improvement/val_split.json
  val_fraction:   0.05
  stratify_by:    op_category
  sft:
    learning_rate:               1.0e-4
    epochs:                      2
    batch_size:                  2
    gradient_accumulation_steps: 8       # effective batch = 16
    max_seq_length:              2048
    mask_prompt:                 true    # response-only loss masking
    lora: { rank: 64, dropout: 0.1, scale: 2.0 }   # scale = alpha/rank (alpha=128, rank=64)
  dpo:
    learning_rate:               5.0e-7
    epochs:                      1
    batch_size:                  1
    gradient_accumulation_steps: 8       # effective batch = 8
    max_seq_length:              4096
    beta:                        0.1
    lora: { rank: 64, dropout: 0.1, scale: 2.0 }
```

All three files are checked into git. Changing any value requires updating both the authoritative file and the mirroring stage YAML; the preflight will refuse to start training otherwise.

## Decisions that need a `docs/decision-log.md` entry once implemented

- Switch from Ollama-Q4 vanilla baseline to MLX-4bit baseline; vanilla deleted.
- QLoRA rank 64, bf16 adapters, with the recipe above; SFT-then-DPO.
- Post-merge re-quantization workflow: merge → fp16 (transient) → convert to 4-bit; only 4-bit checkpoint kept.
- Attempt-0 SFT prompt rendering (winning code regardless of which attempt produced it).
- Same-row DPO pair construction (attempt-0 prompt; chosen = winner; rejected = earliest failing attempt). Failure-mode mix kept as-is (85/15/0 runtime/numeric/compile).
- **Three-eval comparison** (4-bit base, SFT-only, SFT+DPO; all via MLX), joined by `example_id` for paired tests.
- **Chat-template parity**: training data and MLX inference both use `apply_chat_template` from `Qwen/Qwen2.5-Coder-14B-Instruct`; no hand-rolled wrapping.
- **Response-only loss masking** for SFT.
- **Max sequence length: SFT 2048, DPO 4096** validated empirically. SFT: 776 positives, max 1218 tokens, p99 1073. DPO: 1256 sides (628 pairs × 2), max 2506 tokens, p99 993 — 4096 covers all observed rows; 2048 would have clipped 5 sides / ~0.8% of pairs.
- **Headline test thresholds pre-registered**: McNemar two-sided p<0.05 AND pass-rate lift ≥ +3pp; Wilcoxon on log-speedup with rank-biserial effect size and median+IQR reporting.
- Shared experiment-config file at `config/config.yaml` is the single source of truth for seed, generation params, and training hyperparameters. The mlx-lm-lora-specific per-stage configs live next to it as `config/finetuning_sft.yaml` and `config/finetuning_dpo.yaml` and must mirror the authoritative values; `orchestrator.improvement.preflight` enforces this at run start.

## Open decisions

- **DPO validation strategy.** Reward margin / chosen-rejected logprob gap on a held-out slice — deferred to DPO scaffolding time.
- **Training time budget.** Estimated 30 min – 2h SFT, similar for DPO; needs a 50-row smoke run end-to-end before committing to a training window. Eval is the larger time bucket (expected ~1 week per condition based on the prior vanilla run; three conditions plan for ~3 weeks of eval compute).

## What NOT to do

- Don't change `packages/orchestrator/src/orchestrator/prompts/generator/` — the prompt is locked, and training prompt drift relative to inference is a silent regression source.
- Don't pull `eval/holdout/synthetic_fusions.jsonl` rows into the training data, ever. Not as SFT positives, not as DPO chosens, not as held-out validation. The holdout is locked.
- Don't compare fine-tuned against a different-precision baseline. All 4 local Qwen conditions run at 4-bit; same precision on both sides or the result is uninterpretable.
- Don't run DPO before SFT.
- Don't tune hyperparameters by peeking at the holdout eval. Tune SFT against the 5% carved validation split; DPO uses the SFT checkpoint as reference and is evaluated only on the holdout once, at the end.
