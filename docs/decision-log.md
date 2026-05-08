# Decision Log

Lightweight record of non-obvious decisions. Full reasoning lives in the Claude Code session that produced the change. Entries here are the minimum a teammate needs to understand what was decided and why without having to ask.
All entrees here are accepted.
---

## 2026-05-09 — Orchestrator split: dataset pipeline moved into its own subpackage alongside new eval pipelines

The orchestrator originally housed a single pipeline — the dataset-generation agent loop — at `orchestrator/main.py` and `orchestrator/agent.py`. The eval workflow has since grown into two distinct pipelines (`eval_gen/` for synthetic-fusion corpus generation, `eval_run/` for the single-attempt eval runner against the locked holdout), each with its own entry point and helpers. Leaving the dataset pipeline at the package root made the layout asymmetric and obscured that there are now three peer pipelines, not one pipeline plus eval extras.

The dataset pipeline is now `orchestrator/dataset/` (`main.py` + `agent.py`, moved with `git mv` to preserve history). New run command: `TRIED_ROLE=orchestrator uv run python -m orchestrator.dataset.main`. Shared infrastructure — `clients/`, `prompts/`, `improvement/` — stays at the package root because all three pipelines use it. No behavior change; this is a layout refactor recorded so the experiment trail shows when the split happened.

Implementation: `git mv` of the two files into `orchestrator/dataset/`; new `orchestrator/dataset/__init__.py` documenting the split; self-import in `dataset/main.py` retargeted to `orchestrator.dataset.agent`; `packages/tests/pipeline_test.py` import updated; sibling comment in `eval_run/agent.py` retargeted to `orchestrator.dataset.agent`.

---

## 2026-05-09 — `baseline_compile` dropped from EvalRecord; Week 3 t-test target swapped to Triton compile time

Descriptive stats on the cleaned vanilla run revealed that `baseline_compile.inductor_first_call_ms` had a median of 65 ms (max 3.03 s, p99 well under 1 s) — three orders of magnitude below the "60–120 s headline cold compile cost" the schema description claimed. Inductor's on-disk kernel cache survives across rows in a single eval process (and across runs of the same process tree), so only the very first row of a fresh process pays a real cold compile; every subsequent row is a cache hit. The recorded numbers measure "first-invocation latency under a warm Inductor cache", not compile cost. The fine-tuned eval would inherit the same problem because it shares the same harness.

Forcing a real cold compile per row (clearing `torch._inductor` caches between rows) would add 7–14 hours of pure compile to each run, and the metric is secondary — the project's headline claim is `speedup_vs_inductor`, not "Triton compiles faster than Inductor". The course Week 3 rubric requires *a* paired t-test on project data; nothing requires it to be on cold-compile time. So `baseline_compile` is removed entirely from `EvalRecord`, the schema, and the orchestrator writer; the verification preflight API still returns `eager_first_call_ms` / `inductor_first_call_ms` because the corpus generator (`preflight_driver.py`) embeds them in `EvalCorpusRecord.preflight_*_ms`, where each spec is preflight-checked in isolation at sampling time and the cache argument doesn't apply the same way.

The Week 3 paired t-test is re-targeted to `log(attempts[winning].latency.compile_ms)` — Triton compile time, recorded per attempt by the verification server's `/compile` endpoint. Each call is a fresh `triton.JITFunction` compilation (no cross-row cache), so the values measure real compile cost. Vanilla median is 51 ms, IQR 48–57 ms, max 169 ms — clean distribution, log-transformed will be approximately normal and suitable for a paired t-test against the fine-tuned run. Wilcoxon on log-speedup and McNemar on pass rate are unaffected.

Implementation: removed `BaselineCompile` from `shared/models.py`; removed `baseline_compile` from `EvalRecord` and `schema/eval/record.json`; stripped `baseline_compile` from existing `eval/results/qwen2.5-coder:14b-vanilla/eval_rows.jsonl`; simplified `eval_run/agent.py` (no longer builds `BaselineCompile` and no longer rejects preflights with null first-call timings, since `passed=True` is sufficient); replaced `descriptive.baseline_compile_stats` with `descriptive.triton_compile_stats` and re-wired the report; renamed `hypothesis.paired_t_log_compile` to `paired_t_log_triton_compile`; updated `docs/eval-stats.md` (course-week table, Group A/B tables, Week 4 sample-size).

---

## 2026-05-09 — Eval set cleanup to 437 unique example_ids; `model_label` dropped from EvalRecord

The vanilla qwen eval finished and surfaced two issues. First, `eval/holdout/synthetic_fusions.jsonl` carried 4 duplicate `example_id`s from the spec sampler (443 lines, 439 unique), and the harness faithfully ran the duplicates twice — `eval_rows.jsonl` ended at 441 lines / 437 unique with 2 specs that never produced an `EvalRecord`. Second, `model_label` was an in-record field on every `EvalRecord` even though every row in a given file shares the same value; the parent directory `eval/results/<label>/` already identifies the condition, so the field was pure redundancy.

Both files were rewritten to the same 437 unique `example_id`s — duplicate occurrences dropped first-seen-wins, and the 2 specs missing from results were dropped from holdout to keep the fine-tuned eval aligned with vanilla. The held-out eval is in the locked set; this edit was authorized by the user as a one-shot cleanup before the second run, with no agent-loop or scoring change. `model_label` was removed from `schema/eval/record.json` (now 7 required fields), `EvalRecord` in `shared/models.py`, and the orchestrator writer; the orchestrator CLI still takes `--model-label` to determine the output folder. The tier counts moved from the original plan's 105 / 115 / 80 to the empirical 103 / 217 / 117 — `docs/corpus.md` and `docs/eval-stats.md` updated to match. Sample-size analysis at n=437 is at least as powerful as the original n=300 plan.

Implementation: `eval/holdout/synthetic_fusions.jsonl`, `eval/results/qwen2.5-coder:14b-vanilla/eval_rows.jsonl`, `packages/shared/src/shared/schema/eval/record.json`, `packages/shared/src/shared/models.py` (`EvalRecord`), `packages/orchestrator/src/orchestrator/eval_run/agent.py`, `docs/corpus.md`, `docs/eval-stats.md`.

---

## 2026-05-06 — Generator now sees its prior Triton code on retry

Prior to this change the generator received only the judge's natural-language fix advice between attempts; its own previous Triton code was discarded. A 13-row test run produced 12 `all_attempts_failed` outcomes with a clear cyclic pattern: each attempt would apply the latest advice but silently regress an earlier fix because the model had no working memory of the kernel state it was editing. Example trace (`47f75bb3...`, gelu+residual): int64 offsets → fix int32 cast → drop BLOCK_SIZE constexpr → restore BLOCK_SIZE but use invalid `tl.arange(dtype=...)` → drop `dtype` and BLOCK_SIZE again → back to the original int64 error.

The retry loop is meant to model the human-in-the-loop workflow this project automates (developer reads compile error → asks AI for advice → applies a *targeted edit* to the existing kernel), so producing a fine-tuning dataset from "advice → fresh attempt" pairs trains a different (and less useful) skill than what the deployed agent will need. The generator now receives the previous attempt's `triton_code` alongside `prior_advice`, with the prompt instructing it to apply the fix while preserving the parts that were already correct — not rewrite from scratch. This invalidates the existing 13-row dataset.jsonl, which is being scrapped before the official run starts. Locked-prompt status is preserved going forward; team agreed to the change pre-flight.

Implementation: `prompts/generator/generator_user.txt` (`{prior_advice_section}` → `{prior_attempt_section}`), `prompts/generator/__init__.py` (new `prior_code` parameter), `clients/generator_client.py` (forwards `prior_code`), `agent.py` (tracks `prior_code = gen.triton_code` across the retry loop).

---

## 2026-05-06 — Judge switched from Gemini 2.5 Flash to OpenAI o4-mini

Gemini 2.5 Flash was not producing useful fix suggestions — it classified most Triton JIT failures as `other` or repeated generic advice across retries. Root cause is model capability, not prompt design; expanding the prompt with error examples would paper over a reasoning gap rather than fix it.

o4-mini is a dedicated reasoning model at a comparable price point (~$6–11 for the full experiment). It was the original architecture choice before the free-tier Gemini path was taken. `reasoning_effort="high"` replaces `thinking_budget=1024`. Structured output uses `.beta.chat.completions.parse` with the same Pydantic response model. `OPENAI_API_KEY` replaces `GEMINI_API_KEY`. Model and reasoning effort are hardcoded (`o4-mini`, `"high"`).

---

## 2026-04-21 — Initial project structure
UV workspace with three packages (`shared`, `orchestrator`, `verification`). Schema and tolerance policy are locked files — changes require a log entry and team sign-off.

---

## 2026-04-24 — Dataset schema v1 adopted
Row = one PyTorch op, all attempts, final outcome. Key fields: `source` (immutable inputs), `attempts[]` (generate→compile→verify→judge cycles), `final_outcome`, `final_winning_attempt_n`, `tags`. Closed vocabularies for all categorical fields, enforced as Python Enums. Machine-readable spec: `docs/schema.json`.

---

## 2026-04-24 — Sanity check moved to pre-flight, removed from schema
Eager-vs-Inductor agreement check runs before the agent loop starts. Examples that fail are written to `skipped.jsonl` and never enter the dataset. Removed `sanity_check` block and `harness_broken` outcome — every row in the dataset is guaranteed clean.

---

## 2026-04-24 — Removed non-training fields from schema
Dropped: `retrieved_example_ids`, `retrieval_hash` (RAG audit — out of scope), `judge_input` (reconstructable from attempt data + fixed template), `prompt_hash` (redundant once RAG fields gone — prompt is deterministic from `source` + `prior_advice_applied`), `metadata` block (all values are experiment-wide constants — recorded once in `experiment-config.json` instead), `benchmark.device`, `benchmark.n_runs`, `benchmark.warmup_runs` (protocol-fixed constants). None of these feed SFT or DPO.

---

## 2026-04-24 — example_id switched to UUID
Dropped the `<op_category>_<index>` convention. `op_category` is already a field in `source`; embedding it in the ID was redundant and required per-category counter management. UUID is simpler and collision-resistant.

---

## 2026-04-25 — Corpus plan revised: training ~200 examples (50/50 split), eval 130 synthetic fusions
TritonBench ([paper](https://arxiv.org/abs/2502.14752)) is a research benchmark, not a corpus tool. Its 184/166 split refers to TritonBench-G (184 real-world ops) and TritonBench-T (166 synthetic fusion tasks) — the repo only ships 52 extractable operator implementations. The original 184/166 corpus plan was based on a misread of the paper.

**Training (~200 examples, 50/50):** ~100 from tritonbench ecosystem (48 from `operators/`, remainder from submodule refs in Liger-Kernel, Flash-Attention, FBGEMM, and select ATen loader entries) + ~100 from `torch.fx`-traced subgraphs from architecturally diverse models (transformers, CNNs, SSMs, MLPs). Tracer is being built anyway; marginal cost of more examples from it is low. Deduplication across models required — transformer architectures share patterns. `origin` convention: `"tritonbench/<op_name>"` for tritonbench ops, `"hf/<model>"` / `"timm/<model>"` for traced examples.

**Eval (130 synthetic fusions):** Synthetically constructed from tritonbench base ops + plain PyTorch built-ins not in training (`relu`, `sigmoid`, `tanh`, `exp`). Not contaminated by training: fusing ops is a different generation task than generating them individually. Split: Easy (~45, 2-op elementwise) / Medium (~50, 3-op or any reduction) / Hard (~35, 4+ ops or complex memory patterns). Difficulty tiers are for eval reporting only — all failures are recorded regardless. `origin`: `"synthetic/fusion"` for all eval examples. See `docs/corpus.md`.

---

## 2026-04-24 — source_model renamed to origin
`source_model` implied the op was extracted from a named model architecture (BERT, LLaMA, etc.). The field is renamed to `origin`. Field is non-nullable — every corpus row has a known origin. Format is `<source>/<name>` (see 2026-04-26 entry for the generalized convention).

---

## 2026-04-24 — op_category enum extended with loss and embedding
Added `loss` (cross-entropy, KL-divergence, JSD, fused-linear-loss variants) and `embedding` (embedding table lookups) to the `op_category` enum. Both are distinct compute patterns present in the tritonbench 184-op training corpus that do not fit the original 7 categories without overloading `other`. Remaining tritonbench ops that don't map cleanly (rope, mamba2, jagged, etc.) use `other`.
---

## 2026-04-25 — Deterministic corpus IDs and curated training provenance allowed
Corpus extraction now permits deterministic UUIDv5 `example_id` values derived from stable row contents. This keeps reruns joinable when the same PyTorch op is regenerated. Training provenance also now includes `curated/<name>` for hand-written standalone PyTorch patterns that fill underrepresented behavior buckets; these are training rows and are distinct from held-out `synthetic/fusion` eval rows.
---

## 2026-04-25 — Convolution promoted out of other
Scraped ResNet rows previously classified plain `conv2d` and `conv2d -> batch_norm -> relu` patterns as `other`. The corpus schema now includes `convolution` as a first-class `op_category`, and the v1 scraper emits convolutional model patterns with that category. `other` remains a small fallback bucket but should not carry common trainable compute families.

---

## 2026-04-26 — Origin format generalized to `<source>/<name>`
The restricted format list (`tritonbench/<op_name>`, `hf/<model_name>`, etc.) is replaced with the open-ended pattern `<source>/<name>`. Source can be any repo name (e.g. `flash-attention`, `mamba`, `torchao`), a model namespace (`hf`, `timm`), `curated`, or `synthetic`. This accommodates the v2 scraper batch without schema updates for each new source. All previously valid formats remain valid under the new pattern. `curated/train/<name>` shortened to `curated/<name>`.

---

## 2026-04-26 — op_category extended with `quantization`
Added `quantization` to `op_category`. Covers affine quant/dequant, fake quant, qmap, NF4/codebook, FP8, and dynamic per-token quantization patterns sourced from TorchAO, vLLM, and xFormers. Without it, all quantization rows collapse into `other`, obscuring the dominant behavior from the v2 scraper batch. Additional proposed categories (`state_space_scan`, `scatter_gather_indexing`, `logits_sampling`, `cache_update`, `positional_encoding`) are under review pending v2 notebook validation and will be added with separate log entries if accepted.

---

## 2026-05-01 — Benchmark timing variance added; tags removed

Added `triton_std_ms`, `eager_std_ms`, `inductor_std_ms` to the `benchmark` block (schema + Pydantic model). The Lenovo already runs 100 timing iterations to compute the median; std dev is free from those same samples. Without variance, speedup claims in the paper have no error bars. The three fields are required (non-nullable) alongside the existing ms fields.

`tags` removed from schema, CLAUDE.md, and docs/schema.md. It was initially planned as a multi-label stratification field but turned out to be redundant with `op_category`, which already carries the same information. No Enum or schema field was ever written for it; the removal is a documentation cleanup only.

---

## 2026-05-01 — Generator and judge prompt design locked

**Generator prompt structure:** System prompt provides Triton fundamentals and explicit block-size guidance (powers of 2, per-op defaults, masking requirement). User prompt supplies `pytorch_code`, `input_shapes`, `input_dtypes`, and an optional `prior_advice_section` that is empty on attempt 0 and renders the judge's fix suggestion on retries. No kernel examples in the prompt — the model stands on its own.

**Judge prompt structure:** System prompt defines the classification task, the closed vocabulary (9 labels matching `JudgeClassification` enum exactly), and the no-kernel-generation rule. Each judge call is stateless — a fresh API call with a single user message. Prior attempts are embedded as structured text blocks (code + result + fix suggestion per attempt), not sent as a multi-turn message array. This keeps each call independent and makes the system prompt fully cacheable.

**Judge model:** Gemini 2.5 Flash (Google AI Studio), temperature 0, `thinking_budget=1024`. Reasoning capability chosen for fix suggestion quality, not classification. Free tier covers the full experiment (~600 calls). Structured output via `response_mime_type="application/json"` + `response_schema` Pydantic model: `{"classification": "<label>", "fix_suggestion": "<string|null>"}`.

---

## 2026-04-29 — Shared package dataset I/O implemented
Three files added to `packages/shared/src/shared/`:

- `enums.py` — all closed-vocabulary Python enums (`OpCategory`, `Dtype`, `Split`, `Difficulty`, `FinalOutcome`, `JudgeClassification`, `CompileStatus`, `CorrectnessStatus`). Re-exports `TolerancePolicy` from `tolerance.py` as a single import point.
- `models.py` — Pydantic v2 models for `CorpusRecord` (eval_and_training.json shape) and `DatasetRow` (dataset_record.json shape), plus all sub-models. Cross-field invariants are enforced at validation time: shapes/dtypes length match, eval requires non-null difficulty, compile error null iff success, correctness null on compile failure, benchmark null on correctness failure, sequential attempt indices, winning attempt null iff outcome is a failure terminal.
- `dataset/__init__.py` — five I/O functions: `load_corpus_train`, `merge_corpus`, `load_dataset`, `append_dataset_row`, `append_skipped`. `merge_corpus` deduplicates by `example_id` (first-seen wins) and reports skipped duplicates to stdout.

v1 (80 rows) and v2 (110 rows) scraper outputs merged to `data/corpus_train.jsonl` (190 rows, zero duplicates). Duplicate `origin` strings within v1 are intentional — they represent distinct sub-graphs with different `input_shapes` and carry unique `example_id`s.

---

## 2026-05-04 — Verification server complete; orchestrator pipeline ready

All packages operational. Key decisions made during bring-up:

**Verification server (`packages/verification`):** `/health` endpoint added (CUDA availability, device stats, torch/triton versions). `VERIFICATION_API_KEY` now enforced at import time — server refuses to start if the env var is absent. Middleware unconditionally checks the key on every request.

**HTTP timeouts split by endpoint:** `verification_client.py` previously used a single 30 s timeout for all synchronous endpoints. `/preflight` and `/run` both trigger `torch.compile(backend="inductor")`, which takes 60–120 s on cold start. Split into `_COMPILE_TIMEOUT=30 s` (for `/compile`, `/benchmark` submit, job polls) and `_INDUCTOR_TIMEOUT=300 s` (for `/preflight` and `/run`). Configurable via `VERIFICATION_INDUCTOR_TIMEOUT_S` env var.

**Smoke test package (`packages/tests`, module `tried_tests`):** End-to-end test using the first corpus entry (`elementwise_clamp_square_fp32`, 1 M × float32). Uses the same PyTorch wrapper as both reference and candidate — exercises every GPU code path (Inductor JIT, CUDA execution, CUDA-Event timing, async benchmark job) with guaranteed correctness PASSED. Run: `TRIED_ROLE=verification VERIFICATION_API_KEY=<key> uv run python -m tried_tests.smoke`. Note: module named `tried_tests` (not `tests`) because hatchling silently excludes directories named `tests` from editable installs.

**Lenovo system deps required for Inductor:** `gcc` and `python3.12-dev` must be installed (`sudo apt install gcc g++ python3.12-dev`). Inductor generates C code that it compiles at runtime; without these, `/preflight` and `/run` fail with a compiler error.

---

## 2026-05-04 — Run resume and Gemini rate-limit handling

**Problem:** Gemini 2.5 Flash free tier resets daily. If the quota is exhausted mid-run, the process should stop cleanly rather than crash, and restart should continue from where it left off without re-processing any example.

**Resume logic (`main.py`):** On startup, `dataset.jsonl` and `skipped.jsonl` are read to collect all already-processed `example_id`s. The corpus list is filtered to exclude them before the loop starts. Log line reports how many are skipped. Cost: one extra file read at startup — negligible.

**Rate-limit detection (`judge_client.py`):** A `RateLimitError` exception is raised when the Gemini response signals quota exhaustion (HTTP 429, or message containing "quota"/"rate limit"). It propagates through `agent.py` (no catch there) to `main.py`, where it is caught before the generic transport-error handler. On `RateLimitError`, the run logs progress and exits cleanly with code 0. The next restart picks up from the resume filter.

---

## 2026-05-05 — JudgeClassification enum revised; experiment restarted

**Problem observed during test run:** The judge was classifying nearly all Triton JIT failures as `other` or `ambiguous`. Root cause: the generator (Qwen2.5-Coder:14b) makes systematic Triton API mistakes (int64 offset where int32 is required, constexpr parameters missing from the kernel signature, calling non-existent ops like `tl.tanh`, 2-D `tl.zeros` with a runtime dim) — a class of error with no specific label in the original enum, so the judge defaulted to catch-alls. The `other`/`ambiguous` distinction was also poorly defined, causing the judge to bounce between them arbitrarily.

**Changes (experiment restarted; prior test data discarded):**
- Added `triton_api_error` to `JudgeClassification`: covers wrong offset dtype, constexpr placement errors, unsupported Triton operations, and kernel launch signature mismatches.
- Removed `ambiguous`; `other` is now the single fallback for clear failures that don't fit a specific label.
- Judge system prompt updated with the new label (concrete examples included) and the tightened `other` definition.
- JSON schema (`dataset_record.json`) updated to match.

The Gemini structured-output response schema is derived from the Pydantic model at call time, so the client needed no changes.

