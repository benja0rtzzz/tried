# Decision Log

Lightweight record of non-obvious decisions. Full reasoning lives in the Claude Code session that produced the change. Entries here are the minimum a teammate needs to understand what was decided and why without having to ask.
All entrees here are accepted.
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

