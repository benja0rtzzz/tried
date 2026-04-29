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

