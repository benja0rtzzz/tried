# Data Sources

External repositories cloned for training-corpus expansion. Lives outside the locked-files set and is updated as extraction proceeds.

> Eval is **not** sourced from these repos. The held-out eval set (`eval/holdout/synthetic_fusions.jsonl`) is synthetic-fusion only and remains locked.

## Methodology

Bringing the corpus to the ~1000-row target needed to produce ~150 fast-correct positives requires broader coverage — particularly in the op categories where the agent loop currently has 0% pass rate (`fused_attention`, `loss`, `convolution`) and very low pass rate (`matmul`, `normalization`, `embedding`). The repos listed below are cloned to `.repos/` (gitignored, ephemeral) so a codex-driven extractor can pull `(pytorch_code, triton_code)` pairs at scale. Every cloned repo ships **real Triton kernels paired with PyTorch reference implementations**, which is the precondition for high-signal SFT examples; repos that only have CUDA kernels or only have PyTorch are excluded.

The extractor produces standard `CorpusRecord` rows: each row has a single PyTorch function as `source.pytorch_code`, the matching `input_shapes` and `input_dtypes`, an `op_category` from the closed enum, and an `origin` of the form `<repo>/<op_name>` so per-row provenance is preserved. The Triton reference paired with each PyTorch function is **not** stored in the corpus row — the corpus is a list of inputs to the agent loop, not (input, output) pairs.

Reproducibility is anchored on pinned commit SHAs. `.repos/clone.sh` records each repo's HEAD SHA at clone time into `.repos/COMMITS.txt`; re-running the script with `--reset` returns every checkout to those exact SHAs. The SHA list is the canonical citation surface — the per-repo entries below reference `COMMITS.txt`.

Licensing posture: every repo is under a permissive license (BSD-2-Clause-Patent, BSD-3-Clause, Apache 2.0, or MIT). We are not redistributing source code — only derived PyTorch patterns appear in the corpus. Attribution is satisfied at two levels: (1) per-row, via the `origin` field on every dataset record; (2) aggregate, via this document (URL + license + pinned SHA per repo).

## Pinned commits

Each repo's exact commit SHA at the time of extraction lives in `.repos/COMMITS.txt`. The format is tab-separated: `name<TAB>sha<TAB>committer_date_iso8601<TAB>tier<TAB>url`. Re-clone with `.repos/clone.sh`; reset to the exact recorded SHAs with `.repos/clone.sh --reset`.

## Per-repo entries

Tier 1 = best pair quality, broadest category coverage. Tier 2 = filler for under-quota categories.

### linkedin/Liger-Kernel — Tier 1

- **URL:** https://github.com/linkedin/Liger-Kernel
- **License:** BSD-2-Clause-Patent
- **Pinned commit:** `97b6fe2bb6cc06af426c31c6ec9577835cb46ba9` (2026-05-09)
- **Categories targeted:** `normalization`, `loss`, `embedding`, `activation`, `fused_attention`
- **Why:** Designed exactly as PyTorch+Triton drop-ins; every kernel ships with an eager reference in the same module. Highest pair-density per file in the set.
- **Examples extracted:** _populated by extraction tool — list `<repo>/<op>` ids here, e.g._
  - `liger/rms_norm`
  - `liger/layer_norm`
  - `liger/cross_entropy`
  - `liger/jsd`
  - `liger/swiglu`
  - `liger/geglu`
  - `liger/fused_linear_cross_entropy`

### Dao-AILab/flash-attention — Tier 1

- **URL:** https://github.com/Dao-AILab/flash-attention
- **License:** BSD-3-Clause
- **Pinned commit:** `ab66326aaa4fe3529fbc00f3156f3a762dd3141b` (2026-05-08)
- **Categories targeted:** `fused_attention`
- **Why:** Reference for fused causal/non-causal attention. PyTorch reference implementations live in `flash_attn/triton/test_*.py` and `flash_attn/flash_attn_triton.py`.
- **Examples extracted:** _populated by extraction tool_

### pytorch/ao (TorchAO) — Tier 1

- **URL:** https://github.com/pytorch/ao
- **License:** BSD-3-Clause
- **Pinned commit:** `fe5d41ed2bd28a27f165876fd8ede9dc7f4dbe32` (2026-05-09)
- **Categories targeted:** `quantization`, `matmul`
- **Why:** NF4, FP8, int4, fake-quant kernels with eager fallbacks; mixed-precision matmul references.
- **Examples extracted:** _populated by extraction tool_

### pytorch-labs/tritonbench — Tier 1

- **URL:** https://github.com/pytorch-labs/tritonbench
- **License:** BSD-3-Clause
- **Pinned commit:** `4d68cf47f562d486f067193aefd1549deb42e1f9` (2026-05-08)
- **Categories targeted:** broad — `matmul`, `reduction`, `normalization`, `activation`, `loss`, `embedding`, `quantization`
- **Why:** ~48 ops per `operators/` with per-op PyTorch ref wrappers; this is the primary training source already partially extracted in v1/v2.
- **Examples extracted:** _populated by extraction tool_

### triton-lang/triton — Tier 1

- **URL:** https://github.com/triton-lang/triton
- **License:** MIT
- **Pinned commit:** `521c2e378febcebc7cf0c1e41d7c6679e144bd50` (2026-05-09)
- **Categories targeted:** `matmul`, `reduction`, `elementwise_chain`, `activation`
- **Why:** Canonical pedagogical pairs in `python/tutorials/` (`01-vector-add.py` ... `09-persistent-matmul.py`). Small, clean, well-commented — high SFT signal density.
- **Examples extracted:** _populated by extraction tool_

### BobMcDear/attorch — Tier 1

- **URL:** https://github.com/BobMcDear/attorch
- **License:** MIT
- **Pinned commit:** `04c5e3edad907b32006416382db7f0f68af31670` (2025-08-12)
- **Categories targeted:** `normalization`, `activation`, `loss`, `embedding`, `elementwise_chain`
- **Why:** A pure-Python re-implementation of a subset of `torch.nn` modules in Triton. Each module ships with both the eager PyTorch reference and the Triton kernel side by side — densest pair-per-file source in the set.
- **Examples extracted:** _populated by extraction tool_

### FlagOpen/FlagGems — Tier 1

- **URL:** https://github.com/FlagOpen/FlagGems
- **License:** Apache 2.0
- **Pinned commit:** `183cc3d78eace1d4a8b12e36bb4dad5f4b1efe14` (2026-05-09)
- **Categories targeted:** broad — `matmul`, `reduction`, `normalization`, `activation`, `loss`, `embedding`, `quantization`, `fused_attention`
- **Why:** General-purpose Triton operator library targeting LLM training / inference. Coverage breadth comparable to tritonbench, with eager refs adjacent to each kernel.
- **Examples extracted:** _populated by extraction tool_

### fla-org/flash-linear-attention — Tier 1

- **URL:** https://github.com/fla-org/flash-linear-attention
- **License:** MIT
- **Pinned commit:** `07e2bd329df15090d34c7d4b25b638b3cb57c516` (2026-05-10)
- **Categories targeted:** `fused_attention`, `other` (linear attention recurrences)
- **Why:** Linear-attention family kernels (RWKV, GLA, RetNet, Mamba2-style) — structurally distinct from softmax attention in flash-attention.
- **Examples extracted:** _populated by extraction tool_

### meta-pytorch/applied-ai — Tier 1

- **URL:** https://github.com/meta-pytorch/applied-ai
- **License:** BSD-3-Clause
- **Pinned commit:** `2391954b19988bd76cf3c2ea84c1ce74b68d568b` (2025-08-22)
- **Categories targeted:** `matmul`, `quantization`, `fused_attention`
- **Why:** Meta's applied-AI working repo (formerly under `pytorch-labs/`); production-grade patterns with PyTorch refs.
- **Examples extracted:** _populated by extraction tool_

### meta-pytorch/attention-gym — Tier 1

- **URL:** https://github.com/meta-pytorch/attention-gym
- **License:** BSD-3-Clause
- **Pinned commit:** `29185740237a6e02c55740be8333cb744abccbd7` (2026-04-12)
- **Categories targeted:** `fused_attention`
- **Why:** FlexAttention recipes — wide variety of attention masks, biases, and score-mod patterns. Shape diversity within the attention category.
- **Examples extracted:** _populated by extraction tool_

### pytorch/FBGEMM — Tier 2

- **URL:** https://github.com/pytorch/FBGEMM
- **License:** BSD-3-Clause
- **Pinned commit:** `26bfbd50106412e62893de5cf19f093c5b502351` (2026-05-08)
- **Categories targeted:** `embedding`, `quantization`
- **Why:** Strong on jagged-tensor and embedding-bag kernels; rare-tensor patterns underrepresented elsewhere.
- **Examples extracted:** _populated by extraction tool_

### state-spaces/mamba — Tier 2

- **URL:** https://github.com/state-spaces/mamba
- **License:** Apache 2.0
- **Pinned commit:** `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` (2026-05-09)
- **Categories targeted:** `other` (recurrent / state-space scans)
- **Why:** Selective scan and SSD chunk-scan kernels paired with reference PyTorch implementations.
- **Examples extracted:** _populated by extraction tool_

### vllm-project/vllm — Tier 2

- **URL:** https://github.com/vllm-project/vllm
- **License:** Apache 2.0
- **Pinned commit:** `f80aa53c9dc2273a19a6855092069db7e1306fff` (2026-05-09)
- **Categories targeted:** `fused_attention`, `quantization`, `normalization`
- **Why:** Production attention + paged-KV variants. Pair extraction harder (kernels embedded in agent code) — use only for categories still under quota after Tier 1.
- **Examples extracted:** _populated by extraction tool_

### facebookresearch/xformers — Tier 2

- **URL:** https://github.com/facebookresearch/xformers
- **License:** BSD-3-Clause
- **Pinned commit:** `ca6d2aa0d43241fc8a8dcd872debc2406160160d` (2026-04-21)
- **Categories targeted:** `fused_attention`, `normalization`
- **Why:** Mature attention variants. Some kernels are build-flag-gated (CUTLASS path); the Triton path under `xformers/triton/` is the extractable subset.
- **Examples extracted:** _populated by extraction tool_

### shawntan/scattermoe — Tier 2

- **URL:** https://github.com/shawntan/scattermoe
- **License:** Apache 2.0
- **Pinned commit:** `47b5e1502e5a10e82c8e5945d761b877849871e7` (2025-10-03)
- **Categories targeted:** `other` (sparse MoE / scatter-gather)
- **Why:** Triton implementation of sparse Mixture-of-Experts; routing + scatter patterns underrepresented elsewhere in the corpus.
- **Examples extracted:** _populated by extraction tool_

### AlibabaPAI/FLASHNN — Tier 2

- **URL:** https://github.com/AlibabaPAI/FLASHNN
- **License:** Apache 2.0
- **Pinned commit:** `528a9301587f5fb135b25d973a87ba0a40a703a7` (2024-09-09)
- **Categories targeted:** `fused_attention`, `quantization`, `normalization`
- **Why:** LLM serving kernels (paged-KV attention, INT8 GEMM, fused norm + activation). Production patterns alternative to vLLM.
- **Examples extracted:** _populated by extraction tool_

## What is NOT used

- Triton kernel sources from these repos are never indexed into RAG and never appear in any agent-loop prompt.
- These repos do not contribute eval examples. The locked eval set is synthetic-fusion only.
- vLLM and xFormers production code paths gated on CUDA/CUTLASS (not Triton) are excluded.
