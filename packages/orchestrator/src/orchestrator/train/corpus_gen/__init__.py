"""Training-corpus skeleton generator.

Pipeline (mirrors orchestrator.eval.eval_gen but produces training inputs, not eval rows):

  1. discovery     — walk .repos/, classify every @triton.jit callsite into
                     (op_category, structural pattern). Outputs observations.jsonl.
  2. sampler       — stratified sampler over the enum cross-product, weighted
                     by observed frequency × per-category quota. Outputs specs.jsonl.
  3. driver        — codex CLI generates one PyTorch skeleton per spec; AST validator
                     and dedup-vs-eval reject; outputs with_code.jsonl + rejected.jsonl.
  4. preflight     — eager-vs-Inductor sanity check per row on Lenovo; passing rows
                     append to data/preflight_safe.jsonl.

The skeleton (PyTorch only) is what the agent loop sees. Curated Triton from the
cloned repos is intentionally NOT used — repos serve as structural inspiration only.
"""
