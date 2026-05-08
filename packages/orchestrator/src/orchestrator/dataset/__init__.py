"""Dataset-generation pipeline (agent loop + entry point).

The dataset pipeline is the original orchestrator: it iterates the training
corpus, runs the full preflight → generate → compile → run → benchmark →
judge retry loop, and appends DatasetRow records to dataset.jsonl.

The eval pipeline lives in `orchestrator.eval_gen` (corpus generation) and
`orchestrator.eval_run` (single-attempt eval runner) and is intentionally
kept separate.
"""
