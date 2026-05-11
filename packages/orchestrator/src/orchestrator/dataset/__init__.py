"""Dataset-generation pipeline (agent loop + entry point).

The dataset pipeline is the original judged retry orchestrator: it iterates the
preflight-safe training corpus, runs generate → compile → run → judge retry,
and appends DatasetRow records to dataset.jsonl. Benchmarking is eval-only.

The eval pipeline lives in `orchestrator.eval_gen` (corpus generation) and
`orchestrator.eval_run` (single-attempt eval runner) and is intentionally
kept separate.
"""
