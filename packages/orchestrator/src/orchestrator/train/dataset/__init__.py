"""Dataset-generation pipeline (agent loop + entry point).

The dataset pipeline is the original retry orchestrator: it iterates the
preflight-safe training corpus, runs generate → compile → run retries, calls
the judge for every failed attempt, and appends DatasetRow records to
data/dataset/dataset.jsonl. Benchmarking is eval-only.

The eval pipeline lives in `orchestrator.eval.eval_gen` (corpus generation) and
`orchestrator.eval.eval_run` (single-attempt eval runner) and is intentionally
kept separate.
"""
