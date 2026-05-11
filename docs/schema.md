# Dataset Schema

> **Locked file.** Do not make changes. Machine-readable spec: `packages/shared/src/shared/schema/dataset/dataset_record.json`.

Each row in the training dataset represents one complete PyTorch-to-Triton translation job. It contains a `source` block with immutable inputs (PyTorch code, `origin`, input shapes, dtypes, RNG seed, and op category); a list of up to five `attempts`, each recording the generated Triton code, compile result, correctness stats against both PyTorch eager and Inductor, judge classification, and the judge's fix suggestion for the next retry; and a top-level `final_outcome`.

Training dataset rows do **not** record benchmark timings, speedups, or `final_winning_attempt_n`; benchmarking is eval-only. Every training row is guaranteed to have passed a pre-flight eager-vs-Inductor sanity check before the dataset loop starts. All categorical fields use closed vocabularies enforced as Python Enums in `packages/shared/src/shared/enums.py`; adding a value requires a decision-log entry and team sign-off before implementation. For corpus provenance and the train/eval split, see `docs/corpus.md`.
