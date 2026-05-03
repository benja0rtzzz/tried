# Triton Experimental Development (TRIED) - AI Triton Agent

## What this project is

An agent that translates PyTorch code to Triton kernels, verifies the output by executing and comparing against PyTorch eager + `torch.compile` Inductor baselines, and records every attempt into a dataset used for RAG retrieval and LoRA fine-tuning. 2-month student project, 3 devs.

The deliverable is a TUI application (VSCode extension as stretch) on top of a translation backend. The full pipeline includes the agent loop, verification harness, RAG retrieval over past attempts, and LoRA fine-tuning. Industry-relevance framing: developer productivity and GPU cost reduction, not "beat torch.compile".

**This is a controlled experiment.** Schema, tolerance policy, and prompt are set once before data collection begins and do not change during the experiment. Changing any of them mid-run invalidates comparisons between attempts.

## Architecture

Two machines, two roles:

- **M4 MacBook Pro (48GB)** — Orchestrator: runs the agent loop, calls the local LLM via Ollama, HTTP-POSTs to the Lenovo verification server, calls the judge (Azure), manages retries, writes the dataset. Fine-tuning via MLX / mlx-lm.
- **Lenovo LOQ (RTX 4060 8GB)** — Verification server: FastAPI service that accepts Triton + PyTorch code, compiles, executes, benchmarks, and returns structured results. Triton requires CUDA.

Agent loop: PyTorch input → Orchestrator calls local LLM (Ollama) → Orchestrator HTTP-POSTs to Lenovo FastAPI (`/compile`, `/run`, `/benchmark`) → results returned → judge (Azure o4-mini) classifies outcome → retry up to N=5 → record every attempt.

The generator is a local LLM on the MacBook. The judge is Azure-hosted. **The judge never generates kernels, only classifies and advises.** Never conflate these roles.

See @docs/architecture.md for the full data flow and FastAPI contract.

## Package structure (UV workspace)

```
tried/
  pyproject.toml              ← UV workspace root
  packages/
    shared/                   ← schema, enums, tolerance policy, dataset I/O (both machines)
    orchestrator/             ← agent loop, LLM calls, judge, prompts, RAG, fine-tuning (MacBook only)
    verification/             ← FastAPI server, compile harness, benchmark rig (Lenovo only)
  eval/holdout/               ← LOCKED. Never modified programmatically.
  data/                       ← gitignored except small samples
  docs/                       ← knowledge base (see below)
```

## Non-negotiable rules

- **The held-out evaluation set is sacred.** `eval/holdout/` must NEVER enter training, RAG retrieval, or prompt examples. If unsure whether an example belongs here, assume eval and ask.
- **Every attempt goes in the dataset.** Including failures. Never delete failed attempts — they are material for DPO and error-pattern retrieval.
- **Closed vocabularies are enforced in code.** `judge_classification`, `final_outcome`, `op_category`, `tolerance_policy_used` — all Python Enums, validated before write. Never invent a new value; add to the Enum first and discuss with the team.
- **Tolerance policy lives in one place.** `packages/shared/src/shared/verification/tolerance.py` is the single source of truth. Don't hardcode `atol`/`rtol` anywhere else. Record the policy key used with every correctness check.
- **Record all 10 correctness numbers** (5 stats × eager and Inductor): `max_abs_diff`, `max_rel_diff`, `mean_abs_diff`, `n_elements_exceeding_tol`, `pct_elements_exceeding_tol`.
- **Benchmarks are always relative.** Report speedup vs eager and vs Inductor. Absolute ms is not the primary metric.
- **The prompt is fixed.** It lives in `packages/orchestrator/src/orchestrator/prompts/`. Do not change it once the experiment begins. A different prompt is a different experiment.
- **No examples in the prompt.** Including Triton kernel examples in the prompt template would bias the generator and compromise the baseline. The model must stand on its own.
- **Schema is fixed** before data collection begins. Agreed upon by the team, then immutable for the duration of the experiment.

## Locked files — do not edit

These files encode experimental invariants. Editing them mid-experiment invalidates comparisons between runs. They are set once, before data collection, in agreement with the full team.

- `docs/schema.md`, `packages/shared/src/shared/schema/dataset_record.json`, and `packages/shared/src/shared/schema/eval_and_training.json` — dataset and corpus schemas
- `docs/tolerance-policy.md` and `packages/shared/src/shared/verification/tolerance.py`
- `eval/holdout/` — the evaluation set (never touch programmatically)

## Workflow expectations

- Run the full verification harness on any Triton code before committing results. Don't trust "it compiled".
- When writing the generator prompt, assume the model does not know Triton well. Give explicit block-size guidance. Do not include kernel examples.
- Ask before adding dependencies to any package.

## Logging

All logging must go through the shared logger. Never use `print()`, `logging.getLogger()` directly, or any other logging implementation.

```python
from shared.logging import get_logger
logger = get_logger(__name__)
```

Set `TRIED_ROLE=orchestrator` on the MacBook and `TRIED_ROLE=verification` on the Lenovo before running anything. The logger stamps every line with the machine role. Implementation: `packages/shared/src/shared/logging/__init__.py`.

## What NOT to do

- **Never write Triton kernels.** Not as examples, not as fixes, not as references. No exceptions.
- Do not use `torch.allclose` with default tolerances. Always go through the tolerance policy module.
- Do not free-text categorize errors. Use the Enum.
- Do not benchmark on battery power or with other GPU work running.
- Do not edit the held-out eval set. Ever.
- Do not change the prompt, schema, or tolerance policy mid-experiment.
- Do not use `print()` or `logging.getLogger()` directly — always use `get_logger` from `shared.logging`.

## Current state

Week 1 of an 8-week project. Schema, tolerance policy module, held-out eval set locked, model bakeoff results.

## Docs knowledge base

- @docs/architecture.md — two-machine data flow, FastAPI endpoint contract
- @docs/schema.md — human-readable schema explanation (machine specs: `packages/shared/src/shared/schema/dataset_record.json` and `packages/shared/src/shared/schema/eval_and_training.json`)
- @docs/benchmarking-protocol.md — verification steps, timing rules, hardware pinning
- @docs/tolerance-policy.md — tolerance values and rationale
- @docs/model-choices.md — generator/judge selection rationale, bakeoff results
- @docs/dev-setup-macbook.md — MacBook environment setup (UV, Ollama, MLX)
- @docs/dev-setup-lenovo.md — Lenovo environment setup (UV, CUDA, Triton, uvicorn)
- @docs/decision-log.md — ADR-lite log of key decisions
- @docs/corpus.md — corpus plan, train/eval split, extraction rules
