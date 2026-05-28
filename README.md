# TRIED

TRIED is a research experiment about LLM-based Triton kernel generation.

The project studies whether a local code model can improve at generating correct Triton kernels when trained from a verified dataset of model attempts, failures, judge feedback, and successful repairs.

TRIED is not a deployed kernel-generation product. It is an experimental system for creating data, running controlled evaluations, and measuring whether generated verification data can improve a local model under a fixed evaluation protocol.

## Research Question

The central question is:

**Can generated verification data improve a local model's ability to produce correct Triton kernels?**

The experiment focuses on correctness first. A Triton candidate is only meaningful if it passes numerical verification against PyTorch eager execution and `torch.compile` Inductor under the selected tolerance policy. Performance is evaluated only after correctness passes.

## Experimental Design

TRIED separates data creation from evaluation.

The dataset loop is allowed to use retries and judge feedback. A model generates a Triton candidate, the verification server compiles and runs it, failures are classified, and later attempts can use targeted repair advice. Both failed and successful attempts are recorded because the failure patterns are part of the training signal.

The evaluation loop is stricter. Each model gets one attempt per held-out example. There is no judge, no retry, and no manual repair. The same holdout set, prompt structure, tolerance policy, and verification harness are used across model conditions.

This separation is the core experimental control: training data can come from an iterative repair process, but evaluation measures raw first-attempt behavior.

## System Overview

TRIED uses a two-machine setup:

- an orchestrator machine for model calls, dataset generation, evaluation control, and fine-tuning
- a CUDA verification machine for Triton compilation, runtime verification, and benchmarking

The orchestrator sends requests to a FastAPI verification server. The verification server provides endpoints for preflight checks, static Triton validation, runtime correctness checks, and benchmark jobs.

## Documentation

- Repeatability settings and reproduction procedure: `docs/repeatability.md`
- System architecture and verification server flow: `docs/architecture.md`
- Corpus construction and held-out evaluation data: `docs/corpus.md`
- Correctness and benchmarking rules: `docs/benchmarking-protocol.md`
- Numerical tolerance policy: `docs/tolerance-policy.md`
- Model choices and roles: `docs/model-choices.md`
- Fine-tuning procedure: `docs/finetuning.md`
- Evaluation statistics plan: `docs/eval-stats.md`
- Dataset and evaluation schemas: `docs/schema.md`
- Experiment decisions and rationale: `docs/decision-log.md`
