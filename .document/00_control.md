# Document Control

## Core Research Question

Does task-specific adaptation improve one-shot Triton kernel generation over a vanilla code model?

## Audience and Reading Strategy

The paper must be approachable for readers who are curious about the project but not embedded in its implementation details, while still becoming specific enough to satisfy readers who understand GPU kernels, compilers, fine-tuning, and evaluation methodology.

The writing should therefore progress from accessible to technical:

1. Start each major section with the practical problem and why it matters.
2. Define specialized terms before relying on them.
3. Separate the research argument from implementation details.
4. Introduce complexity gradually, so each section rewards the reader with deeper technical precision.
5. Avoid overclaiming: distinguish confirmed results from planned SFT and DPO work.

## Source-Anchored Writing Principles

Use lightweight Obsidian-style source anchors when a drafting rule comes directly from the class material. These anchors are not formal citations; they are writing-memory links that explain why the document is structured this way.

Format:

```md
Principle text. Source: [[filename.pptx#slide-N]]
```

Global principles to apply across all notes:

| Principle | How to Apply It | Class Source |
|---|---|---|
| The paper is an argument, not a build log. | Every section should support the research question, not merely report what was implemented. | [[01__Que_es_Investigacion.pptx#slide-14]] |
| Start from the practical problem, then guide the reader toward the research question. | Begin with why GPU kernel generation matters before introducing SFT, DPO, metrics, or implementation details. | [[01__Que_es_Investigacion.pptx#slide-16]], [[01__Que_es_Investigacion.pptx#slide-17]] |
| Keep problem, research questions, and hypothesis distinct. | The problem states the gap; RQs organize what we ask; the hypothesis predicts what the data may show. | [[01__Que_es_Investigacion.pptx#slide-17]], [[01__Que_es_Investigacion.pptx#slide-20]] |
| Make claims falsifiable. | Improvement claims must be measurable against the locked vanilla baseline; avoid vague claims such as "better" without a metric. | [[01__Que_es_Investigacion.pptx#slide-19]], [[03_Planteamiento_Problema.pptx#slide-11]] |
| Order questions from general to specific. | Establish whether the effect exists before analyzing cost, failure modes, or transferability. | [[01__Que_es_Investigacion.pptx#slide-23]], [[03_Planteamiento_Problema.pptx#slide-8]] |
| Use synthesis, not paper-by-paper summary. | When drafting literature review material, connect papers by theme, tension, method, or gap. | [[02_Revision_Bibliografica.pptx#slide-11]] |
| Treat disagreement and failure as useful evidence. | Negative results, weak SFT gains, or unexpected DPO behavior should be framed as findings, not as absence of value. | [[02_Revision_Bibliografica.pptx#slide-13]], [[03_Planteamiento_Problema.pptx#slide-10]] |
| Compare using explicit dimensions. | Use tables for model condition, training method, compile success, correctness, speedup, failure mode, and scope. | [[02_Revision_Bibliografica.pptx#slide-8]], [[02_Revision_Bibliografica.pptx#slide-10]], [[02_Revision_Bibliografica.pptx#slide-14]] |
| Keep the problem neutral. | Do not phrase the project as if SFT/DPO must win; the evaluation should allow improvement, no improvement, or regression. | [[03_Planteamiento_Problema.pptx#slide-5]], [[03_Planteamiento_Problema.pptx#slide-7]] |
| Use quantitative success criteria. | Define compile rate, correctness rate, faster-than-Inductor rate, and speedup distributions before interpreting outcomes. | [[03_Planteamiento_Problema.pptx#slide-11]], [[03_Planteamiento_Problema.pptx#slide-17]] |
| Define specialized terms before relying on them. | Introduce terms such as Triton kernel, TorchInductor, SFT, DPO, correctness check, and speedup before using them as assumptions. | [[04_Marco_Teorico.pptx#slide-2]], [[04_Marco_Teorico.pptx#slide-10]] |
| Separate theoretical framework from literature review. | Framework explains concepts the reader needs; literature review explains who did what and where this project fits. | [[04_Marco_Teorico.pptx#slide-4]], [[04_Marco_Teorico.pptx#slide-6]] |
| Use a funnel from accessible to technical. | Start with parallel computation and kernel motivation, then move toward Triton, model adaptation, DPO, and statistical evaluation. | [[04_Marco_Teorico.pptx#slide-7]], [[04_Marco_Teorico.pptx#slide-9]] |
| Add narrowing statements. | End technical explanations by connecting them back to the research question, so the reader never loses the thread. | [[04_Marco_Teorico.pptx#slide-13]], [[04_Marco_Teorico.pptx#slide-15]] |
| State limitations early and precisely. | Be explicit that SFT/DPO may improve behavior but do not guarantee optimal kernels or eliminate all semantic GPU errors. | [[04_Marco_Teorico.pptx#slide-12]] |

## Canonical Project Framing

This project evaluates whether adapting `qwen2.5-coder:14b` to the Triton-kernel-generation task improves one-shot translation from PyTorch reference functions to Triton kernels.

The adaptation sequence is:

1. Vanilla baseline: `qwen2.5-coder:14b`.
2. Supervised fine-tuning: `qwen2.5-coder:14b-tried`.
3. DPO: label TBD.

The locked held-out evaluation set contains 437 synthetic fusion tasks. The vanilla baseline has already been run and should be treated as the current confirmed baseline. SFT and DPO comparisons are planned or pending until their eval result files exist.

## Evidence Sources

Use these repository files as the source of truth while drafting:

- `docs/decision-log.md`: project bitacora and accepted decisions.
- `docs/corpus.md`: training/eval corpus design and held-out set counts.
- `docs/eval-stats.md`: planned statistical analyses and course-to-analysis map.
- `docs/benchmarking-protocol.md`: correctness and benchmark protocol.
- `docs/tolerance-policy.md`: correctness tolerance rationale.
- `eval/results/qwen2.5-coder:14b-vanilla/eval_rows.jsonl`: confirmed vanilla baseline.

## Document Map

| File | Class Unit | Main Purpose | Must Cover |
|---|---|---|---|
| `01.md` | What Is Research? | Establish the research foundation. | Project context, current problem, why current models struggle, core research question, broader diagnostic questions, hypothesis, why this is research rather than only engineering, scope, baseline, and initial contribution claim. |
| `02.md` | Literature Review | Position the work inside existing research. | Thematic literature buckets: LLM code generation, GPU kernel generation, Triton and compiler baselines, SFT, DPO/preference optimization, grammar or constraint-guided generation, and evaluation methodology. Emphasize synthesis over paper-by-paper summary. |
| `03.md` | Problem Statement | Formalize the measurable problem. | Final problem statement, RQs, H0/H1 hypotheses, metrics, success criteria, vanilla baseline results, SFT/DPO comparison design, and statistical tests. |
| `04.md` | Theoretical Framework | Build the conceptual foundation. | GPU parallelism, kernels, Triton, PyTorch eager execution, TorchInductor, LLM code generation, SFT, DPO, correctness checking, performance benchmarking, and how these concepts connect to the research question. |

## Writing Rules

- Write in English.
- Keep one Markdown file per class unit.
- Use `docs/decision-log.md` as the bitacora for chronology and accepted project decisions.
- Do not claim SFT or DPO improvement until their eval results exist.
- Use vanilla eval numbers only as baseline evidence.
- Prefer clear, precise claims over broad claims about AI or code generation.
- Preserve a path from accessible explanation to technical depth in every major section.

## Global Principle Coverage Checklist

Use this table as the project-wide writing tracker. Coverage should point to the internal `.document` note where the principle is currently addressed. Status values:

- `Not started`: not yet addressed.
- `Drafted`: addressed in a first-pass note, needs review.
- `Partial`: present but missing depth, evidence, or cross-links.
- `Covered`: reviewed and sufficiently represented.

| Principle | Status | Internal Coverage | Notes |
|---|---|---|---|
| The paper is an argument, not a build log. | Drafted | [[01.md#8-why-this-is-research-not-just-engineering]] | `01.md` distinguishes the research claim from the engineering apparatus. |
| Start from the practical problem, then guide the reader toward the research question. | Drafted | [[01.md#1-project-context]], [[01.md#4-core-research-question]] | Current draft moves from GPU motivation to the formal RQ. |
| Keep problem, research questions, and hypothesis distinct. | Drafted | [[01.md#2-research-problem]], [[01.md#5-supporting-research-questions]], [[01.md#6-working-hypothesis]] | Structure is present; review wording for crisp separation. |
| Make claims falsifiable. | Drafted | [[01.md#7-falsifiability-and-success-criteria]] | Needs later update once SFT/DPO results exist. |
| Order questions from general to specific. | Drafted | [[01.md#5-supporting-research-questions]] | Current order goes baseline diagnosis, SFT, DPO, transfer. |
| Use synthesis, not paper-by-paper summary. | Not started | [[02.md]] | To be addressed in the literature review note. |
| Treat disagreement and failure as useful evidence. | Drafted | [[01.md#7-falsifiability-and-success-criteria]] | Also belongs in `03.md` when formalizing negative-result interpretation. |
| Compare using explicit dimensions. | Partial | [[01.md#10-current-baseline]] | Baseline table exists; full comparison tables belong in `03.md`. |
| Keep the problem neutral. | Drafted | [[01.md#6-working-hypothesis]], [[01.md#7-falsifiability-and-success-criteria]] | Current language allows no-improvement/regression outcomes. |
| Use quantitative success criteria. | Partial | [[01.md#7-falsifiability-and-success-criteria]], [[01.md#10-current-baseline]] | Metrics are named; formal H0/H1 and thresholds belong in `03.md`. |
| Define specialized terms before relying on them. | Partial | [[01.md#1-project-context]], [[04.md]] | `01.md` defines kernels lightly; full definitions belong in `04.md`. |
| Separate theoretical framework from literature review. | Not started | [[02.md]], [[04.md]] | To be enforced once those notes exist. |
| Use a funnel from accessible to technical. | Drafted | [[01.md#1-project-context]], [[01.md#3-why-the-current-state-is-insufficient]] | `01.md` starts accessible and becomes more specific. |
| Add narrowing statements. | Partial | [[01.md#3-why-the-current-state-is-insufficient]], [[04.md]] | Some sections reconnect to the RQ; `04.md` should make this systematic. |
| State limitations early and precisely. | Drafted | [[01.md#9-scope-and-boundaries]], [[01.md#11-initial-contribution-claim]] | Current draft states limits and keeps improvement claim provisional. |
