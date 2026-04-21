# Tolerance Policy

> **Locked file.** Do not make changes. Implementation: `packages/shared/src/shared/verification/tolerance.py`.

## Rationale

A single shared tolerance policy enforces consistency across all correctness checks. Hardcoding `atol`/`rtol` at call sites would make it impossible to audit what threshold was in effect for any given attempt. Every correctness check records the `tolerance_policy_used` key so any result can be reproduced.

## Policy table

Policies are keyed by `op_category`. The implementation is the authoritative source; this document explains the intent.

| Policy key | `atol` | `rtol` | Intended for |
|---|---|---|---|
| `default_fp32` | TBD | TBD | General float32 ops |
| `default_fp16` | TBD | TBD | Half-precision ops |
| `reduction` | TBD | TBD | Sum/mean reductions (accumulated error) |

> Values are TBD — to be filled in by the team before the first data collection run.

## What NOT to do

- Do not pass `atol`/`rtol` directly anywhere outside `tolerance.py`.
- Do not add a new policy key without updating both this document and the Enum, and without team agreement.
- Do not change values once data collection has started.
