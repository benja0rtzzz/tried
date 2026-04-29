# Tolerance Policy

> **Locked file.** Do not make changes. Implementation: `packages/shared/src/shared/verification/tolerance.py`.

## Rationale

A single shared tolerance policy enforces consistency across all correctness checks. Hardcoding `atol`/`rtol` at call sites would make it impossible to audit what threshold was in effect for any given attempt. Every correctness check records the `tolerance_policy_used` key so any result can be reproduced.

## Policy table

Records store only the `tolerance_policy_used` key. The implementation is the authoritative source for the numeric thresholds and the comparison mode attached to that key.

| Policy key | `atol` | `rtol` | Comparison mode | Intended for |
|---|---:|---:|---|---|
| `default_fp32` | `1e-5` | `1e-5` | `numeric` | General float32 ops |
| `default_fp16` | `1e-3` | `1e-3` | `numeric` | Half-precision ops |
| `reduction_fp32` | `1e-4` | `1e-4` | `numeric` | Sum/mean reductions in float32 with accumulated error |
| `reduction_fp16` | `5e-3` | `5e-3` | `numeric` | Sum/mean reductions in fp16/bf16 |
| `exact_integer` | `0.0` | `0.0` | `exact` | Integer, bool, packed-code, and index/scatter outputs |
| `masked_logits` | `1e-5` | `1e-5` | `inf_aware_numeric` | Logits or attention-like outputs with `±inf` masks |
| `attention_softmax_fp16` | `5e-3` | `5e-3` | `numeric` | fp16/bf16 softmax/logsumexp attention paths where operation ordering differs from Inductor |
| `recurrent_scan_fp16` | `1e-2` | `1e-2` | `numeric` | Selective scan, SSD chunk scan, and SSM state-update recurrences with many multiply-add/exp steps |
| `low_precision_dequant` | `2e-2` | `2e-2` | `numeric` | int4/NF4/codebook/FP8 dequantized float outputs |
| `fp8_cast` | `1e-2` | `1e-2` | `numeric` | Outputs produced through FP8 cast/decast |

## Comparison Modes

`numeric` is the standard tolerance check:

```python
abs(candidate - reference) <= atol + rtol * abs(reference)
```

`exact` requires exact dtype and exact value equality. It is intended for integer, bool, index, packed-code, and other discrete outputs where numeric tolerance is the wrong abstraction.

`inf_aware_numeric` is a numeric tolerance check with an explicit non-finite pre-pass:

- same-position `+inf` vs `+inf` passes;
- same-position `-inf` vs `-inf` passes;
- finite vs non-finite fails;
- `+inf` vs `-inf` fails;
- `nan` fails unless a future policy explicitly opts into NaN equality;
- finite-only positions then use the usual numeric tolerance and populate the recorded diff stats.

The comparison mode is not stored in dataset records. It is derived from `tolerance_policy_used` through `tolerance.py`.

## What NOT to do

- Do not pass `atol`/`rtol` directly anywhere outside `tolerance.py`.
- Do not add a new policy key without updating both this document and the Enum, and without team agreement.
- Do not change values once data collection has started.
