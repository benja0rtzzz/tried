# Benchmarking Protocol

> **Locked file.** Do not make changes.

## Verification steps (in order)

1. Run eager PyTorch on input → reference output.
2. Run `torch.compile(fn, backend="inductor")(input)` → Inductor output.
3. **Sanity check:** eager and Inductor must agree within tolerance. If they don't, the harness is broken — abort, do not blame the candidate.
4. Run candidate Triton → candidate output.
5. Compare candidate vs eager (primary) and candidate vs Inductor (secondary). Record all 10 correctness stats.
6. If correctness passes, run benchmarks.

## Benchmark rules

- **Warmup:** 10 runs before timing.
- **Timing:** 100 runs, report the **median**.
- **Timer:** `torch.cuda.Event` only. No wall clock (`time.time`, `perf_counter`).
- **Sync:** `torch.cuda.synchronize()` before starting and after the final run.
- **Reported metrics:** speedup vs eager and speedup vs Inductor. Absolute ms is recorded but is not the primary metric.

## Hardware pinning (Lenovo)

```bash
CUDA_VISIBLE_DEVICES=0  # always set before launching the server
```

Verify with `nvidia-smi` that the RTX 4060 is the active device before a benchmark run. The Ryzen iGPU will silently capture work otherwise.

- Do not benchmark on battery power.
- Do not run other GPU workloads concurrently.
- The server process should be the only CUDA process during a benchmark window.

## Tolerance

Tolerance values are defined in `packages/shared/src/shared/verification/tolerance.py`. See @docs/tolerance-policy.md for rationale and the policy table.
