---
type: Reference
title: GPU-native Phase 2 capture, benchmarking and resume
description: Keep full gradients and linear-nudge ranking on CUDA, export sparse vectors only, benchmark against the CPU reference, and resume complete stopped captures without modifying their originals.
tags: [reference, phase2, gpu, performance, resume]
status: stable
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T18:27:25Z }
---

# Execution controls

Phase 2 defaults now include:

- `capture_backend="auto"`: device-native capture for linear CUDA jobs; CPU/MPS
  and iterative nudges retain the reference path. `device` explicitly chooses the
  new linear implementation (also useful for small CPU tests); `reference` keeps
  the old dense CPU implementation. No silent precision or parameter-filter change.
- `resume_run=null`: optionally reuse complete captures from a stopped Phase 2 run.
- `release_unused_models=true`: evict cached models other than the active model,
  without unloading the active model or its parameter space.

This first optimization remains **one sample on one GPU**. It does not claim
multi-GPU point sharding, per-example-gradient batching, AMP/TF32, or GPU voting.
GPU-native capture addresses the measured dense CPU postprocessing bottleneck first.

# Data path and semantics

`ParamSpace.flat_grad(device=...)` can pack the gradient on its model device;
its default remains CPU for compatibility with budget/iterative callers.
One backward supplies both linear directions. Compute per-tensor gradient energy
and RMS on the device, score there using scalar per-tensor normalization, and run
one device top-k. Both nonzero linear nudges are scalar multiples of the same
gradient, so their absolute/RMS-normalized rankings coincide mathematically.
Reuse that support; scale only its selected values and the small RMS/energy arrays.
Never materialize/export two dense CPU nudge vectors or a repeated P-element RMS
array. Transfer indices and selected values plus per-tensor summaries only.

Skipped already-satisfied directions retain the old metadata. Zero-gradient/zero-
scale nudges have all-tied scores; the new implementation shares a valid support
and emits zero values/energy with RMS fallback 1. CPU and CUDA reductions/rank
ties are not promised to be bit-identical. Sparse files retain the old schema:
`idx`, `val`, `rms`, `energy`, allowing unchanged aggregation and Phase 3 consumers.

Each newly captured point writes synchronized stage timings to
`capture_timings.jsonl`: audio, forward/backward, device gradient packing,
selection/export, output writes and total. Summary timing means exclude reused
points. The scientific capture objective, selected parameters, point list, k,
voting, null/FDR rules and output format are unchanged.

# Registered benchmark job

The Phase 2 UI has a **Benchmark GPU capture (2 points)** button. CLI/API type:
`phase2_benchmark`, with the same Phase 2 science config plus `benchmark_points=2`.
It computes the real gradient once per point on CUDA, compares the new sparse
selection with the CPU reference on that identical gradient, and records:

- forward/backward, gradient-pack, device-selection, CPU export and CPU reference times;
- support overlap and changed indices;
- matching selected values, per-tensor RMS and energy;
- whether differences lie only at a numerically equivalent selection boundary.

Benchmark artifacts: `benchmark.jsonl`, `benchmark.json`, normal job state/logs.
Pass criteria: support overlap >= 0.9999, common values and RMS/energy close at
rtol 2e-5 (small absolute tolerances), and changed-support reference scores within
2e-5 relative of the reference threshold. It reports any non-identical membership;
do not call it bitwise parity or hide the numerical boundary differences.
The benchmark deliberately exercises slow dense CPU selection for comparison.

# Resuming preserved captures

Set `resume_run` to the cancelled/interrupted/completed Phase 2 run ID and keep
the scientific config identical. Before reuse, check source job type/status,
configuration, parameter-space signature, model identity from Phase 1 points,
both direction records, and readable sparse arrays of the expected dimensions,
sorted unique in-range indices and finite/nonnegative statistics. Incomplete or
invalid points are recaptured, not treated as complete.

A complete point includes both directions, including an explicit skipped record
when already satisfied. Hard-link its existing vector files into the new run
(copy only if linking is unavailable), append provenance-marked point records,
and restore energy accumulation. Original run/files are not modified or deleted.
The new run records `resume.json`, `reused_from` on reused point rows, backend and
reused counts. Parameter-configuration mismatches fail rather than silently mixing
incompatible captures. Aggregation is then recomputed over all captures together.

# Known remaining bottleneck

Aggregation/voting/null statistics remain NumPy/CPU, and `common_k=null` can leave
a large common set. Low GPU utilization during that stage is expected; the UI
labels it separately from capture. Do not change common_k, rounds, filter or
precision merely for speed without evaluating the scientific consequences.
