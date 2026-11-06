---
type: Finding
title: Phase 2 GPU-native capture benchmark and preserved-capture resume
description: Real 740M-parameter CUDA selection is 12–13x faster than CPU reference selection with validated sparse outputs; original captures preserved and the same 100-point experiment resumed.
tags: [finding, phase2, performance, cuda, benchmark, resume]
status: stable
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T18:30:05Z }
---

# Authorized change and preserved work

Owner approved pausing Phase 2 and fixing GPU-native capture before resuming.
`phase2-20261106-200924-2731` was cooperatively cancelled after **31 complete
points / 62 direction records**. Its vectors and logs were not deleted or modified.

The optimized run is `phase2-20261106-203005-7d37`, with the **same 100-point list,
objective, all_no_embed filter, 1,000,000 per-sample budget, RMS normalization,
linear targets, voting rounds, null and FDR settings**. It uses device capture and
`resume_run=phase2-20261106-200924-2731`. Its `resume.json` confirms all 31 points
were validated/reused; only 69 points need new gradients. Original vectors are
linked/copied into the new run, with provenance-marked point records.

# Real-model benchmark

Registered job: `phase2_benchmark-20261106-202725-6992`; artifact `benchmark.json`.
Local large-v3-turbo, CUDA:0, 739,996,160 selected parameters, same two real Explore
decision points as the paused run. Each comparison uses the identical gradient
for the new and reference selectors; no synthetic timing extrapolation.

| Stage (seconds) | Point 10037 | Point 10032 |
|---|---:|---:|
| Forward/backward | 1.0827 | 0.5452 |
| Device gradient packing | 0.0147 | 0.0145 |
| New device sparse selection/export | 2.4917 | 2.2088 |
| Dense CPU gradient export (reference comparison only) | 1.8248 | 1.7865 |
| CPU reference sparse selection | 29.8318 | 29.7192 |
| Selection-only speedup | 11.97x | 13.46x |

This confirms forward/backward was not the dominant delay; the former dense
CPU postprocessing was. Speedups above are **selection-only**, not end-to-end
throughput or voting speedups. Initial model loading, audio reads and capture
file writes are separate costs.

Both directions on point 10037 had identical selected support. On point 10032,
both directions shared 999,999 of 1,000,000 reference indices (one replacement,
two entries in the symmetric difference). That difference was at a numerically
equivalent cutoff; shared selected values, RMS and energy passed rtol 2e-5 checks.
All benchmark gates passed. This is validated numerical parity, **not bitwise
identity**, and no claim that downstream candidate sets must be bit-identical.

# Implementation and limits

See [GPU capture reference](/reference/phase2-gpu-capture.md). Dense gradients,
per-tensor reductions, scoring and top-k stay on CUDA; linear direction rankings
are shared. Only sparse payloads and per-tensor summaries leave CUDA. Budget and
iterative callers keep their legacy CPU APIs/behavior. New per-point timings are
written to `capture_timings.jsonl`.

The UI server was restarted while no jobs were active to load the new code,
which released the seven unused Phase 1 replicas. Future jobs also evict unused
cache entries automatically while retaining the active model. A live resumed-run
snapshot showed **GPU 0 at 100% utilization, GPUs 1–7 at ~4 MiB each**; this is a
snapshot, not a claimed sustained utilization average. The new controls and
benchmark button were browser-checked without submitting duplicate jobs.

At a 53-new-point resumed-run timing snapshot, mean capture time was **2.9128 s**:
audio 0.0650, forward/backward 0.5558, gradient pack 0.0145, selection/export
2.2135, output writes 0.0622 seconds (remaining difference is orchestration).
These per-point timings exclude initial model loading and reused-point work.
All 31 reused points were verified in `resume.json`, including 51 unchanged
hard-linked vector files; the other 11 direction records were explicitly skipped
as already satisfied, so correctly have no vector file.

Validation at deployment: **65 tests passed, 11 skipped**, including CUDA support/
value parity, shared selection, zero-gradient behavior, preserved-capture resume,
incomplete-capture handling, config mismatch rejection and cache retention.

This is still a **single-GPU, one-sample-at-a-time capture**. Multi-GPU sharding,
gradient microbatching, mixed precision and GPU aggregation were not implemented.
CPU common-set voting/null/FDR may become the next bottleneck, especially with
uncapped common_k. Scientific defaults were deliberately not narrowed for speed.
No Phase 2N, budget or Phase 3 job was started by this optimization.
