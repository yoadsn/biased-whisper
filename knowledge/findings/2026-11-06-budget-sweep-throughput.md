---
type: Finding
title: Budget sweep device optimization and unchanged-grid validation
description: Legacy CPU bottleneck diagnosis, owner-authorized device-native ranking and sparse-edit replacement, and real-grid benchmark validation.
tags: [finding, performance, budget-sweep, cuda, gradients]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T19:16:41Z }
---

# Active run and observation

Owner requested an efficiency check on `budget_sweep-20261106-211412-59e7`.
Configuration: turbo checkpoint, all_no_embed (739,996,160 weights), auto sequence
contrast, RMS ranking, 24 train / 12 test points, ten fractional budgets through
0.3 of the entire parameter space. The inspection left the job unchanged.

At the snapshot (~2.4 minutes after start), progress was still `train 1/24`, 0/36,
with no completed training-point log yet. Six one-second GPU samples showed GPU 0
at 0% SM activity, ~14,519 MiB resident; GPUs 1–7 were free (~4 MiB each). Resident
memory can include cached allocator buffers from earlier GPU work, not active
computation. A five-second pidstat sample averaged ~102% process CPU (roughly one
core), many minor page faults/system CPU cycles, and RSS rose from ~24.1 to ~33.0
million KiB. These are snapshots, not an exact function-level profiler breakdown.

# Code-level bottlenecks

1. `budget.py:117,147,162` calls `_eval_grad` without `gradient_device`.
   `phase2.py:80-92` deliberately retains `gradient_device="cpu"` for old callers.
   Thus every gradient is exported as a ~2.96 GB CPU vector. The Phase 2 fix was a
   dedicated sparse-capture path, not a global migration of all callers.
2. `budget.py:119-123` squares that entire CPU vector, fully sorts all 740M values,
   converts sorted values to float64 and cumulatively sums them. Full-size values,
   indices, doubles, scores and sort workspace cause substantial memory traffic.
3. `budget.py:124-133` runs a separate full-vector CPU top-k for each of ten budgets;
   consensus selection repeats ten more top-k operations (`:139-140`). These
   nested supports could be obtained from a shared largest-budget ranking, with
   tie/index-order parity explicitly handled.
4. `ParamSpace.add_sparse` (`paramspace.py:135-146`) maps each index to a tensor,
   then makes a full `tid == t` mask for every touched tensor. With sorted indices,
   contiguous slices/searchsorted boundaries could avoid these repeated scans.
   Large supports incur O(k * number-of-touched-tensors) boolean scanning plus
   host-to-device index/value transfers, for every edit application.

The largest budget is **221,998,848 weights**, not one million: ~0.888 GB values
and ~1.776 GB int64 indices before temporary arrays/undo buffers. If an edit touches
all 584 tensors, its current masking loop examines ~129.65 billion index positions.
This is an upper bound from the algorithm, not a measured per-edit operation count.

# Genuine experiment cost versus avoidable overhead

With this grid, the planned sweep does **60 backward passes** (24 initial train,
24 calibration recomputations, 12 test), plus **480 perturbed objective evaluations**
(24 x 10 self effects, 12 x 10 real and 12 x 10 shuffled transfers). The progress
denominator of 36 does not expose this nested work or the calibration pass.

Those evaluations are part of what the sweep measures. Because the current filter
edits encoder weights, encoder states cannot be cached across different edits
without changing the result. Calibration gradient recomputation is a deliberate
memory/compute tradeoff (rather than storing 24 full dense gradients), not the
first bottleneck to optimize. Parallel edits also require separate model replicas;
they must not mutate one shared model concurrently.

# Initial recommendation (subsequently authorized)

Pause/restart rather than let the legacy implementation process this grid:

- Keep gradients, energy reductions, consensus and rankings on the model device.
- Compute one appropriate max-budget ranking/prefix-energy curve, avoiding the
  full CPU sort and ten independent selection scans; sort prefix indices where
  required to preserve existing shuffle/control ordering.
- Apply/undo sorted support slices without per-tensor full-array masks; keep
  indices/values on device where possible.
- Benchmark actual edit evaluation and the largest requested budget with memory
  guards before assuming the A40 can hold every dense workspace simultaneously.
- Add per-stage/per-budget timings, accurate progress, cancellation checks inside
  budget loops, and persisted partial metrics. The current sweep holds curves in
  memory until its final summary and cannot generally resume completed curves.

Consider an explicitly approved smaller absolute-k grid: the current default's
30%-of-model endpoint probes 222M weights, which may be irrelevant to a small-edit
goal. That is a scientific scope change, not something to silently truncate for
speed. No cancellation, configuration change or code optimization was made during
the initial check. The owner subsequently approved pausing and optimizing it.

# Authorized implementation and validation

The legacy job was stopped and all its artifacts retained. It had not persisted
any partial curve, so it cannot resume. The new `budget_device.py` backend keeps
gradients, consensus and ranking on the model device; derives all supports from one
largest-budget ranking; and applies sorted per-tensor slices with guaranteed undo.
No fractions, precision, train/test points or objective were reduced. NumPy control
shuffle ordering is retained, including selected-value CPU transfers.

Each completed budget/calibration measurement is journaled, with stage timing,
accurate nested progress and cancellation checks. Compatible stopped device runs
can resume using consensus and control RNG checkpoints. The UI exposes backend
selection and a registered smallest/largest-grid benchmark job.

Validation before deployment: **81 passed, 11 skipped**, including full toy-sweep
metric parity, CUDA support selection, interruption/resume, control RNG ordering,
and exact rollback after evaluation/edit failures. The real-grid benchmark is
`budget_benchmark-20261106-213235-c809` passed on CUDA:0 with the real turbo model
and original grid. Energy was finite/monotone, objective results finite, and every
selected model weight was restored exactly after both endpoint edits/controls.

| Stage | Measured seconds |
|---|---:|
| Forward/backward | 1.091 |
| Device gradient packing | 0.015 |
| Raw-energy curve (one largest-budget top-k) | 2.350 |
| RMS support ranking (one largest-budget top-k) | 2.348 |
| k=740 support sort / edit-eval-undo | 0.022 / 0.199 |
| k=221,998,848 support sort / edit-eval-undo | 0.119 / 0.248 |
| Largest NumPy control shuffle plus transfers | 10.330 |
| Largest shuffled edit-eval-undo | 0.249 |

Peak CUDA allocation was **22,937,035,776 bytes** (22.94 GB), including a full
consensus-sized reserve and an exact full-weight GPU snapshot for restoration
checks. This validates the real endpoints, not a CPU-versus-GPU full-sweep speedup
claim. The preserved NumPy shuffle is now a visible large-budget CPU bottleneck.

The full unchanged 24-train/12-test, ten-budget sweep was restarted as
`budget_sweep-20261106-213326-bff5`, with explicit device backend and provenance
`restart_of=budget_sweep-20261106-211412-59e7`. It does not claim to resume the
uncheckpointed legacy run. Event-driven completion monitoring is installed; no
Phase 3 or automatic subsequent scientific job is scheduled. Multi-GPU sharding
and gradient batching remain unimplemented.

Startup verification reached train point 5/24 with 40 persisted budget records;
the fourth point's largest support took 0.119 s to sort and 0.274 s for the self
edit/evaluation/undo. All original config fields were checked unchanged. Final
local checks after adding benchmark coverage: **82 passed, 11 skipped**, JavaScript
syntax and diff whitespace checks clean; OKF lint: 24 concepts, zero problems.

# Full sweep completion

`budget_sweep-20261106-213326-bff5` completed successfully in **467.87 s (7.80 min)**,
542/542 progress units and **384 checkpoint records** (240 train-budget, 24
calibration, 120 test-budget). No runtime error; no subsequent phase was started.

Mean held-out cardinal-directed log-odds changes were already -2.009 at k=740,
-2.003 at k=74,000, and -1.848 at k=221,998,848; corresponding shuffled controls
were +0.0421, +0.0140, and -0.0000835. Thus this 12-point held-out sample does not
show a mean-transfer benefit from the huge supports, despite RMS-selected gradient
energy increasing from 0.79% to 97.22%. Smallest-support effects vary more across
points (standard deviation 1.93 versus 0.53 at the largest support). These are
objective changes, not generated-transcription/WER validation; edits are calibrated
to a predicted training change of -3 and are not equal-norm across budgets.

Recorded NumPy shuffle/transfer time totals **190.18 s**, ~40.6% of total wall time.
Train self-edit evaluation/undo totals 54.82 s; held-out real/control evaluation
totals 27.02/27.20 s. These stage totals exclude other work and should not be
mistaken for an exhaustive wall-time profile. The legacy run never completed,
so a measured full-sweep before/after speedup is unavailable.

# Post-job model cleanup

The owner later observed that the completed sweep's cached model/allocator buffers
still occupied ~30 GB on CUDA:0. The retained working set was cleared with an
idle-server restart. Budget sweeps and `budget_benchmark` now default to
`release_model_after_run=true`: the common GPU worker unloads models and empties
allocator caches after job-local tensors and exception frames have unwound,
including success, cancellation and failure. Cleanup is recorded in
`model_cleanup.json` before terminal job state is saved. This execution-only
setting is excluded from resume-plan science-config compatibility checks.

Real diagnostic `budget_benchmark-20261106-215842-81f8` repeated the original
smallest/largest endpoints, passed correctness checks, peaked at 22,937,285,632
allocated bytes, and released its model afterward: **17,616,896 allocated /
37,748,736 reserved bytes** remained as small runtime/library workspaces. CUDA
context overhead may still appear in nvidia-smi; the multi-GB model/working set
is no longer retained. No scientific sweep was repeated for this check.
