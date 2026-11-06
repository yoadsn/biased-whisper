---
type: Finding
title: Phase 2 weight-space capture is dominated by CPU-side dense processing
description: The user-started Phase 2 job backpropagates on cuda:0 but exports all 740 million gradients to CPU and ranks dense nudges there; the other seven GPU models are cached Phase 1 replicas.
tags: [finding, performance, phase2, gradients, gpu, batching]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T18:15:22Z }
---

# Active job and scope

The owner reports completing Phase 1.5 and starting Phase 2 through the UI.
Verified `phase1_5-20261106-200747-48a7` is done: 16,112 samples, 10,940 valid pairs,
17,583 digit spans, five before/two after satellites. Active job is **weight-space
`phase2`**, not component `phase2n`:
`phase2-20261106-200924-2731`, using 100 accepted points from
`phase1-20261106-192356-6310` (Explore continuation), objective auto -> seq_contrast,
linear nudges, all_no_embed, RMS normalization, sample_k=1,000,000, rounds=2,000,
common_pct=30, common_k=null. Selected parameter count: **739,996,160**, 584 tensors.
No job was paused, reconfigured or restarted during this read-only review.

# What actually runs where

- `phase2.py:149-151,173-181` loads one model using the active device (`cuda:0`)
  and processes one decision point at a time; it has no batching or multi-GPU dispatch.
- `phase2.py:60-84` performs one encoder forward and two teacher-forced decoder
  forwards for the sequence contrast, then one backward pass on CUDA. This is
  not Phase 1's autoregressive generation loop. The backward is shared by both
  linear nudge directions, not repeated for each direction.
- `paramspace.py:81-83` immediately detaches, flattens and copies **every** selected
  gradient tensor to CPU, then concatenates them on CPU. One FP32 vector is
  **2,959,984,640 bytes (~2.96 GB)** before further temporaries.
- `phase2.py:87-124` computes the norm and materializes up to two full dense nudge
  vectors on CPU. `:133-144` performs per-tensor RMS reductions, creates another
  full-size `np.repeat(rms, sizes)` array, dense score arrays, and CPU torch.topk
  over all 740 million entries, separately for each needed direction.
- Only the selected million indices/values per direction are saved (~8 MB after
  uint32 index conversion). Exporting all gradients before selection is avoidable.
- `aggregate.py` is entirely NumPy/CPU. Voting, null and candidate statistics will
  not activate the GPUs after capture; an uncapped common set can be very large.

The seven other models are still held in `model._cache`, keyed by model/device
from Phase 1. They are not Phase 2 workers. They consume memory without helping
the current job; GPU residency is not evidence of active GPU computation.

# Live evidence and limits

A six-second dmon sample showed all eight GPU SM counters at zero. GPU 0 held
~8,833 MiB, while GPUs 1–7 held ~3,731–3,733 MiB each. A separate five-second pidstat
sample showed heavy host allocation/page faults and system CPU activity, with RSS
growing from ~10.8 to ~19.6 GB. These are snapshots, not a complete CUDA profile.

For the first 15 captured points, excluding initial setup, log-interval medians:

- one non-skipped nudge: **15 s** (six intervals, range 14–15 s).
- two non-skipped nudges: **27 s** (eight intervals, range 26–28 s).

Since both cases use one shared backward, this strongly implicates direction-wise
dense CPU postprocessing. Exact forward/backward/copy/RMS/top-k/I/O timing shares
have **not** been instrumented. Do not present a guessed percentage or guaranteed
speedup as measured. The CPU top-k implementation may also allocate substantial
workspace; no exact workspace size was established in this review.

# Why normal training is faster and how to improve this implementation

Normal minibatch training usually computes one summed/averaged gradient of the
batch loss, keeps it on GPU, and updates weights. This method needs each sample's
gradient/support/signs separately for agreement voting. Averaging samples first
would change the experiment, but does not force the present serial CPU design.

Preserving the algorithm, prioritize:

1. Keep gradients, norm/RMS, scoring and top-k on CUDA; transfer sparse outputs only.
2. Reuse support/ranking for the two **linear** directions: each is a scalar multiple
   of the same gradient, and absolute/RMS-normalized rankings coincide mathematically.
   Handle skipped/zero scales and numerical ties explicitly; test parity.
3. Shard independent points across GPU processes without DDP gradient averaging or
   independent job-manager instances. Then evaluate per-sample-gradient microbatches
   (`torch.func`/batched VJPs) if worthwhile. Naively retaining a B x P Jacobian is
   expensive: B=16 would need ~47.36 GB for FP32 gradients alone, before weights or
   activations, so ordinary large training batches are not a drop-in solution.
4. Instrument synchronized CUDA timings and CPU timings on a few points before
   selecting batch sizes, precision, parameter scope or hardware counts.
5. Release unused cached replicas safely, without unloading the active Phase 2 model.

Component-space `phase2n` avoids ranking hundreds of millions of weight entries,
but is a different scientific analysis, not an invisible performance switch.
Likewise narrowing the parameter filter or capping common_k changes the search
space and needs an explicit experimental decision. Optimization is recommended,
but no such code/execution changes were applied in this investigation.

Subsequently the owner approved the optimization. See
[GPU-native capture benchmark](/findings/2026-11-06-phase2-gpu-capture.md): CPU
selection ~29.8 s versus CUDA 2.2–2.5 s, parity gates passed, 31 captures preserved
and reused in the same resumed 100-point experiment. This supersedes the
"not yet applied" status above; multi-GPU/batching/aggregation remain future work.
