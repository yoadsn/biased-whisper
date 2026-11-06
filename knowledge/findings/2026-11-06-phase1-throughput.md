---
type: Finding
title: Phase 1 GPU underutilization and sample-budget review
description: Live telemetry confirms low GPU utilization with eight threaded single-sample replicas; batching is feasible but needs a parity-tested decoder, and the existing decision-point pool is already large.
tags: [finding, performance, phase1, batching, sampling]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T16:48:18Z }
---

# Observation

Owner reported very low GPU utilization during `phase1-20261106-181106-3926`.
An eight-second `nvidia-smi dmon` sample showed GPU SM utilization mostly zero,
with observed peaks up to 9%; each device held about 3,731–3,733 MiB. The UI process
averaged 114% CPU in a separate five-second pidstat sample (about 1.14 CPU cores).
The run does execute on CUDA, but does not use the devices efficiently.

At a results-file snapshot it had 4,300 completed samples in about 37 minutes,
including model/replica initialization (~1.94 samples/s overall):

- cardinal: 2,876; no_digits: 972; no_number_words: 300.
- partial: 82; degenerate: 62; no_divergence: 8; sample errors: 0.
- Sources: 610 plenum, 3,690 committee, no Explore results yet in this increment.

The earlier 32-sample validation is separate. Explore download has now reached
10,000 stored clips successfully (resume job added 9,980 in 877.05 s); local corpus
is also 10,000. Download/data size does not require decoding every item in Phase 1.

# Implementation, not a CUDA-placement failure

`src/bwhisper/phase1.py:247-252` uses eight threads, one model replica and one
static strided sample shard per GPU. `:292-297` still analyzes exactly one sample
per replica at a time. There is no sample batch dimension in the explicit decoder.

`src/bwhisper/model.py:203-221` runs a Python autoregressive loop with a decoder
call per token, repeated scalar GPU-to-host conversions for argmax, digit mass and
log-probability, and new token tensor construction. `:127-129` extracts features
per sample on the host before transfer. Shared interpreter/GIL contention and
small synchronous decoder steps are plausible host-side bottlenecks consistent
with the telemetry; they have not yet been separated by a detailed profiler.
Do not claim a measured per-component time breakdown or a guaranteed speedup.

# Batching is feasible

Encoder inputs can be stacked, and baseline decoding can use a true batch with
per-row EOS handling. Suppressed continuations additionally need per-row rewind
positions, prefixes, digit/satellite masks and token budgets. Grouping compatible
prefix lengths or teacher-forcing prefixes within a batched scheduler can preserve
the per-sample semantics. This is not accomplished by merely setting a batch_size
flag or replacing the whole custom path with unmodified generate(). A candidate
batched implementation needs token/verdict parity checks against the serial path
and a benchmark; 8–16 samples/device is a trial range, not a validated optimum.
One process per GPU would also avoid the shared GIL, but must not instantiate
independent job managers that mark each other's active jobs interrupted.

# Budget recommendation, not yet an execution change

The existing 2,876 automatic cardinal decision points already exceed the needs of
an initial existence proof / small attribution fit. They are not human-verified
or numerically value-checked. Prioritize a few hundred representative, reviewed
pairs, add a small recording-discovery Explore cohort for domain coverage, and
retain the large recording-held-out corpus for later evaluation. Laptop results
do not prove the large checkpoint behaves identically, but thousands more local
examples are unnecessary just to establish existence on turbo.

Recommended next decision: stop the large current run while retaining its partial
results; disable the automatic full-Explore Phase 1 increment; process only a
small balanced Explore discovery sample. No cancellation, budget reduction,
batching implementation or phase advancement has been authorized/applied in this
review; ask the owner before changing those running/queued actions.

Subsequent update: the previously scheduled Phase 1 increments have now all
completed (16,112 unique discovery samples), before any budget reduction was
approved. There is no longer a running Phase 1 job to stop. The smaller-budget
and batching recommendations now apply to future work; see
[completed turbo results](/findings/2026-11-06-turbo-phase1-validation.md).
