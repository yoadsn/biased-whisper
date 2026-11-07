---
type: Finding
title: Phase 3 batched CUDA inference and encoder-edit correctness
description: Pre-run diagnosis followed by owner-authorized batched tokens-only decoding, feature reuse with condition-local encoding, batched probes, cleanup and real-model parity benchmarking.
tags: [finding, phase3, performance, cuda, decoding, batching]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T19:49:50Z }
---

# Scope and evidence

Owner requested an efficiency check before starting Phase 3. No Phase 3 job was
submitted, no evaluation outputs generated, and no runtime settings/code changed.
This is a code-level assessment, not a measured Phase 3 utilization or speedup
benchmark. `graft` remains unavailable (`command not found`); reviewed source
directly. No active jobs were present during the check.

Settings resolve to `cuda:0` and the local ivrit-ai large-v3-turbo checkpoint.
`phase3.py:63` loads through `model.load_model`, which uses these settings unless
an explicit model override is given. The CUDA:0 snapshot showed 30,317 MiB resident
and 0% utilization while idle; residency is not active work or evidence of useful
GPU utilization. Other GPUs were idle.

# Likely bottlenecks

- `phase3.py:119-149` iterates conditions and utterances serially, calling
  `M.greedy` for one sample at a time. No inference batch-size control exists.
- `model.py:220-238` runs a Python loop per generated token, obtains a host argmax
  token and digit probability, converts the selected log-probability to a Python
  float, then constructs the next GPU input tensor from the host token. These
  device-to-host scalar reads impose synchronization points.
- Phase 3 uses only `r.tokens` (`phase3.py:145-146`), but `M.greedy` computes
  log-softmax, digit probability and `Step` diagnostics every token. These are
  valuable to Phase 1, but unused by Phase 3 transcription metrics.
- `phase3.py:139-142,155-157` correctly invalidates encoder-output reuse when an
  edit touches the encoder. The latest Phase 2 artifacts contain 40,447 encoder
  weights among 50,000 to_digits candidates and 41,278 among 50,000 to_cardinal
  candidates. Thus both full candidate sets (and their shuffled-value controls)
  require fresh encoding for each condition; random-index conditions determine
  this from their sampled support. Candidate limits may change the distribution.
- Audio waveforms are loaded once, which is good, but `M.encode` calls CPU
  feature extraction every time (`model.py:144-151`). Mel features could be
  precomputed once without reusing stale encoder outputs.
- Probe evaluation is also serial (up to 30 points per condition), with a scalar
  GPU-to-host conversion per point. For sequence contrast it runs separate
  teacher-forced digit and cardinal forwards (`phase2.py:64-76`).
- Each sample opens/appends outputs and saves progress to disk
  (`phase3.py:147-149`, `jobs.py:55-74,98-102`). Likely secondary, not measured.

# Improvements already inherited / legitimate work

The recently optimized sorted-slice `ParamSpace.add_sparse` is already used by
Phase 3 (`phase3.py:132`); it no longer makes a full support mask per tensor.
Edits happen once per condition and are undone in a finally block. There is no
dense gradient export, full-model ranking, backprop, or huge budget-control
shuffle in this path. Decoder KV caching and inference mode are already enabled
in `M.greedy`; this is not uncached prefix decoding.

At defaults (two nudge types, four strengths, two controls), there are 25
conditions, 1,250 utterance decodes for 50 evaluation samples, and up to 750
additional probe objectives. Autoregressive steps within each utterance remain
dependent; the independent utterances within a condition supply the batching
opportunity. Running different edits concurrently on the same model is unsafe.

# Recommendation before a large run

1. Add a Phase-3-specific tokens-only decoder with true same-condition batching,
   per-row EOS handling, and the existing forced-prefix/suppression semantics.
   Retain the scalar diagnostic decoder for Phase 1. Validate token/text and
   condition-metric parity at unchanged precision, including edited weights.
2. Precompute mel features; batch fresh encoder inference for each encoder-edit
   condition. Only cache encoder outputs when the actual support leaves encoder
   weights unchanged. Bound cache/batch memory rather than silently changing scope.
3. Add a registered small baseline/nudge/control benchmark with synchronized
   feature/encoder/decoder/probe/edit/I/O timings, throughput, peak allocated GPU
   memory and exact restoration checks. Tune batch size from measurements, not
   an assumed A40 optimum. No numerical speedup or utilization estimate is
   supported by this static review alone.
4. Batch compatible probe objectives, throttle progress saves, and add explicit
   resume if needed. Existing per-sample outputs are useful but Phase 3 has no
   implemented resume path or stage timings; progress excludes probe work.

Do not silently switch precision, reduce evaluation samples, strengths, controls
or token limits. The owner has not yet authorized implementation or Phase 3.

# Separate evaluation-cohort caveat

The default pool excludes only selected Phase 2 sample IDs, not every discovery
sample or all related recordings (`phase3.py:70-100`). For the prepared
recording-disjoint evaluation, pass its explicit `eval_sample_ids`; the default
pool is not a guarantee of that stronger holdout. This is a validity issue,
separate from GPU efficiency.

# Owner-authorized implementation

The owner subsequently authorized optimization and cleanup of the completed budget
model cache. An idle localhost-server restart released the retained GPU working
set (all GPUs subsequently reported 0 MiB in nvidia-smi). Future budget sweeps and
benchmarks now default to `release_model_after_run=true`. GPU-worker cleanup runs
after job frames have unwound, including failures/cancellation; clears model
references, collects garbage, empties allocators on cached CUDA devices, and
records allocated/reserved bytes in `model_cleanup.json`. Phase 3 and its benchmark
use the same lifecycle policy. Terminal job state is saved only after cleanup.

`phase3_inference.py` adds a CPU feature bank, true batched tokens-only decoding
with per-row EOS and unchanged prefix/suppression/log-softmax precision, and
batched sequence-contrast probes. Probe scoring projects only divergent positions;
right-padded positions never enter span sums. Other legacy probe objectives remain
supported. Each condition encodes fresh batches under the currently edited model;
no encoder output or KV cache crosses a condition boundary, even for controls.

`phase3_device.py` adds the batched runner and `phase3_benchmark`. The original
runner remains available as an explicit reference. Candidate ordering, random
control sampling, strengths, sample ordering, metrics and token limits are
preserved. Default batch size is 8, exposed in the UI with backend selection and
a discovery-only GPU benchmark button. Progress includes probes, with synchronized
per-stage timings and peak CUDA allocation recorded. Phase 3 is not resumable yet.

Local tests: **102 passed, 11 skipped**, including random-Whisper decoder/probe
parity, mixed per-row EOS and suppression, full condition/metric parity with encoder
edits, feature reuse with encoder recomputation, exception/cancellation rollback,
and cleanup after success/failure/cancellation frames are gone. JavaScript syntax
and diff whitespace checks pass.

Real turbo/CUDA benchmark `phase3_benchmark-20261106-215624-8265` uses eight Phase 2
discovery samples, both 50,000-weight candidate sets at strength 2, both controls,
96-token limit, and batches 1/4/8. The frozen recording-held-out evaluation is not
used. Measured parity/throughput results follow after completion; no full Phase 3
experiment is authorized or submitted by this implementation work.

## First real-model result (strength 2)

All **70 checks passed**: exact token sequences and decoded metrics for all seven
baseline/nudge/control conditions at batch sizes 1/4/8, probe agreement (maximum
absolute error 0.0000782), and exact full-space restoration after every condition.
The benchmark keeps a full GPU weight snapshot solely for these restoration checks.

Measured stage-time sums across seven conditions, eight decodes and eight probes
per condition (excluding load/preparation/edit/restoration overhead):

| Path | Encoder seconds | Decoder seconds | Probe seconds | Total measured inference seconds |
|---|---:|---:|---:|---:|
| Serial reference | 9.796 | 4.049 | 11.038 | 25.288 |
| Tokens-only, batch 1 | 9.386 | 3.603 | 10.575 | 23.973 |
| Tokens-only, batch 4 | 9.235 | 1.398 | 9.621 | 20.674 |
| Tokens-only, batch 8 | 8.968 | 0.904 | 9.350 | 19.619 |

Batch 8 delivered **4.48x decoder speedup and 1.29x measured inference speedup**.
Encoder inference is now dominant: batching does not imply an equally large
whole-experiment speedup when encoder computation already saturates the GPU.
An eight-second activity sample during the benchmark recorded SM utilization
77–100% (short live sample, not a per-stage or full-run utilization average).
Largest batch-8 allocation was 7,108,117,504 bytes including the restoration
snapshot; no precision setting changed. Cleanup recorded one model released,
9,097,216 allocated bytes and 16,777,216 reserved bytes afterward. Small runtime/
library workspaces and the CUDA context can remain; the multi-GB model is gone.

## Large-budget cleanup check

`budget_benchmark-20261106-215842-81f8` re-exercised the original smallest/largest
grid endpoints, passed all restoration/finite-energy checks, and peaked at
22,937,285,632 allocated bytes. The common post-job cleanup released its model:
`model_cleanup.json` records **17,616,896 allocated / 37,748,736 reserved bytes**
afterward, rather than leaving ~23–30 GB cached. No budget sweep was repeated.

A second discovery-only parity benchmark,
`phase3_benchmark-20261106-215900-23b9`, checks batch 8 at the remaining default
strengths 5/10/20 against the scalar reference before final handoff.

It also **passed all 76 checks**, including exact tokens/metrics and full-weight
restoration, with maximum absolute probe error **0.0001116** and 7,117,489,152-byte
peak allocation including the restoration snapshot. Thus both directions, both
controls, and all default strengths 2/5/10/20 have been checked on discovery audio
(146 total checks across the two benchmarks); these are not a proof of universal
bitwise equality for arbitrary near-tied floating-point logits. Precision is
unchanged and objective comparisons allow small batching/reduction roundoff.

Final cleanup again released the model, leaving 17.6 MB allocated / 37.7 MB
reserved. nvidia-smi reported **375 MiB on CUDA:0, 0% activity**, consistent with
remaining context/library overhead rather than the prior ~30 GB model cache;
other GPUs reported 4 MiB each. No active jobs remain. The localhost UI/API has
the validated implementation loaded; refresh the UI to see the new controls.
No full Phase 3 run was submitted. Final lint: 25 concepts, zero problems;
102 tests passed / 11 skipped. `graft build` could not run because the executable
is unavailable.
