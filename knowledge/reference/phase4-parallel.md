---
type: Reference
title: Phase 4 parallel teacher cache and DDP training
description: Eight-GPU Phase 4 orchestration, content-addressed teacher outcomes, global-batch-preserving DDP, resumable checkpoints, single-writer monitoring and validated NCCL routing.
tags: [reference, phase4, distributed, ddp, lora, cache, monitoring]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T16:00:00Z }
---

# Launch configuration

`target_style=cardinals` is the backward-compatible default. For a **digit**
adapter, select `target_style=digits`, keep parallel mode enabled, and set
`teacher_source_run` to a completed cardinal Phase 4 run with a complete teacher
cache. Start from the untouched base with `resume_run` empty, not from the
opposite adapter. New run IDs include direction (`phase4-digits-...` or
`phase4-cardinals-...`); configuration, manifest, monitor and checkpoint metadata
record it. Historical runs lacking the field are cardinal runs. Test and run
pickers display the direction explicitly and reject checkpoint/run mismatches.

Digit training derives targets only from the source run's accepted
`suppressed_cardinal` pairs: the cached baseline text (already containing digits)
is re-tokenized without special tokens and must round-trip exactly, contain
digits, fit decoder context and pass degeneration/token checks. Naturally
cardinal examples are excluded; accepted no-number preservation targets can be
retained. Rejected source examples stay rejected. There is no new teacher
inference and no automatic corpus expansion. This is full-target supervised
LoRA, not a contrastive/reversal objective; an individual target can still contain
mixed numeric styles if the cached base did.

Reuse is explicit provenance, not a claim that old generation code matches new
code. The source must be completed, cardinal, and match the split/order/seed,
generation cap, selection filters, language/timestamps and base/tokenizer content.
Source cache identity and each outcome checksum are verified, and current audio/
metadata identities must match. A distinct direction-specific cache records the
source teacher fingerprint and derivation-code identity. Original cache entries
and the cardinal run are never rewritten. Different objectives cannot silently
share targets or optimizer resume state.

Phase 4 now defaults to **parallel=true**, eight devices `cuda:0` through
`cuda:7`, and **global_batch_size=8**. Keep `resume_run` empty for a fresh
student run. Compatible completed teacher outcomes are reused automatically.
`parallel=false` retains the original one-device online path and its
`gradient_accumulation` setting.

The user cancelled the original single-GPU run and explicitly approved this
implementation. Do not start the production server or full experiment on the
user's behalf: the user will launch them after the harness is ready. No new
manual correctness review is authorized by implementation; cumulative spend
remains **50/100 sentences**, in ten-sentence packets.

```bash
uv run bw serve --port 8765
# UI: Phase 4 -> parallel=true, devices=cuda:0,...,cuda:7,
# global_batch_size=8, epochs=1..3, resume_run empty -> Start.
```

For two GPUs, an NVLink pair such as `cuda:0,cuda:3` is a sensible starting
point. World size must divide the global batch; with batch 8, supported counts
include 1/2/4/8. Rank-local microbatch stays one, with accumulation 8/4/2/1
respectively. LR and warmup remain in global optimizer updates. Parallel mode
ignores legacy `device` and `gradient_accumulation`; the UI hides them.

## NCCL routing on this A40 server

The server has NVLink **pairs**, not a fully connected NVLink fabric. Synthetic
NCCL validation completed on one GPU and a two-GPU NVLink pair, but the
four-GPU run stalled during DDP setup with default P2P routing. Restricting
direct P2P to NVLink (`NCCL_P2P_LEVEL=NVL`) resolved that test; subsequent
1/2/4/8-GPU runs completed with matching FP32 adapter updates.

`nccl_p2p_level=NVL` is therefore the explicit, tested default here. Other
links use NCCL's remaining transports. `auto` defers to NCCL/environment
defaults; other supported levels are available as advanced controls. This is
an observed configuration workaround, not a proven diagnosis of a hardware
fault. No driver, kernel or system-wide configuration was changed.

# One parent job, two child stages

`phase4.py` dispatches to `phase4_parallel.py`. The normal API/CLI job is the
only Job Manager. Parent subprocesses exec fresh interpreters; they never fork
live CUDA state into Python workers, instantiate Manager, or create top-level
child jobs. `jobs.gpu_reservation` provides cross-process advisory device leases
for harness GPU jobs; outside programs that ignore those leases are not managed.
The parent clears its old inference cache before starting GPU workers.

## 1. Independent teachers

One worker per device runs the **frozen base**, no LoRA installed, same eval
precision and Phase 1 span rules as the serial path. A transactional job-local
SQLite claim queue dynamically balances sample IDs; each pending ID is claimed
once. Rank-owned outcome histories and atomic cache entries retain successes
and rejections. Failure is explicit; it is not converted into a silent skip.

The live SQLite database and journals reside on local disk under
`/tmp/opencode/phase4-tasks-*`, not on the NFS-mounted run directory.
`teacher/tasks.db` links to that local file while workers run; after supervised
workers exit it is replaced with a diagnostic snapshot and local scratch is
removed. Completed cache outcomes remain the recovery authority. This avoids
shared SQLite write locking over NFS; merely increasing the lock timeout is not
the remedy. Queue errors roll back their transaction before propagating.

Cache location: `data/phase4_cache/<teacher-fingerprint>/`. Identity includes
checkpoint/tokenizer/config content hashes, language/timestamp settings, seed,
generation/filter settings, relevant source and library versions, and per-sample
audio/metadata identity. Payload checksums detect corrupted entries. Preparation
is protected by a cache lock. Missing entries are regenerated; incompatible or
corrupt existing entries fail clearly rather than silently becoming labels.

Old online runs lack the full new cache provenance and are **not imported
automatically**. Their files remain untouched. A new run may therefore regenerate
the original cancelled run's small teacher prefix. This is preferable to claiming
unsafe cache equivalence. New parallel runs can reuse compatible cached outcomes
after cancellation or across different student hyperparameters/world sizes.

After all teachers exit, the parent validates complete coverage, merges outcomes
in frozen cohort order and freezes accepted IDs before any distributed backward.
The original cohort/confirmation exclusions, natural cardinal/preservation
examples, optional numeric filter and heuristic label caveats remain unchanged.

## 2. DDP students

`python -m torch.distributed.run` starts one rank per device; each loads the
full frozen base, installs identical LoRA and wraps the student in DDP/NCCL.
Only trainable adapter parameters have gradient reduction. The initial DDP
setup still synchronizes model parameters, so startup is not free.

All ranks have equal numbers of microsteps and optimizer updates. Each global
batch is sharded across ranks; final unused slots run zero-weight dummy forwards
so no real sample is dropped or counted twice. Non-final microsteps use
`no_sync`. Loss is the mean of per-clip mean-token cross entropies, matching the
serial objective. Scaling accounts for DDP's gradient averaging and actual
global tail size. Clip after synchronization/normalization, then update once.

Optional top encoder LoRA uses fresh features/encoder computation; no adapted
encoder outputs are reused across updates. CPU threads per process are bounded
(`cpu_threads_per_worker=2` by default). More advanced batching/prefetch and
FSDP/model parallelism are not part of this version.

# Monitoring, cancellation and failure

Before GPU workers start, the parent hashes checkpoint contents and every selected
training/evaluation audio file on CPU to establish cache identity. The monitor's
`fingerprinting` stage reports `fingerprinted_samples`; the UI shows this against
`train_samples + eval_samples`. Epoch/update/seen counters remain zero here, and
GPUs are expected to be idle. The first owner-run preparation took approximately
four minutes for 16,162 samples; this is not a guaranteed duration. Teacher model
loading follows, then target counters advance; student training starts only after
all teacher outcomes have been collected.

Worker-owned `teacher/rank-N.json` and `distributed/rank-N.json` snapshots
include independent heartbeat, last progress time, stage, counts and allocated/
peak GPU memory. Teacher counters distinguish inference time from audio load;
student counters separate feature/audio preparation, forward/backward and update
wall time. Rank 0 alone writes distributed scalar/evaluation/checkpoint histories;
the parent imports complete lines and alone writes top-level monitor/job files.

The UI shows cache hits/misses, teacher and accepted-target throughput, global
student clips/s and updates/s, per-rank health/memory, global mean loss and LR.
Student throughput includes startup/development-evaluation overhead and resets
its numerator correctly on resume. It is not an idealized kernel throughput.

Cancellation creates a stop request. Teachers finish the current sample and
preserve completed outcomes. Students stop at a synchronized update boundary and
save an atomic resumable checkpoint. Evaluation notices cancellation between
clips and all ranks reach the next coordinated boundary. Process exceptions,
startup stalls and progress/collective timeouts fail the parent; it terminates
the process group and any remembered launcher descendants with bounded fallback.
Workers also exit if their parent process disappears. No infinite barrier waits
are intended. `worker_timeout_seconds=900`; unusually long evaluation/loading
may require a larger timeout. Failed runs preserve already complete checkpoints.

# Checkpoints and resume

Each committed `update-step-XXXXXX`, `epoch-NN-step-XXXXXX` or cancellation
checkpoint contains adapter-only safetensors/config plus `training_state.pt`:
optimizer state, global step, next epoch/batch cursor, per-rank Python/NumPy/Torch/CUDA
RNG, world size and a strict training fingerprint. The parent run's `resume.json`
points to the latest complete directory. `.partial` directories are never
advertised as complete. Old serial checkpoints remain inference-only.

For training resume, set `resume_run` to an **inactive parallel Phase 4 run with
a committed training checkpoint**, keep epochs/hyperparameters/global batch/world
size/data identical, and launch a new job. Inputs, accepted target content, base,
adapter code and relevant training settings must match; incompatible resumes
are rejected. This version deliberately does not support changing world size,
changing the epoch horizon or resuming old serial optimizer state. To reuse only
teacher work (including a run stopped before student training), leave resume_run
empty and let the content-addressed cache match automatically.

The existing checkpoint Reload & evaluate action accepts the new checkpoint
directory names. Evaluation is rank-0-only using the unwrapped model while other
ranks wait; it is unsuppressed and reports raw prevalence/quality counts on the
same fixed development cohort. It is not new independent confirmation evidence.

In the **Phase 4** tab, select a run and click **View base vs adapter** beside
an epoch/update in the monitoring table. The browser displays that pass's saved
`evaluation.jsonl` rows, base and adapter text side by side, with corpus audio
playback. This reads existing predictions, not a new inference/evaluation job.
Epoch 0 is the initialization comparison, not the trained result; select the
final epoch/update to judge the trained adapter. The raw artifact is
`data/runs/<run>/evaluation.jsonl` (one row per sample per monitoring pass).
The **Test** tab also supports independently loading any complete checkpoint
from Phase 4 runs for owner-provided microphone/upload audio; see
[interactive testing](/reference/interactive-test.md).

## Interpreting the first completed epoch

Owner-run `phase4-20261107-173859-1395` completed one epoch over 13,445 accepted
clips (1,681 global optimizer updates). On its reused 50-clip monitoring cohort,
clips containing digits changed from 37/50 in the base to 1/50 in the final
adapter, with no detected degeneration, truncation or empty output. This is
evidence of strong style preference learning, not checked number-value accuracy
or content preservation. The base already represents Hebrew cardinal forms;
LoRA distills a preference for those forms rather than learning ASR from scratch.
One epoch was configured, not selected by validated early stopping. There is no
epoch-2/3 comparison or within-epoch evaluation curve for this run. More epochs
may improve difficult cases or reinforce heuristic teacher mistakes; review
content/numeric correctness and use a consistent development comparison before
claiming an optimal training duration. Protect final confirmation from tuning.

# Validation and performance interpretation

Tests cover cache identity/checksums, dynamic claims, cache reuse, empty target
sets, equal rank slots and correct global tails, subprocess failure/cancellation/
stall bounds, cross-process GPU leases, CPU DDP parity and bit-exact same-world
resume with dropout disabled. Existing tiny-Whisper tests cover base restoration,
LoRA updates/reload and inference.

Bounded **synthetic tiny-Whisper** GPU probes validate NCCL 1/2/4/8 ranks,
FP32 update equivalence (rtol 3e-4 / atol 3e-5), a BF16 two-rank run (comparison
rtol 0.1 / atol 0.001), and two-rank top-encoder LoRA. Benchmark artifacts are
local under `/tmp/opencode/phase4-gpu-validation-nvl-final/`. These probes are
not corpus training or ASR-quality checks. Their startup-dominated timings do
**not** establish a production speedup or prove eight GPUs optimal for the real
checkpoint. Eight GPUs is the requested machine-utilizing default; the new stage
timings should guide any later production tuning.

The actual spawned teacher pipeline also passes an eight-synthetic-WAV check
against serial Phase 1 generation using a random tiny checkpoint. The all-rejected
fixture explicitly stops before DDP. See the
[validation finding](/findings/2026-11-07-phase4-parallel-validation.md) for exact
scope, benchmark values, memory measurements and remaining interpretation limits.
