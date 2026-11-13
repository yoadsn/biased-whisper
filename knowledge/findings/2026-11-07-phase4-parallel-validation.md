---
type: Finding
title: Phase 4 parallel validation - eight ranks pass after NVLink-only P2P routing
description: Synthetic frozen-teacher and tiny-Whisper probes validate parallel target generation, DDP update parity, BF16 and encoder LoRA; default four-GPU NCCL routing stalled and NVL restriction resolved the probe.
tags: [finding, phase4, ddp, nccl, a40, validation, synthetic]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T16:10:00Z }
sources:
  - id: implementation
    resource: /reference/phase4-parallel.md
    title: Parallel Phase 4 configuration and lifecycle
---

# Scope

The owner cancelled the original serial Phase 4 run and approved implementation
of the multi-GPU plan. Tests here use synthetic/random checkpoints, synthetic
features/WAVs and isolated temporary data directories. **No production server,
full corpus training, held-out confirmation predictions or new manual correctness
review was launched.** Existing review spend remains 50/100 sentences.

# Communication issue found by testing

With default NCCL routing, one-GPU and NVLink-paired two-GPU probes completed.
The four-rank probe stalled during DDP setup and was terminated by its bounded
test timeout. GPUs were verified clear afterward. Restricting direct P2P to
NVLink (`NCCL_P2P_LEVEL=NVL`) allowed the same four-rank probe to finish.
Subsequent full 1/2/4/8-rank synthetic tests passed with that setting.

The machine reports NVLink pairs 0/3, 1/2, 4/7 and 5/6, not an all-to-all
NVLink fabric. This supports using explicit tested routing, but does not
diagnose the underlying default-route stall conclusively. No system/driver
settings were modified. The harness exposes `nccl_p2p_level`, default NVL.

# Bounded tiny-Whisper measurements

Artifact: `/tmp/opencode/phase4-gpu-validation-nvl-final/summary.json`.
Each FP32 case trains the same 11 synthetic examples in global batches of 8
and 3, with dropout disabled, one epoch and a randomly initialized tiny Whisper.
Tail padding is zero-weighted; no real example is dropped or counted twice.

| GPU ranks | Engine seconds | End-to-end seconds | Peak PyTorch allocation per rank (max) |
|---:|---:|---:|---:|
| 1 | 0.743 | 8.50 | 18,266,112 bytes |
| 2 | 0.637 | 8.86 | 18,266,112 bytes |
| 4 | 0.733 | 10.06 | 18,266,112 bytes |
| 8 | 0.771 | 12.16 | 18,257,920 bytes |

FP32 saved adapters match the one-rank baseline within rtol 3e-4 / atol 3e-5.
The two-GPU case uses physical GPUs 0/3. Additional two-rank tests pass for
BF16 (rtol 0.1 / atol 0.001 versus FP32) and one adapted top encoder layer.
Their engine/end-to-end times are approximately 0.704/9.24 and 0.801/9.28
seconds respectively. Peak allocation excludes CUDA context/driver overhead.

These startup-dominated fixtures **do not demonstrate an eight-GPU production
speedup**. They validate execution and numerical semantics. Eight GPUs remains
the requested machine-utilizing default; real-checkpoint throughput must be
judged from the new stage measurements during authorized runs.

# Actual spawned teacher path

Artifact: `/tmp/opencode/phase4-teacher-spawn/summary.json`. A random tiny
checkpoint with the existing tokenizer and eight short synthetic WAVs exercises
the production parent, dynamic claim queue, all eight teacher processes,
checkpoint loading, Phase 1 decoding/filtering, atomic cache and outcome merge.

All eight outcomes exactly match serial `generate_one` results for verdict,
reason, baseline and target. The random fixture yields zero accepted targets;
the parent fails explicitly before starting DDP, as intended. This is not
evidence about real Phase 1 label quality. Summed worker teacher inference time
is 0.971 s, separate from 0.437 s of audio loading. No output text was manually
reviewed or assigned correctness labels.

# Other gates

CPU/Gloo tests cover numerical parity and bit-exact same-world checkpoint
resume with dropout disabled, tail coverage, compatible cache reuse, corrupt
cache rejection, all-rejected target handling, cross-process GPU leases and
bounded cancellation/failure/stall/shutdown cleanup. UI configuration and
eight-worker monitoring were inspected using explicitly synthetic data on an
isolated server, including switching between parallel and legacy form fields.

The full suite passed 227 tests, one skipped. After the final RNG-state
hardening (including NumPy and per-sample teacher preprocessing seeds), the
targeted Phase 4 suites passed 40 tests, including strict same-world resume.
