---
type: Reference
title: Harness architecture
description: Modules, job system, data layout, API and CLI of the biased-whisper harness.
tags: [reference, architecture, harness]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:40:00Z }
resource: src/bwhisper
---

# Layout

## Phase 4 adapter training

Independent of discovery/edit phases: `phase4.py` registers `phase4` online
frozen-teacher training and `phase4_evaluate` checkpoint reload/evaluation jobs.
`lora.py` implements adapter-only safetensors and reversible module installation.
`phase4_parallel.py` owns fingerprinted teacher caches and subprocess lifecycle;
`phase4_worker.py` runs independent teacher replicas or torchrun/DDP students.
All harness GPU jobs use advisory device reservations; only the parent owns a
Job Manager and top-level monitoring/artifact writes. See
[parallel training](/reference/phase4-parallel.md) for NCCL routing and resume.
The Phase 4 tab polls an atomic monitor snapshot plus bounded scalar-history
tails; no training targets are exposed by monitoring. See
[Phase 4](/plan/phase-4-lora.md) for split protection, defaults, artifacts and limits.

```
src/bwhisper/
  config.py        paths, settings.json, phase default hyperparameters
  model.py         load Whisper, digit-token set, explicit greedy decoder (parity-tested against generate)
  hebrew_numbers.py number-word lexicon (prefix stripping), digit runs, word diff, auto verdicts
  audio.py         ffmpeg decode/crop -> 16 kHz mono
  corpus/          store.py (SQLite), hf_cache.py, explore.py, pockettts.py (aligned local manifests), jobs.py
  setup_scale.py   hardware/model preflight, frozen recording cohorts, configs only (no phase submission)
  jobs.py          job registry/runner; run dir = data/runs/<id>/{job.json,log.txt,*.jsonl,...}
  phase1.py        Phase 1 (v1 first-digit rewind, v2 span mode: StyleTokens, divergence region, paired transcripts)
  phase1_5.py      co-occurrence mining around digit spans, edit patterns, satellites.json
  phase2.py        legacy weight-voting nudges; point_objective (mass | pair | seq_contrast | auto) shared by all phases
  phase2_capture.py device-native legacy sparse selection, synchronized timings, registered CPU/GPU benchmark job
  phase2n.py       default component attribution (neurons, heads, residual), BH-FDR + voting, translation to weight edits
  budget.py        budget sweep (energy, self effect, held-out transfer vs k)
  phase3.py        control experiments (applies any candidates_<type>.npz)
  phase3_device.py same-condition batched control experiments and discovery-only reference-parity benchmark
  phase3_inference.py CPU mel bank, tokens-only batched cached decoder, batched sequence-contrast probes
  phase1_latin.py  Latin-axis Phase 1 (`phase1_latin` job); `latin_script.py` holds the transliteration screen and content patterns; UI tab `latin1`
  interactive.py   Test-tab audio validation, Phase 3 edit provenance, cached-model comparisons and queued release
  paramspace.py    stable flat addressing of a parameter subset, sparse edit/undo
  aggregate.py     counts -> common set -> voting -> permutation null -> FDR (pure numpy, unit-tested)
  api.py           FastAPI: settings, corpus, jobs, phase views, knowledge
  cli.py           `bw serve | corpus-import | run <type> <json> | jobs | job-types`
web/               index.html, app.js, style.css (Alpine.js from CDN, no bundler; RTL for Hebrew)
knowledge/         this OKF bundle
tests/             decoder parity, number lexicon, aggregation statistics
data/              gitignored: corpus.db, audio, runs, secrets, settings.json
```

# Jobs

`@register(name, kind, defaults)` adds a job type. `kind='gpu'` jobs run one at a time, `'io'` jobs on a separate pool.
Each run writes `job.json` (state), `log.txt`, and phase outputs; the UI polls `/api/jobs` and reads the same files, so a run started
from the CLI shows up in the UI and vice versa. Job state survives restarts (active jobs are marked `interrupted`).

The [Test tab](/reference/interactive-test.md) submits `test_transcribe` to the same
GPU queue, never edits a model inside an HTTP handler, and keeps the restored model
cached between comparisons. `model_release` explicitly tears it down in that queue.
Uploaded/recorded audio lives outside the corpus in ignored `data/test_audio/`.

Budget sweeps/benchmarks and Phase 3/its benchmark default to
`release_model_after_run=true`: the GPU worker clears model caches, collects
garbage and empties CUDA allocator buffers after the job's call/exception frames
have unwound, then saves `model_cleanup.json` and terminal job state. This also
runs after cooperative cancellation or failure. Unloading inside a job's own
finally would leave local tensors alive and is insufficient.

Phase 1 can use explicit `devices` for one replica/shard per GPU inside its single
registered job; all other GPU phases still use one device. See [server setup](/reference/server-setup.md).
Phase 2 linear CUDA capture now keeps dense gradients/selection on device and can
resume compatible stopped captures; aggregation is still CPU-only. See
[GPU capture and benchmarking](/reference/phase2-gpu-capture.md).

# Decoder

`model.greedy` is a cached greedy decoder with explicit control of the forced prefix, ban sets, per-step bans (`step_ban`) and
rewinds. `model.seq_logprobs` gives teacher-forced log-probs for the sequence-contrast objective. Tests assert it
matches `WhisperForConditionalGeneration.generate` token for token, and that rewinding reproduces the full decode.
Whisper's `suppress_tokens` are applied; digits are never in that list.

# API (selection)

`GET/POST /api/settings`, `/api/corpus*`, `/api/explore/*`, `GET/POST /api/jobs`, `/api/jobs/{id}/{log,rows,file/<name>.json}`,
`POST /api/phase1/{id}/verdict`, `/api/phase2/{id}/candidates`, `/api/knowledge*`.

# Environment

Python 3.13 via `uv`; torch picks CUDA > MPS > CPU (`device` setting). `ffmpeg` must be on PATH. Hugging Face models load from the local cache first.
