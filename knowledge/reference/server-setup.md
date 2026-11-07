---
type: Reference
title: A40 server preparation
description: Reproducible large-v3-turbo environment, aligned-manifest import, multi-GPU Phase 1 configuration and recording-held-out cohorts; setup never launches experimental phases.
tags: [reference, server, cuda, corpus, reproducibility]
status: stable
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T15:44:56Z }
---

# Scope and execution gate

The server has eight A40 GPUs (46,068 MiB each). The checkpoint is local:
`/mlspeech/data/yoadsnapir/models/hf/ivrit-ai/whisper-large-v3-turbo`.
Prepare data, validate the environment and write configs first. **Do not launch Phase 1,
Phase 1.5, budget sweeps, Phase 2/2N or Phase 3 until the owner explicitly authorizes it.**

Authorization update (2026-11-06): the owner has approved **Phase 1 only**. A
32-sample corrected-discovery validation run passed on eight GPUs. The corrected
10,000-clip local corpus and 10,000-clip Explore corpus are complete. All three
disjoint Phase 1 increments have finished: 16,112 discovery samples, zero runtime
errors; 3,888 recording-held-out evaluation samples remain untouched. See
[completed turbo results](/findings/2026-11-06-turbo-phase1-validation.md).
This does not authorize Phase 1.5, budget or later phases.

Later authorization update: the owner subsequently ran Phase 1.5, authorized the
Phase 2/budget optimizations and explicitly requested Phase 3. The first prepared
50-clip held-out Phase 3 evaluation is now complete; see
[turbo results](/findings/2026-11-06-turbo-phase3-results.md). This historical setup
gate does not prevent the specifically authorized work; additional scientific
experiments still require their own scope/approval.

# Environment

```bash
uv sync --python 3.13
```

Linux torch uses the explicit CUDA 12.8 wheel index, with torch 2.8.x. The server
driver supports CUDA 12.9, not the CUDA 13.2 runtime selected by unconstrained
PyPI torch. Transformers stays on the 4.x API used by this harness. Dependencies
are isolated in `.venv`, resolved by `uv.lock`; no driver or global Python changes.

# Local corpus job

```bash
uv run bw run corpus_import_pockettts '{"limit":10000,"workers":8,"seed":0}'
```

Reads `train_aligned.jsonl` and `valid_aligned.jsonl` under
`/mlspeech/hebrew/pockettts-data/{plenum,committee,crowd_transcribe_v5}` only.
The prepared `transcript` field is **already TTS-normalized**, so must not be
filtered directly. Restore the original Knesset text using inclusive inverse spans
in `alt_words.norms[*].replace` (the producer stores original text under this
misleading label). For older crowd manifests, join `clip` to original `text` in
the cached `meta/*-index.json.gz`. Filter only the recovered original for digits
and Hebrew. No text normalizer is called and no numeric references are invented.
See the [provenance correction](/findings/2026-11-06-original-text-correction.md).
Use aligned `start` / `duration`, not `original_start` / `original_duration`.
Rows with alignment errors or invalid/nonfinite bounds are excluded; MMS
alignment fallback is accepted (digit strings are unsupported by the Viter aligner).
Duration range: 1.5–28 seconds. Audio is cropped to local 16 kHz mono WAVs.

Selection is seeded and balanced across available sources, with backfill when a
source is exhausted. Deduplicate by source + original audio path + aligned range,
including duplicate train/valid rows and existing corpus entries. Reservoirs are
bounded at twice the target per source. Crop failures use replacement candidates.
The target is **new additions**; a later run repairs any version-1 references,
skips existing audio slices and adds more. Repaired references carry version 2
and explicit original-text provenance in metadata.
The result reports actual additions and any shortfall rather than fabricating data.
No English or recital manifests are read. `inventory.json`, `samples.jsonl` and
optional `errors.jsonl` live in the job run directory.

# Explore corpus

```bash
uv run bw run corpus_fetch_explore '{"target":10000,"per_seed":500,"per_episode":6,"workers":8,"mode":"range","seeds":[234608111]}'
```

Requires an authenticated Explore configuration in `data/secrets/explore.json`,
configured through the existing corpus UI. See [Corpus sources](/reference/corpus-sources.md).
The laptop's cookie/corpus is **not** automatically transferred to the server.
Authentication is checked before fetching. Do not put cookies in git or job configs.
This job targets 10,000 **new** samples, using episode-level diversity and existing
Range extraction. `per_episode` caps are shared across seeds and include existing
corpus samples, so changing shuffle seeds cannot keep filling the same recordings.
It reports whether the target was reached (at most 100 seeds).

# Preflight and frozen cohorts

```bash
uv run bw run setup_scale '{}'
```

Registered job + overview UI form. Runs a small CUDA arithmetic check on each
requested device, loads the checkpoint/tokenizer/feature extractor on the first
device without transcription, records architecture/parameter count, then unloads.
Sets persistent active/inference model and device to the checkpoint and `cuda:0`.
Does not change the generic laptop defaults in source.

For imported digit samples, SHA-256(seed, source, recording identity) assigns
roughly 80% to discovery and 20% to evaluation. All slices of a recording share
one assignment. Local recording identity = large WAV path; crowd = clip parent;
Explore = episode UUID. This prevents within-source recording leakage, not
unverified content duplicates across different datasets. Non-digit crowd samples
are not silently admitted to satisfy the digit target.

Artifacts under `data/runs/setup_scale-.../`:

- `preflight.json`: hardware, checkpoint and cohort counts.
- `cohorts.json`: frozen IDs, recording assignments and per-source counts.
- `phase1.json`: discovery IDs, local checkpoint, eight explicit CUDA devices,
  192-token decode cap for the longer aligned clips.
- `phase3.template.json`: held-out evaluation IDs, conservative 50-sample eval cap,
  192-token decode cap (Phase 2 run still needs filling).
- `explore-fetch.json`: 10,000-sample fetch configuration, no credentials.

**Re-run setup after adding Explore samples**; old files remain frozen and never
silently expand to include new corpus entries. No phase job is submitted by setup.

# GPU execution boundaries

Phase 1 accepts `devices: ["cuda:0", ..., "cuda:7"]`: one model replica and one
disjoint sample shard per device, shared job/results file with locked writes and
combined progress/summary. Empty `devices` preserves single-device behavior.
Replicas use threads (CUDA operations release the GIL); this avoids independent
job-manager processes marking each other's jobs interrupted. This path has unit
tests with mocked analysis; actual large-model decode throughput/parity must be
validated **after** the Phase 1 execution gate is lifted.

Phase 2/2N, budget and Phase 3 are **still single-device**. No claim of distributed
attribution/aggregation support. Do not scale Phase 2's `max_points` straight to
10,000: sparse captures are ~16 MB/point for both directions at `sample_k=1e6`,
before overhead; the existing 8 GB guard should remain active until budgeted.
Phase 3 now uses bounded same-condition batches (default 8), with CPU mel-feature
reuse but **no encoder outputs cached across edits**. Its full held-out template
remains a cohort definition, not authorization to consume every held-out sample;
the first owner-authorized run used the prepared `eval_limit=50`.
Large-model edit strengths and budgets require calibration; laptop findings are
not assumed to transfer unchanged.
