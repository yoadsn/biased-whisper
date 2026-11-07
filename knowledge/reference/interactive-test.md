---
type: Reference
title: Interactive Test tab and local audio comparisons
description: Record or upload at most 30 seconds, compare an untouched baseline with any number of independent Phase 3 candidate edits, and reuse a single CUDA model with exact sparse rollback.
tags: [reference, ui, interactive, phase3, inference, audio, cuda]
status: stable
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T20:14:25Z }
---

# User workflow

The **Test** tab selects a completed Phase 3 run, then accepts microphone recording
or an audio file (up to **30 seconds / 25 MB**). Add/remove any number of edit rows,
each selecting an available candidate nudge type and a numeric strength, with an
optional label. The untouched base model is always included. Rows are independent,
not cumulative; duplicate edits/strengths are allowed. An empty row list performs
baseline-only transcription. Zero is unchanged; negative strengths reverse an edit.
Large strengths may degrade output; arbitrary amounts are exploratory, not validated
operating points.

The candidate statistic (`mean_all`, etc.), candidate limit, model identity and
token cap come from the selected Phase 3 run. Model identity resolves from that
run or its recorded Phase 2 provenance, **not today's active-model setting**.
Models without recorded provenance cannot be silently substituted. The UI warns
when a result reaches the selected run's token cap.

# Execution / correctness

`test_transcribe` is a registered **GPU job**, using the same single-worker GPU
queue as scientific jobs. Upload/metadata endpoints never mutate model weights.
CUDA:0 is chosen when available, otherwise the normal auto CPU/MPS fallback.
One model is loaded/reused, with one small set of device-native candidate supports;
no replica or full-model snapshot is made per config.

For each row: apply that row's sparse delta, freshly encode the audio under the
current weights, decode with the Phase 3 tokens-only path, and restore the saved
original values in `finally`. Only CPU mel features survive between rows. Neither
encoder outputs nor KV caches cross edits. Failure/cancellation rolls back the live
edit; a restoration failure invalidates the cached model before another test can
reuse it. Results/timings are appended to the job as each row completes.

Unlike scientific budget/Phase 3 jobs, interactive tests default to
`release_model_after_run=false`, retaining the **restored baseline model** between
comparisons. **Unload model** queues `model_release`; its teardown happens after
other active GPU work, never concurrently with inference. Scientific jobs can also
clear idle cached models according to their lifecycle policy.

# Audio handling / privacy

The browser microphone path uses mono PCM sample counting and automatically stops
at exactly 30 seconds of captured samples, rather than relying on timer/container
duration. Microphone access needs HTTPS or localhost and browser permission. Leaving
the Test tab stops recording and preserves the captured clip through the upload path.

Uploads use a streamed raw audio body, bounded at 25 MB. FFmpeg decodes at most
30.001 seconds to bounded mono PCM; anything over 30 seconds is **rejected**, not
silently cropped. Empty, invalid and undecodable audio is rejected. Local audio
demuxers/protocols are allowlisted; uploads cannot fetch remote playlists.
Normalized 16 kHz PCM WAV and metadata stay under ignored `data/test_audio/<id>/`.
Audio is not added to the corpus, discovery cohort or held-out evaluation. The
localhost-only API/UI deployment is unchanged.

# API and artifacts

- `GET /api/test/runs/<phase3-run>`: model and edit metadata, no GPU loading.
- `POST /api/test/audio?filename=...`: raw audio upload; returns id/duration/playback URL.
- `GET /api/test/audio/<id>` and `/info`: normalized playback and metadata.
- `POST /api/test/transcribe`: `{phase3_run, audio_id, edit_configs:[{edit,strength,label}]}`;
  validates the selection and queues `test_transcribe`.
- `model_release`: normal registered job, exposed through the existing jobs API.

Each comparison lives in `data/runs/test_transcribe-.../`, with `job.json`,
`results.jsonl`, `summary.json` and timing fields. The Phase 3 experiment is not
rerun; only one supplied audio sample is transcribed under each selected edit.

# Validation on the turbo server

Unit/API coverage includes decoded duration enforcement at exactly 30 and over
30 seconds, invalid/oversized uploads, MP3/WebM/M4A normalization, run provenance,
finite edit validation, one model for all rows, independent repeated/zero edits,
and exact full-space rollback on inference failure or cancellation. Final suite:
**126 passed, 11 skipped** (one upstream Starlette TestClient deprecation warning).

Real-browser validation used Chrome with a **fake microphone fed discovery audio**;
the user's physical microphone was never accessed. It verified file upload,
short recording (~1.02 s), automatic recording stop and successful normalized upload
at exactly 30.00 s, rejection of a 31-second file without replacing valid audio,
four independently configured edits plus baseline, baseline-only mode, repeated
comparisons, queued unloading and reload. Zero-strength and repeated edits matched
their expected base/duplicate transcripts on the real CUDA model. Dropdown values
were checked against edit state, with no page errors or 390px mobile overflow.

First real comparison `test_transcribe-20261106-221209-4012` (2.98-second discovery
clip, baseline + four configs) took 25.78 s including cold model startup; the next
comparison took 1.56 s. Subsequent warm jobs stayed around 1.55–1.60 s server wall
time; single-config inference was ~0.24 s. These are sample-specific measurements,
not a guaranteed latency for every 30-second recording or arbitrary config count.

`model_release-20261106-221619-a66f` released one model and left 8.52 MB allocated /
44.04 MB reserved CUDA workspaces. A final baseline-only job,
`test_transcribe-20261106-221620-a6ee`, reloaded it and reproduced the original base
transcript; the restored model is intentionally left warm for the owner's next
interactive test. Browser diagnostic results/screenshots are under
`/tmp/opencode/test-tab-browser-results.json`, `test-tab-desktop.png` and
`test-tab-mobile.png`. Diagnostic audio/results are local ignored data, not a new
scientific evaluation.
