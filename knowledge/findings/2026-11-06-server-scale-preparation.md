---
type: Finding
title: Server-scale preparation and original-digit availability
description: Full aligned-manifest inventory, CUDA compatibility fix and preparation for large-v3-turbo; no experimental phases executed.
tags: [finding, corpus, server, a40, whisper-large-v3-turbo]
status: deprecated
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T15:44:56Z }
---

# Correction: text inventory below was NOT pre-normalization

The original-digit availability claims in this finding are **invalid**. The first
importer incorrectly assumed the prepared manifest's `transcript` was original.
It is TTS-normalized, including number expansion. See the superseding
[text-provenance correction](/findings/2026-11-06-original-text-correction.md).
Hardware/model checks below remain valid; corpus text/coverage must be repaired
and cohorts regenerated before experiments. In particular, crowd-transcribe does
have original digit references, and 5,799 is not a valid original-digit ceiling.

# Initial inventory (normalized transcript only; superseded)

Scanned all aligned train/valid manifests, not a sample. No English or recital
source was included, and no normalized text was consulted.

| Source | Rows | Original transcript contains digits | Hebrew + 1.5–28 s bounds | Distinct audio paths among those bounds |
|---|---:|---:|---:|---:|
| plenum | 1,560,905 | 902 | 900 | 531 |
| committee | 1,999,926 | 5,866 | 5,863 | 2,244 |
| crowd_transcribe_v5 | 112,620 | 0 | 0 | 0 |
| Total | 3,673,451 | 6,768 | 6,763 | 2,775 |

These are **pre-import upper bounds**, before duplicate removal, alignment-error
filtering, path existence and audio decode checks. The requested 10,000 local
original-digit examples are not present under these constraints. Do not count
digits in filenames/speaker metadata or in `alt_words.norms` as reference digits.
Crowd-transcribe has no original-digit cohort; adding its number-word examples
would be a distinct cohort/design decision requiring owner approval.

Some digit-bearing aligned rows use `alignment_backend=mms` with
`alignment_fallback_reason="digits are unsupported by Viter"`. This is a valid
fallback, not by itself an alignment error. Nonempty `alignment_error` rows are
excluded from the importer even when a time range is present.

# Environment and architecture

Eight NVIDIA A40 GPUs are visible, each with 46,068 MiB memory. Initial unrestricted
dependency resolution installed torch 2.15.0+cu132, which cannot initialize on the
server's CUDA-12.9-capable driver. Linux torch is now pinned to 2.8.x from the
CUDA 12.8 index; torch 2.8.0+cu128 successfully initializes CUDA and sees all eight
GPUs. Transformers is pinned below 5 to preserve the harness's 4.x decoder API.

The local checkpoint config describes a **32-layer encoder, 4-layer decoder,
1,280 hidden width, 128 mel bins**, vocabulary 51,866. The 3.236 GB safetensors
file is available locally. Do not carry over whisper-small layer indices or
candidate weights: the turbo decoder is structurally different.

# Completed import and preflight

Import run: `corpus_import_pockettts-20261106-174337-03de`.
Final clean corpus: **5,799 clips**, **17.15 hours**, **2,391 recordings**:

- plenum: 737 clips.
- committee: 5,062 clips.
- crowd-transcribe: 0 original-digit clips.

Of the 6,763 duration-eligible references, 964 carry nonempty alignment errors
and were excluded. No duplicate or crop failures occurred in this import. The
10,000-sample local target therefore falls short by 4,201 under the quality rules.
Every retained reference is unaltered original text; no normalized forms were used.

Preflight run: `setup_scale-20261106-174945-c276`.
All eight GPUs passed arithmetic checks. The real local checkpoint, tokenizer and
feature extractor loaded successfully on `cuda:0`, then were unloaded without
transcription. Verified parameter count: **808,878,080**; 427 digit-bearing tokens.

Frozen split: 4,716 discovery clips (1,951 recordings) and 1,083 evaluation clips
(440 recordings), with zero recording overlap. Per-source discovery/evaluation:
plenum 620/117, committee 4,096/966. Twenty sampled stored WAV headers matched
their database durations. Both run statuses are `done`; no phase/budget jobs exist.

The ready-but-unsubmitted config is
`data/runs/setup_scale-20261106-174945-c276/phase1.json`. Its sibling files are
`cohorts.json`, `phase3.template.json`, `explore-fetch.json`, and `preflight.json`.
Settings point at the local turbo checkpoint on `cuda:0`. GPU free memory returned
to its preflight baseline after the process exited.

Validation: 36 tests passed, 11 skipped (uncached whisper-tiny-dependent tests in
offline mode); OKF lint passed. Browser-checked the setup and importer forms using
an isolated temporary data directory, then closed the test tab and stopped the
test server. No persistent web server was started.

# Execution gate and remaining preparation

See [A40 server preparation](/reference/server-setup.md) for import/preflight jobs,
eight-device Phase 1 configuration and recording-disjoint cohort artifacts.
Explore fetching targets 10,000 new examples, but this fresh checkout does not
contain the laptop's authentication cookie. It must be configured securely in
`data/secrets/explore.json` through the existing UI (never committed).

No Phase 1, Phase 1.5, budget, Phase 2/2N or Phase 3 job has been authorized or
started. Multi-GPU Phase 1 is prepared and unit-tested with mocked sample analysis;
large-model decode parity/throughput remain gated on execution approval.
