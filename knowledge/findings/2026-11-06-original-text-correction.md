---
type: Finding
title: Correction of original-text provenance in PocketTTS import
description: Prepared transcript fields are TTS-normalized; restore Knesset originals from inverse replacement spans and join crowd clips to their original source index before filtering digits.
tags: [finding, corpus, correction, provenance, normalization]
status: stable
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T15:58:00Z }
---

# Initial assumption was wrong

The first importer filtered `row.transcript` directly. It did not call a
normalizer, but that field had **already** been normalized during TTS manifest
construction. Thus the first inventory and the claimed 5,799-example ceiling
missed most numeric original references. The owner caught this error before
any experimental phase ran. The [earlier finding](/findings/2026-11-06-server-scale-preparation.md)
is deprecated for text/data conclusions, not for its hardware/checkpoint checks.

# Verified producer contract

Producer source:
`/home/yoadsnapir/tts/pocket-tts/pocket-tts/hebrew_training/build_official_manifest.py:31-58`.
`transcript` and `words` contain normalized text. Despite its name,
`alt_words.norms[*].replace` contains the text the original corpus wrote **before**
normalization. `from`..`to` are inclusive indices into `words`. Alignment preserves
word count/text, so these indices remain valid in aligned manifests.

Restore all spans right-to-left, then filter the resulting original text for
digits. This is inversion of recorded replacements, not a new normalizer or a
heuristic inverse-text-normalization model. Never filter normalized `transcript`
before restoring originals. If no replacement spans exist in a Knesset row, the
builder contract says the text was unchanged.

Older crowd manifests do not carry these alternates. Join `row.clip` to `uuid` in
`crowd_transcribe_v5/meta/*-index.json.gz` and take its original `text`.
Follow the builder's preferred-record order (retranscribed first, then shard/row)
for duplicate annotations. The cached index has 224,538 annotation rows, including
**11,023 digit-bearing original texts** (10,024 train, 999 test), before annotation
deduplication, manifest membership, alignment and duration checks. These are not
all usable audio slices; do not claim that count as imported examples.

# Repair and validation

Importer reference version 2 records `text_origin` as
`restored_pre_normalization`, `unchanged_manifest` or `crowd_source_index`.
It repairs prior version-1 corpus references in place while scanning, preserving
audio files and sample IDs, then adds requested new clips from all three sources.
Aligned start/duration are still used for cropping. No number expansion is run.

Regression tests cover digits absent from normalized text but present in the
restored original, inclusive/multiple inverse spans, and crowd source-index joins.
The UI is served on localhost:8765; credentials entered through its cURL form
persist under the ignored `data/secrets/` directory. No secret values are recorded
here. All experimental phases remain gated on explicit owner approval.

The owner's saved Explore credentials passed the authentication check. Corrected
local repair/import (4,201 new additions toward 10,000 total local clips) and the
10,000-sample Explore fetch were submitted to the live UI job manager. Initially,
a data-only coordinator waited on filesystem changes to submit `setup_scale` after
both finished. Final source counts must be
read from these jobs, not from the deprecated initial inventory. Tests at this
point: 39 passed, 11 skipped (offline, uncached whisper-tiny-dependent checks).

# Phase 1 authorization update

The owner subsequently authorized Phase 1. The data-only coordinator was stopped
without stopping its independently running UI jobs, and replaced by an authorized
Phase-1 coordinator. `phase1-20261106-180130-0a91` is a 32-sample validation run
using only already-repaired version-2 local references / original Explore text,
all from the recording-discovery side of the deterministic split, on eight GPUs.

If validation has no sample errors, both data targets are reached, and no version-1
local references remain, the coordinator refreshes cohorts and submits the full
remaining discovery run. It excludes validation IDs from that continuation and
records the two disjoint run IDs in `phase1-runs.json`. Evaluation recordings
remain held out. Phase 1.5, budget and Phases 2/2N/3 are not authorized or submitted.

Update: validation passed, local references are repaired and the local total is
10,000. The first Explore fetch failed on a long SQLite repair transaction. This
was fixed with short batched transactions; Explore is resumed and Phase 1 proceeds
on the current frozen discovery cohort without waiting for the entire download.
See [turbo validation and recovery](/findings/2026-11-06-turbo-phase1-validation.md)
for the final crowd-index membership check, run IDs and disjoint continuation.
