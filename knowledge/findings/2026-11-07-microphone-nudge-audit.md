---
type: Finding
title: Microphone failure audit - correct Test nudges, brittle steering, and one successful higher-gain head edit
description: The owner's saved 1948 microphone comparison confirms no effect below steering collapse; Test provenance is correct, held-out digit metrics overstate quality, and a bounded follow-up head gain at strength 10 spells the year correctly.
tags: [finding, interactive, phase2n, phase3, turbo, evaluation, failure]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T09:25:04Z }
sources:
  - id: prior
    resource: /findings/2026-11-07-turbo-phase2n-heldout.md
    title: Original 50-clip held-out aggregate results
  - id: small
    resource: /findings/2026-11-06-phase2n-neuron-edits.md
    title: Small-model unit-gain strength comparison
---

# Saved owner recording and comparisons

The 5.0346875-second `microphone.wav` is preserved locally at
`data/test_audio/66edddfdb6db498e907a8eef42dc2c93/audio.wav`, with `audio.json`.
The three completed owner comparisons are
`test_transcribe-20261107-111304-3b02`,
`test_transcribe-20261107-111411-e0a9`, and
`test_transcribe-20261107-111645-699b`; each has `summary.json` and row artifacts.

Baseline is `בשנת 1948 הוקמה מדינת ישראל.`. Steering toward cardinal at
0.3 and 0.5 leaves it unchanged, while 0.7, 0.9, 1 and 2 loop and reach the
192-token cap. Neuron gains at strengths 1, 3 and 5 leave it unchanged.
This is a genuine failure of the recommended operating points, not evidence
that the owner's recording was lost or the UI result was imagined.

# Test-path audit

The saved comparisons and the live metadata endpoint both resolve
`phase3-20261107-101134-2c45` to `phase2n-20261107-093026-e404`, the local
ivrit-ai large-v3-turbo checkpoint, statistic `mean_all`, no candidate limit,
and token cap 192. These are the intended new component edits, not legacy
weight-voting candidates or whisper-small candidates.

`interactive.source_spec` follows the Phase 3 run's Phase 2 provenance;
`interactive.transcribe` checks the parameter signature and loads the same
`candidates_<edit>.npz` arrays/statistic/sorting as `phase3_device.prepare`.
Both paths add `strength * delta`, freshly encode the audio, use
`phase3_inference.greedy_tokens`, and restore original weights in `finally`.
Test uses device-native sparse application; Phase 3's host-input sparse
application performs the same indexed addition. No sign reversal, dropped
strength, candidate substitution or cache-reuse bug was found in this audit.
This is a source/provenance audit, not a new full-model runtime parity test.
`uv run pytest tests/test_interactive.py -q`: 24 passed, one upstream warning.

# Bounded new comparison on the same saved microphone audio

Submitted through the running server's existing GPU job queue, not a CLI
restart: `test_transcribe-20261107-112423-214a`, completed successfully.
It compares the untouched baseline and three independent edits:

| Edit | Strength | Transcript outcome |
|---|---:|---|
| steering toward cardinal | 0.6 | unchanged digits |
| 64-neuron gain toward cardinal | 10 | `בשנת אלף תשע מאות תשע מאות שמונה הוקמה מדינת ישראל.` (wrong number) |
| 8-head gain toward cardinal | 10 | `בשנת אלף תשע מאות ארבעים ושמונה הוקמה מדינת ישראל.` (correct full conversion) |

None of these new rows hit the token cap. The head edit demonstrates that
the Test application path can produce the intended conversion on this audio.
It is one owner-example success after inspecting failures, not an unbiased
estimate of generalization or a validated default strength.

# Re-audit of existing held-out transcripts (no new evaluation audio)

Read all baseline/cardinal condition outputs from the existing 50-clip
Phase 3 artifact. Of 50 baseline clips, 38 contain digits. At steering strength
0.3, 25 of those 38 lose all digits; 12 retain identical digit groups and one
changes digit groups without removing all digits. At strength 1, 37 lose all
digits, but 11 of the 50 outputs are detected as degenerate.

There are genuine conversions at 0.3, e.g. 24 -> `עשרים וארבע`,
60 -> `שישים`, and 800,000 -> `שמונה מאות אלף`. However, spot checks also show:

- sample 5911: baseline 14 becomes `עשר` (10), not a value-preserving conversion;
  the corpus reference itself says 13, so baseline accuracy is also not assured;
- sample 3470: 80 and 14 convert, but 16 becomes `שפע`, not a number;
- sample 18858: 1865 becomes unfinished `אלף שמונה מאות שישים וחמ`;
- baseline years 1998, 1988, 1986 and 2009 remain digits at strength 0.3.

These are spot checks, not a complete manually labeled value-accuracy score.
The digit-share metric counts existing number words, ordinals and new number
words without verifying their values; `_strip_numbers` removes recognized
number words before measuring text drift. Consequently wrong-number outputs
can improve digit share and evade much of the drift penalty. A zero
`is_degenerate` rate means no detected repetition, not correct or coherent
numbers. The previously reported 21.7% digit share is reproducible but is
**not a 78.3% correct-conversion rate**.

At strength 1, neuron edits remove all digits in only 5/38 baseline-digit
clips, and heads in 3/38. At 0.3 both leave all 50 transcripts exactly
unchanged. The original held-out grid therefore did not characterize the
larger unit-gain operating range that worked on small.

# Interpretation and next experiment

The small-model result is more promising for unit gains, but its strongest
cardinal neuron condition was strength 10 (100% gain adjustment), not strength
1 (10%). Turbo's held-out unit-gain grid tested only 0.3 and 1. Steering and
unit gains must not share one universal safe/default strength: high steering
is clearly unsafe, while larger head gains warrant separate evaluation.
Small's own value-correctness and transcription-quality gaps still apply.

The saved turbo attribution keys also reveal a discovery skew: 124 committee,
70 plenum and only 6 Explore samples among the first 200 accepted points.
This is mixed-source but not balanced; any link to the microphone failure
remains a hypothesis. Turbo has four decoder layers versus small's twelve,
so layer localization and gains cannot be assumed to transfer directly.

Current conclusion: a real but brittle intervention signal, **not a reliable
general style controller**. The next scientific evaluation should separately
calibrate neuron/head gains (including the small-comparable range), use a
discovery-only calibration set, then report value-preserving span conversions,
year/date subgroups, failures and normalized WER against matched controls.
Do not promote the one successful head-gain example to a default or keep
tuning on the same held-out cohort. No new held-out inference was run here;
only saved outputs were audited, plus the one new microphone comparison.
