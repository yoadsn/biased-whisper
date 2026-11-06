---
type: Finding
title: Large-v3-turbo eight-GPU Phase 1 validation
description: Completed large-v3-turbo Phase 1 over all 16,112 discovery samples, with 10,940 automatic cardinal verdicts, no runtime errors, and 3,888 recording-disjoint evaluation samples untouched.
tags: [finding, phase1, whisper-large-v3-turbo, a40, corpus]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T16:11:06Z }
---

# Runtime validation, not a population estimate

Run `phase1-20261106-180130-0a91` completed on eight independent A40 replicas of
the local 808,878,080-parameter large-v3-turbo checkpoint. It used 32 already
corrected, recording-discovery samples while imports continued; max decode cap
192, span mode enabled, built-in satellites. There were **zero sample/runtime errors**.

| Automatic verdict | Count |
|---|---:|
| cardinal | 18 |
| no_number_words | 8 |
| no_digits | 6 |

26 free decodes contained digits; mean digit mass at the decision was
0.7505905915. These small, deliberately source-balanced validation counts only
prove the real multi-GPU path runs; they are not an unbiased model success-rate
estimate. Automatic cardinal verdicts do not establish numerical value correctness.

# Corrected local corpus

`corpus_import_pockettts-20261106-175753-63ac` completed with 4,201 new clips and
5,799 repaired references; all 10,000 local samples now carry reference version 2.
Totals: plenum 2,838, committee 7,162 (about 29.63 hours together), no crop failures.
The scan found 156,183 new eligible plenum references and 242,717 committee
references beyond existing samples; the old normalized-transcript ceiling was wrong.

Crowd is more subtle than merely reading its original index: the preferred source
index has 10,071 digit-bearing UUIDs, but **none** of the 112,620 aligned manifest
rows matches a digit-bearing original annotation (including non-preferred duplicate
annotations). All 112,620 UUIDs were matched to the source index. Thus this prepared
aligned subset provides no original-digit crowd cohort; the source as a whole does.
Do not fabricate crowd references or use normalized-word forms as original digits.

The initial 5,799 clips are retained with repaired text, not randomly resampled.
Their initial selection favored digit strings the TTS normalizer left unchanged,
so the final local corpus is not a uniform draw from all recovered digit examples.
Use source / `text_origin` / import-batch strata when interpreting results, and do
not claim its number-type distribution represents the entire Knesset corpus.

# SQLite interruption and fix

The first Explore fetch (`corpus_fetch_explore-20261106-175753-7ee5`) saved 20 clips
then failed with `OperationalError: database is locked`. The reference-repair
scan left a partial write transaction open while reading the rest of a large
manifest. The full-cohort coordinator correctly stopped instead of launching
against incomplete inputs.

Repairs are now collected in memory and committed in short batches of at most
128, with no write transaction around scanning. `add_sample` now uses a connection
transaction context, ensuring duplicate/failed inserts roll back and release locks.
Regression tests check duplicate rollback and independent-writer access after a
repair batch. Suite at this point: 42 passed, 10 skipped.

# Running continuation (Phase 1 only)

- Explore resume: `corpus_fetch_explore-20261106-181103-951d`, target 9,980 new
  additions to reach 10,000 total, preserving the 20 already stored.
- Refreshed cohort: `setup_scale-20261106-181103-3c2d`, 8,104 discovery / 1,916
  evaluation samples at its snapshot (10,000 local + 20 Explore). Model reload was
  skipped because the real checkpoint and all eight devices had already validated.
- Phase 1 continuation: `phase1-20261106-181106-3926`, **8,072 samples on eight GPUs**,
  excluding the 32 validation IDs, all recording-held-out evaluation samples excluded.

A replacement event-driven coordinator lets the local run proceed while Explore
downloads. After both succeed it freezes the expanded corpus, runs Phase 1 only
on discovery IDs not processed in either preceding run, and writes all disjoint
run IDs in `phase1-runs.json` under the new setup run. It stops if a dependency
fails or a Phase 1 run reports sample errors. No Phase 1.5, budget or Phase 2/2N/3
job is authorized or submitted.

# Completed expanded Phase 1

All previously scheduled increments subsequently completed; no sample-budget
reduction was approved/applied before their completion. Runs are disjoint:

| Run | Samples | Cardinal | No digits | Other verdicts | Runtime errors |
|---|---:|---:|---:|---:|---:|
| `phase1-20261106-180130-0a91` | 32 | 18 | 6 | 8 | 0 |
| `phase1-20261106-181106-3926` | 8,072 | 5,651 | 1,587 | 834 | 0 |
| `phase1-20261106-192356-6310` | 8,008 | 5,271 | 2,338 | 399 | 0 |
| Total | 16,112 | 10,940 | 3,931 | 1,241 | 0 |

Combined remaining verdicts: 836 no_number_words, 198 degenerate, 171 partial,
36 no_divergence. These are automatic surface-form verdicts, not human-reviewed
value correctness or controlled steering results. Do not automatically fit on
all 10,940 cardinal pairs; a smaller, source/number-type-stratified reviewed pool
is a more practical next starting point.

Per-source discovery totals / automatic cardinal counts:

- plenum: 2,281 / 1,526.
- committee: 5,804 / 4,134.
- Explore: 8,027 / 5,280.

Final artifacts:
`data/runs/setup_scale-20261106-192355-9dca/{cohorts.json,phase1-runs.json}`.
Verified from saved results: exactly 16,112 unique IDs, equal to the frozen
discovery set; no duplicate IDs across runs; no ID or recording overlap with the
**3,888 untouched evaluation samples** (557 plenum, 1,358 committee, 1,973 Explore).
Corpus totals are 10,000 local + 10,000 Explore clips.

The event-driven coordinator has exited successfully and will submit no more
jobs. Phase 1.5, budget and Phase 2/2N/3 remain unsubmitted and unauthorized.
The [throughput review](/findings/2026-11-06-phase1-throughput.md) still applies to
future runs; no batching/worker optimization was implemented during this run.
