---
type: Reference
title: Bounded style-controller validation and review budget
description: Owner-authorized candidate calibration, frozen confirmation cohorts, ten-sentence review packets, hard 100-sentence ceiling, and explicit correctness/uncertainty labels.
tags: [reference, controller, validation, review-budget, statistics]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T11:20:00Z }
---

# Owner constraints

On 2026-11-07 the owner requested a reliable weight-based style controller.
Do not require mass manual number-value labeling. Use sample checks in batches
of **10 sentences**, and **consult the owner before reviewing more than 100
sentences**. This limit applies across this controller-establishment effort,
not anew for each candidate or session. Do not reset a campaign to evade it.

The implementation conservatively charges repeated examples under different
candidates again, and charges the maximum number of sentences on either side
of a comparison (reference, baseline, candidate). Reference/baseline text is
shared context, not an additional independent correctness judgment. Automatic
generation/metrics are not manual correctness checks. Keep inference bounded
as well; a review budget is not permission for unlimited automated experiments.

# Current campaign and separation

Local ignored artifacts: `data/controller_campaigns/turbo-style-20261107/`.
`manifest.json` and `campaign.json` freeze the initial calibration pool of 20
discovery clips (balanced local/Explore and year/non-year strata), and an
80-clip previously unevaluated confirmation pool, excluding the old Phase 3
50 clips. Recordings/episodes are unique within these pools. They come from
the original recording-disjoint discovery/evaluation assignment.

Calibration, including any adaptive candidate changes, is exploratory.
Freeze the selected candidate, strength, statistic and file hash before
opening confirmation predictions. Fix the confirmation review subset using
metadata only, not edited-output quality. Confirmation failures are not an
invitation to tune and retest on the same cohort. If a sample is unclear,
count it as an uncertain failure, not a silent exclusion or replacement.

# Jobs

- `controller_candidates`: compose/filter saved sparse candidates, with source
  model/filter/objective provenance; no new gradient captures.
- `controller_refit`: re-rank the existing Phase 2N full/decision attribution
  maps into decoder-only 32/64-neuron and 2/4-head gain edits; explicit bounded
  counts/towers/maps also allow larger 128/256/512-neuron supports at lower gains. This is a
  derivative of saved discovery, not a new independent attribution experiment.
  `capture_sample_ids` can restrict saved captures before statistics; it updates
  the recorded `sample_ids`/`max_points` so probes and exclusions use that actual
  subset. Missing or duplicate requested captures fail explicitly.
- `controller_residual_candidates`: expose the original normalized residual
  gradient direction at each decoder layer using the original Phase 2N scale;
  this does not calculate new gradients or load a model.
- `phase3`: actual batched candidate inference and matched controls.
- `controller_review_batch`: reserves a packet and charges its sentence cost
  **before exposing its text**; rejects packets over 10 sentences or a campaign
  spend over 100. Packet rows live under the job run's `packet.json`.
- `controller_review_submit`: immutable judgments with `value_ok`, `style_ok`,
  `text_ok`, `uncertain` and a required explanatory `note`. Success requires all
  three correctness flags, no uncertainty and no detected repetition. Explicit
  all-cardinal judgments cannot retain digits.
- `controller_freeze`: saves a reviewed calibration candidate and its hash;
  confirmation packets must use that candidate without changes.
- `controller_review_summary`: reports counts and exact binomial intervals;
  naturally word-style baselines are not successful *digit conversions*.

Jobs use the normal registry/UI/CLI artifacts. Submit GPU work through the
running server queue; do not start another Manager while an API job is active,
because the existing Manager startup logic can mark another process's job
interrupted. Restart the UI to register new job code only when no jobs are active.

# What correctness means here

Judge complete number conversion, preservation of numeric meaning, and
non-number sentence meaning separately. Unchanged digits, missing numbers,
incorrect years, malformed/incomplete number phrases, loops and material
content changes are failures, not favorable digit suppression. Compare both
reference and baseline: their disagreement can itself make a case uncertain.
Transcript comparison alone is not an audio-grounded accuracy assessment.

Report the numerator and denominator, confidence interval, review spend and
remaining failure types. An exact binomial interval assumes independent
recording-level examples, not multiple numbers from one recording. Statistical
significance of a style shift is not evidence of sufficiently high reliability.
Do not repeatedly inspect ordinary p-values/intervals and stop at significance;
use a fixed final confirmation size, with early rejection allowed for futility.

The campaign's working acceptance gate is a one-sided 95% lower bound above
90% for complete, value-preserving, meaning-preserving conversions, alongside
no detected looping/truncation. This is an explicit research gate, not a
guarantee for arbitrary audio or a reason to weaken labels to pass. If the
budget or results cannot support it, report that rather than promoting a preset.

# Explicit balanced rediscovery

`phase2n.sample_ids` optionally restricts accepted Phase 1 points **before**
`max_points` is applied. Invalid, duplicate or unaccepted requested IDs fail
explicitly. Phase 3 reference/batched paths use the same recorded subset to
reconstruct probes and exclude discovery samples; they must not revert to the
old first-N ordering. The UI hides this advanced field to avoid parsing null as
a text input; explicit cohorts are supplied in recorded job configs.

The bounded balanced rerun selects 200 new discovery points: 100 local / 100
Explore and 100 year / 100 other, one per recording, excluding calibration and
confirmation recordings and the original 200 attribution points. It also
requires Phase 1 rest-match >=0.95, matching reference/baseline digit groups
after punctuation removal and no detected target loop. These are automatic
quality filters, **not verified cardinal-value labels**. The frozen selection
and filters are in the campaign's `balanced_discovery.json`. Any resulting
candidate still needs the same calibration and held-out confirmation gates.

`number_values.py` offers a cheap, conservative ordered-integer gate for
discovery pairs. It does not rewrite outputs, guess unknown dates/decimals/
fractions or replace sampled validation. Unknown parses are excluded from
discovery refits, not labeled as errors. Matching parsed values do not prove
audio-grounded correctness, grammaticality or non-number preservation.
