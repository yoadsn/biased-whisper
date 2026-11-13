---
type: Finding
title: Controller establishment - bounded sampling rejects digit-suppression-only wins
description: A tracked ten-sentence review campaign tests sparse gain/steering variants, corrects source imbalance, and identifies contaminated or ambiguous Phase 1 contrast targets without mass manual labeling.
tags: [finding, controller, phase2n, calibration, sampling, correctness]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T11:55:00Z }
sources:
  - id: protocol
    resource: /reference/controller-validation.md
    title: Review budget and fixed confirmation protocol
  - id: audit
    resource: /findings/2026-11-07-microphone-nudge-audit.md
    title: Owner microphone failure and initial metrics audit
---

# Scope and budget

The owner authorized work toward a reliable style controller, with correctness
checks in batches of 10 sentences and consultation before exceeding 100. Local
campaign: `data/controller_campaigns/turbo-style-20261107/`. Review reservations
are charged before exposing text; another candidate on the same sentence is
charged again. Current spend is **50 sentences in five packets**, comprising seven
independent discovery recordings checked under five candidates. These are
paired, exploratory calibration checks, not 35 independent recordings or a
held-out population accuracy estimate. Uncertain cases count as failures.

Calibration is 20 discovery clips, balanced local/Explore and year/non-year,
one per recording; none is in the original attribution subset. An 80-recording
confirmation pool is frozen separately. A metadata-only amendment removes four
recording overlaps with earlier held-out work (not merely identical clip IDs),
retaining the prior manifest and reason. The primary review list is fixed before
opening confirmation predictions. **No confirmation predictions have been
opened and no controller has been declared reliable.**

# Original-attribution experiments

All these runs use the same 20 discovery clips; they do not consume new held-out
correctness samples. Code and parameter signatures, exact sparse rollback and
fresh encoding remain shared with the Test tab.

- `phase3-20261107-131439-1b47`: original 64-neuron / 8-head edits, gains 3/5/7/10.
- `controller_refit-20261107-132102-c2e2`,
  `phase3-20261107-132127-16cf`: decoder-only 32/64 neurons and 2/4 heads,
  full versus decision maps, gains 5/10. Some head selections coincide across maps;
  these are not distinct discoveries.
- `controller_candidates-20261107-132616-0734`,
  `phase3-20261107-132617-1289`: head/neuron mixtures, encoder-only subsets and
  low-steering mixtures. Strong suppression also brings degeneration.
- `controller_residual_candidates-20261107-132920-29d2`,
  `phase3-20261107-132920-fdde`: all four decoder residual directions, strengths
  0.1/0.3/0.5, using the original gradient normalization/scale. No general solution.
- `controller_refit-20261107-133359-61e5`,
  `phase3-20261107-133427-3a22`: 128/256/512 global neurons, gains 1/3/5.
- `phase3-20261107-133705-6888`: 512-neuron gain 4 with both legacy controls.
  Digit share 75.7% -> 13.9%, no detected loops, drift 0.0653; controls retain
  baseline digit share. Yet the ten-sentence review still rejects it.

# Sampled correctness versus style metrics

Each reviewed candidate uses the same seven recording-level examples, totaling
ten sentences per packet. Complete clean conversions (numeric meaning, style,
non-number meaning, no uncertainty) were:

| Candidate | Clean successes / recordings |
|---|---:|
| original 64-neuron gain 10 | 1/7 |
| original 8-head gain 5 | 0/7 |
| original-source 512-neuron gain 4 | 1/7 |
| balanced-source 256-neuron gain 7 | 3/7 |
| value-filtered balanced-source 512-neuron gain 5 | 3/7 |

The judgments are conservative and include unresolved ASR/reference/identifier
ambiguities as failures, not asserted numeric mistakes. These small calibration
fractions are not claims of population accuracy or a statistically established
ranking. The common recording identities must be respected in comparisons.

The balanced candidate converts 2024 and 2021 correctly, unlike earlier edits,
but omits a whole year in one date case and changes 13 to 3 in another. Zero
digits and zero detected loops therefore remain insufficient acceptance criteria.

# Balanced discovery and target-quality problem

The old 200-point attribution pool was 194 local / 6 Explore. A bounded new run,
`phase2n-20261107-134609-f7fb`, uses 200 explicit points: 100 local / 100 Explore,
100 year / 100 other, one per recording, excluding calibration/confirmation
recordings and old attribution points. Automatic filters require accepted Phase 1
paths, rest-match >=0.95, matching baseline/reference digit groups and no detected
target loop. `phase3-20261107-134839-b6c0` evaluates its 256-neuron and 8-head
edits at 1/3/5/7/10. Neuron gains 7 and 10 eliminate digits without detected
degeneration on all 20 clips, but gain 7's sampled failures above still block
promotion to confirmation.

`number_values.py` adds a **conservative discovery filter**, not an output
rewriter or universal Hebrew inverse-text normalizer. It recognizes ordered
integer mentions, including standard Hebrew year/hundred/thousand constructions;
unsupported dates, fractions, decimals, vague quantities and ambiguous readings
return unknown. It must not be used to replace manual correctness statistics.

On the new 200 paired transcripts, it finds 146 matching integer-value sequences,
10 differing recognized sequences and 44 unparsed/ambiguous pairs. Unknown is
not evidence of an error. Even this already-filtered discovery pool thus has
recognizable value disagreements; Phase 1's `cardinal` verdict is not a
value-correct target label.

The next bounded refit takes 32 matching captures per source/year stratum:
128 total, balanced 64 local / 64 Explore and 64 year / 64 other. It reuses
existing captures, with no new gradients or manual mass labeling. Selected IDs
and policy are in `curated_captures.json`. The refit's config records the actual
subset so Phase 3 exclusion/probes cannot silently revert to the old first 200.
It still needs sampled calibration and fixed confirmation before promotion.

# Value-filtered refit result and stopping decision

`controller_refit-20261107-135554-082d` re-ranks the balanced 128-capture subset;
`phase3-20261107-135621-686d` tests 128/256/512-neuron gains 3/5/7 on the same
20 calibration clips. At 256 neurons gain 7, or 512 neurons gain 5, every output
is digit-free with no detected loops. The latter has drift 0.0791; 512 neurons
gain 7 begins looping and is rejected automatically.

The conservative rule checker finds matching integer sequences in 11 of the
12 parseable baseline/output pairs at 512 neurons gain 5; eight pairs remain
unknown. This is **not a manual accuracy estimate**, nor evidence about those
eight unknowns. The next reserved ten-sentence packet
`controller_review_batch-20261107-135915-242e` yields three clean examples,
three uncertain/malformed cases and one definite numeric failure: baseline
2005 -> `אלף וחמש` (1005). The source reference is itself 2025; no reading
supports treating the edit as a correct year conversion. The 13 case now has
a recognizable but malformed thirteen rather than the earlier outright 3.

Latest ledger summary: `controller_review_summary-20261107-140229-137c`.
All reviewed candidates remain below the clean-conversion research gate;
the 3/7 calibration interval is approximately [9.9%, 81.6%], and the paired
sample is too small/exploratory to establish a population ranking. The independent
confirmation pool is **not consumed** to rubber-stamp a candidate already
showing important failures. No reliable preset or adapter is promoted.

Stop expanding simple strength/support grids here rather than exhausting the
remaining review budget on known failure modes. The next methodological choice
is between directly optimizing a small fixed set of gains on value-consistent
pairs with content-preservation loss, versus a clearly labeled adaptive guarded
controller that may abstain/return baseline. The latter is not the original
universal fixed-weight-adapter deliverable; do not silently substitute it.
Remaining correctness budget is **50 sentences**, still in ten-sentence batches;
consult the owner before cumulative spend exceeds 100.
