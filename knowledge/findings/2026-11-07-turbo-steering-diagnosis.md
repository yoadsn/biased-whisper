---
type: Finding
title: Turbo steering weakness is confounded by edit method and missing post-selection calibration
description: Read-only comparison of saved turbo artifacts with small-model findings identifies different discovery methods, vote-versus-effect selection, weak directional consistency, and uncalibrated truncated edits; no new model experiment was run.
tags: [finding, turbo, phase2, phase2n, phase3, calibration, diagnosis]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T06:58:01Z }
---

# Scope

Owner asked why small steered strongly while large-v3-turbo did not, whether
data/model size/defaults explain it, and whether layer localization is suspicious.
This investigation read code and existing local artifacts and computed NumPy
statistics over saved captures. It did not load a model, alter scientific code,
submit jobs, or consume additional held-out audio. Graft executable/index were
absent in this checkout, so code was inspected directly.

# The experiments are not matched

The strongest [small results](/findings/2026-11-06-phase2n-neuron-edits.md) used
`phase2n-20261106-125436-0ef2`: gradient-times-activation component discovery,
significance gating followed by effect-size ranking, and coherent neuron output
gain edits (whole `fc2.weight` columns). Sixty-four neurons reached 0% digit share
at strength 10 on 40 utterances, with non-number drift 0.30. Independent
[cross-domain tests](/findings/2026-11-06-cross-domain-transfer.md) reached 7%.

Turbo `phase3-20261106-220319-1429` instead used ordinary weight-space
`phase2-20261106-203005-7d37`, `all_no_embed`, 100 automatic-cardinal points,
1M captured entries per point, 50k final candidates, `mean_all` values.
No turbo Phase 2N run exists in the inspected local run directory.
Small's earlier ordinary Phase 2 used **bitfit**, not this full weight space,
and the single-decision mass objective rather than span sequence contrast.
Thus even that comparison changes parameter filter and objective.

Strength numbers are not portable: Phase 2N neuron strength 1 changes selected
unit gains by 10%, whereas ordinary Phase 2 multiplies an averaged minimum-norm
weight delta. The numerical label does not specify a common causal dose.

# Direct evidence of an underpowered final adapter

Saved Phase 3 probe mean sequence contrasts:

| Cardinal strength | Mean contrast | Change from baseline |
|---:|---:|---:|
| 0 | +5.07336 | 0 |
| 2 | +5.01322 | -0.06014 |
| 5 | +4.92373 | -0.14963 |
| 10 | +4.77551 | -0.29785 |
| 20 | +4.48209 | -0.59127 |

Even the strongest tested condition remains digit-favoring on this in-sample
probe average. Generated digit share only moves 62.67% -> 58.67%. These are weak
steering results, not reliable cardinal forcing.
The owner's saved `test_transcribe-20261107-085350-3221` and
`test_transcribe-20261107-085430-d3ab` use the correct turbo provenance and
same `mean_all` adapter: strengths 2 and 4 leave the 1948 transcript unchanged.
This is consistent with Phase 3, not evidence by itself of a separate UI fault.

Code computes dense per-point minimum-norm nudges using the full gradient norm
(`phase2.py:95-112`), truncates them, votes, caps candidates by votes
(`phase2.py:351-360`), and averages raw sparse values with absent entries zero
(`aggregate.py:118-140`). Phase 3 applies these values times strength, without
post-selection objective calibration (`phase3_device.py:51-59,89-101`).
Truncating and averaging a dense target-reaching first-order delta does not
preserve its target-reaching property. This is a methodological weakness,
not a demonstrated CUDA or weight-addressing bug.

# Voting stability is not directional consistency or effect size

Recomputed from saved candidate/vote NPZs:

| Statistic | To digits | To cardinal |
|---|---:|---:|
| FDR-passing coordinates before cap | 854,696 | 1,074,136 |
| Saved candidates | 50,000 | 50,000 |
| Median count among available captures | 54 / 76 | 70 / 100 |
| Median sign consistency when present | 56.36% | 55.56% |
| Candidates with sign consistency >= 80% | 5,876 | 4,273 |
| Strength-1 mean-all edit L2 | 0.000392425 | 0.000820166 |

Voting sums signed normalized vectors in groups of three but ranks the absolute
group sum (`aggregate.py:95-111`). Repeatedly large, poorly sign-aligned entries
can still receive votes. There is no directional-consistency gate; capping
primarily by votes does not optimize objective movement. High split-half vote
overlap is therefore not sufficient evidence of a reusable causal controller.
`mean_present` cardinal edit L2 is only 0.00109527 (about 1.34x `mean_all`), so
zero-filled averaging alone is not an adequate explanation.

# Localization: distinguish counts, capacity, gradients and causality

Turbo has 32 encoder and only 4 decoder blocks, versus small's 12/12;
the turbo architecture is not simply a deeper version of small's decoder.
Encoder parameters account for **85.82%** of the inspected non-embedding space,
while **82.56%** of cardinal candidates are encoder entries. Raw tower counts
do not establish encoder enrichment.

Actual cardinal candidate counts peak at encoder L03 (6,050), decoder L00
(5,236), encoder L02 (5,140), and encoder L04 (3,874). Here low encoder indices
are input-side, not the final encoder blocks. In contrast, the largest raw
gradient-energy group is decoder L03 `fc2.weight` (10.95%), followed by that
layer's cross-attention output/value matrices; late encoder `fc2.weight`
groups L28-L31 are also prominent. Only 196 final candidates are decoder L03.

Per-tensor RMS standardization divides by the nudge/gradient tensor's RMS,
not the pretrained weight RMS. It removes between-tensor magnitude differences
for selection. Candidate frequency under that metric is not a causal map.
An encoder-output/decoder-cross-attention interface is a plausible intervention
site, but necessity or unique localization requires direct interventions.

# Data and model-size hypotheses

Phase 2 did not train on 16,112 examples: it took the first 100 accepted rows
from one Phase 1 run (`phase1.decision_points`, `phase1.py:332-347`), with no
shuffle/stratification there. These points were Explore-only and had zero human
verdict overrides. The expanded corpus therefore does not establish that the
edit was diluted by thousands of dirty examples. Automatic cardinal labels
still do not verify numeric value or preservation of surrounding text;
cohort ordering, recording diversity and pair quality remain risks.

The existing calibrated turbo budget sweep already moved held-out decision
contrast by about -2.009 at 740 weights and -2.003 at 74k weights against near-zero
shuffled controls. It calibrates the final restricted consensus direction to
predicted train change -3 (`budget_device.py:319-335,357-360`). This is evidence
of transferable local signal, not proof of generated-output control; it uses
a different consensus estimator and discovery-derived test points.

# Recommended discriminating work, not yet executed

1. Review/stratify a small discovery-only pair set, including values, prefixes,
   non-number changes and recording diversity. No need to enlarge Phase 1 first.
2. Compare current adapter, calibrated consensus and Phase 2N neuron/head edits
   on matched discovery pairs. Measure predicted and actual objective movement
   after the final support is selected; use effect size and directional coherence.
3. Calibrate strengths per model/edit method rather than reusing small's labels.
   Check generated output, number correctness, text drift and degeneration there.
4. Only after choosing settings, perform a separately approved recording-held-out
   evaluation without tuning on that set. Raising strength blindly is not a fix.

Current diagnosis: selection/averaging/calibration and changed edit method are
better-supported explanations than "turbo cannot represent cardinals" or
"more data is dirty." A matched causal experiment is needed to apportion them.
