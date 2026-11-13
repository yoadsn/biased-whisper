---
type: Finding
title: Turbo Phase 2N held-out control - residual steering is strong at low strength; unit gains are gentler
description: Calibrated 50-sample recording-held-out Phase 3 on large-v3-turbo found bidirectional, dose-dependent Phase 2N control with flat controls; decoder layer-2 steering reaches 21.7% digit share at strength 0.3 with drift 0.043 and no degeneration, while strength 1 collapses toward cardinals.
tags: [finding, phase2n, phase3, turbo, held-out, steering, neurons, heads]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T08:22:31Z }
sources:
  - id: p2n
    resource: /findings/2026-11-07-turbo-phase2n-parity.md
    title: Turbo Phase 2N attribution and discovery-only parity
  - id: run
    resource: data/runs/phase3-20261107-101134-2c45
    title: Calibrated Phase 2N Phase 3 on 50 recording-held-out clips
---

# Run

Owner-approved continuation after low-strength discovery calibration.
`phase3-20261107-101134-2c45` evaluated the Phase 2N edits on the same frozen
50-clip recording-held-out set used by the earlier weak weight-voting run. Six
edit types, strengths **0.3 and 1**, both shuffled-value and random-index
controls, 37 conditions, 2,960 progress units. The actual CLI job finished
successfully; the already-running UI server had briefly marked its job record
interrupted at startup. Post-run cleanup released the model (8.52 MB allocated /
14.68 MB reserved).

Baseline held-out metrics were unchanged: 47 digit mentions, 28 word mentions,
62.67% digit share, 76% outputs containing digits, 0% detected degeneration.

# Main effect table

| Edit | Strength | Digit share | Digit-output rate | Text drift | Degenerate | Probe log-odds |
|---|---:|---:|---:|---:|---:|---:|
| baseline | - | 62.67% | 76% | 0 | 0% | +4.089 |
| neurons -> digits | 0.3 | 62.67% | 76% | 0 | 0% | +4.346 |
| neurons -> digits | 1 | 65.33% | 80% | 0.009 | 0% | +4.941 |
| neurons -> cardinal | 0.3 | 62.67% | 76% | 0 | 0% | +3.833 |
| neurons -> cardinal | 1 | **54.67%** | 66% | 0.003 | 0% | +3.239 |
| heads -> digits | 0.3 | 63.16% | 78% | 0.003 | 0% | +4.316 |
| heads -> digits | 1 | 64.00% | 78% | 0.017 | 0% | +4.835 |
| heads -> cardinal | 0.3 | 62.67% | 76% | 0 | 0% | +3.861 |
| heads -> cardinal | 1 | 57.33% | 70% | 0.010 | 0% | +3.319 |
| steering -> digits | 0.3 | 72.37% | 90% | 0.023 | 0% | +8.990 |
| steering -> digits | 1 | 93.02% | 98% | 0.323 | 4% | +44.481 |
| steering -> cardinal | 0.3 | **21.74%** | 26% | 0.043 | 0% | +0.339 |
| steering -> cardinal | 1 | **1.61%** | 2% | 0.262 | **22%** | -6.138 |

Both controls stay essentially flat around baseline; the largest control
variation is steering shuffled values at strength 1 (56-62% digit share, drift
0.046-0.055), far weaker than the real steering direction.

# Interpretation

**Subsequent owner-example audit:** the aggregates below do not establish a
reliable style controller. The saved microphone year case is unchanged at
steering 0.3/0.5/0.6 and loops from 0.7; held-out spot checks contain wrong and
unfinished numbers even at 0.3. See the [microphone audit](/findings/2026-11-07-microphone-nudge-audit.md).
Treat "useful operating point" below as a cohort-level digit-suppression result,
not value-correct conversion accuracy or a generally safe recommendation.

- **The corrected turbo pipeline now steers.** This is not the old weak Phase 2
  result: decoder residual steering changes held-out digit share by about
  40-45 points already at strength 0.3, with no detected degeneration.
- **A useful cardinal operating point is `steer_to_cardinal ×0.3`:** digit share
  62.67% -> 21.74%, digit-output rate 76% -> 26%, text drift 0.043.
- **Strength 1 is not a safe default for full steering:** cardinal reaches 1.61%
  digit share but with 22% detected degeneration and 0.262 non-number drift;
  digits reach 93% with 4% degeneration and 0.323 drift.
- **Unit-gain edits are more conservative.** Neurons at strength 1 move cardinal
  digit share by 8.0 points and digits by 2.7 points with drift <= 0.017 and no
  degeneration. Heads are similar but slightly more drift-prone. On turbo, unlike
  small's strongest early experiment, 64 units alone do not force the style.
- Directional probe means and generated-output share agree qualitatively across
  the six edits. The probe can saturate or overstate the final behavioral change,
  especially for every-position steering.

# Caveats and next steps

This remains 50 clips, without value-correct cardinal verification, confidence
intervals, WER after number normalization, style-coherence scoring or an
episode-disjoint Explore split. Before treating the decoder layer-2 vector as a
portable adapter, test neighboring layers, a random-neuron control matched to
neuron edits, fewer units, and cross-domain recordings. Steering should be
position-limited or otherwise constrained before any use where preserving all
non-number text matters.
