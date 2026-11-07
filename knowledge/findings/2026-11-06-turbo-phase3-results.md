---
type: Finding
title: Turbo Phase 3 completes with modest held-out steering at strength 20
description: Owner-authorized 25-condition batched evaluation on 50 recording-held-out clips found small directional shifts only at the largest tested strength, flat generated-output controls and no detected degeneration.
tags: [finding, phase3, turbo, held-out, steering, controls]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-06T20:14:25Z }
---

# Authorized execution

Owner explicitly requested Phase 3 after the CUDA optimization/validation.
`phase3-20261106-220319-1429` completed successfully in **441.07 seconds (7.35 min)**:
25 conditions, 50 held-out clips, 30 in-sample probe points per condition,
1,250 transcription records and 2,000/2,000 progress units. Source candidates:
`phase2-20261106-203005-7d37`, 50,000 weights per direction, `mean_all` statistic.

The unchanged prepared `setup_scale-20261106-192355-9dca/phase3.template.json` supplied
the explicit 3,888-ID recording-held-out pool, conservative eval cap 50, 192-token
cap, both directions, strengths 2/5/10/20, both shuffled-value/random-index controls,
seed 0, and the pinned local ivrit-ai large-v3-turbo checkpoint. Execution overrides
were batched backend, batch size 8 and post-run cleanup. All 50 transcribed IDs
were verified members of the evaluation split and absent from discovery.

# Generated-output results

| Condition | Digit mentions | Cardinal-word mentions | Digit mention share | Outputs containing digits | Degenerate rate | Non-number text drift |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 47 | 28 | 62.67% | 76% | 0% | 0 |
| to_digits, strength 20 | 48 | 27 | 64.00% | 78% | 0% | 0.00487 |
| to_cardinal, strength 20 | 44 | 31 | 58.67% | 72% | 0% | 0 |
| All generated-output controls | 47 | 28 | 62.67% | 76% | 0% | 0 |

Strengths 2/5/10 had unchanged generated-output metrics for both directions.
The in-sample mean sequence-contrast log-odds moved monotonically with real edit
strength: baseline +5.073, to_digits at 20 +5.338, to_cardinal at 20 +4.482;
control probe changes remained much smaller.

This is a **modest directional effect**, not strong global control. No numeric
value-correctness review, normalized WER, recording-cluster confidence intervals
or additional held-out evaluation was performed. The number-word mention detector
and non-number drift metric have their existing limitations. The 50-clip result
does not justify silently increasing strengths, switching statistics or consuming
the remaining held-out pool. The budget sweep's per-sample/consensus objectives
are not equivalent to these capped, averaged Phase 2 candidate edits.

# Efficiency and cleanup

Peak allocated CUDA memory was **4,081,495,040 bytes**; encoder/KV caches did not
cross conditions. Post-run cleanup released the model, leaving 17,616,896 bytes
allocated / 37,748,736 reserved for runtime/library workspaces. Original artifacts
remain intact. The new [Test tab](/reference/interactive-test.md) allows local
audio exploration without rerunning the scientific evaluation.
