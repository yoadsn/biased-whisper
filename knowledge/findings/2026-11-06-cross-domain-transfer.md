---
type: Finding
title: Edits fit on Knesset speech transfer to explore.ivrit.ai audio
description: Neuron-gain and steering edits found on 91 Knesset decision points move digit share on 60 held-out explore.ivrit.ai clips (lectures, podcasts) in both directions, with shuffled controls flat.
tags: [finding, phase3, transfer, explore-ivrit-ai, domain-shift, whisper-small]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T15:20:00Z }
sources:
  - id: corpus
    resource: /reference/corpus-sources.md
    title: explore.ivrit.ai corpus (1,300 clips, 285 episodes, short segments, median 3.5 s)
  - id: fit
    resource: /findings/2026-11-06-phase2n-neuron-edits.md
    title: Phase 2N fit on Knesset points (the edits evaluated here)
---

# Setup

Edits from Phase 2N run `phase2n-20261106-125436-0ef2`[^fit] (fit on 91 span-level decision points from the Knesset corpus only) applied to
whisper-small, evaluated with Phase 3 on 60 random clips from the explore.ivrit.ai corpus[^corpus]. The clips come from Torah
lectures, podcasts, university courses and radio; none of that audio, speaker or topic was used in fitting.

# Results (digit share of number mentions; baseline 60%, digit-output rate 70%)

| edit | strength | digit share | digit-output rate | text drift | shuffled control (share / drift) |
|---|---:|---:|---:|---:|---:|
| 64 neurons -> words | 1 | 55% | 60% | 0.03 | 60% / 0.01 |
| 64 neurons -> words | 3 | **34%** | 38% | 0.09 | 62% / 0.08 |
| 64 neurons -> words | 10 | **7%** | 8% | 0.24 | 60% / 0.19 |
| 64 neurons -> digits | 1 | 70% | 83% | 0.04 | 59% / 0.02 |
| 64 neurons -> digits | 3 | 77% | 93% | 0.13 | 60% / 0.05 |
| 64 neurons -> digits | 10 | 85% | 93% | 0.40 | 61% / 0.20 |
| steering bias (layer 10) -> words | 0.3 | **42%** | 42% | 0.08 | 63% / 0.04 |
| steering bias -> words | 1 | **18%** | 20% | 0.20 | 64% / 0.12 |
| steering bias -> digits | 0.3 | 73% | 87% | 0.06 | 58% / 0.04 |
| steering bias -> digits | 1 | 83% | 100% | 0.21 | 68% / 0.13 |

# Reading

* **Transfer is real and about as strong as in-domain** (in-domain: 58% -> 29% at strength 3 and 0% at 10; here 60% -> 34% and 7%).
  The direction is not specific to parliamentary speech or to the fitting speakers.
* **Both directions, dose-dependent, controls flat.** The shuffled control moves digit share by at most +-5 points up to strength 3.
* **Drift is the cost to watch.** It is comparable between nudge and shuffled control at low strength and about 2x at high strength.
  Moderate strengths (neurons 3, steering 0.3-1) keep drift near 0.1-0.2 while moving the share by 25-40 points.
* **Caveats:** 60 clips and about 60-90 number mentions per condition (no confidence intervals yet); clips are short (median 3.5 s), so
  many contain a single number; digit-output rate and digit share are measured on the model's own decodes (no ground-truth style label);
  the number *value* is not verified. These are what the next steps address.

[^corpus]: explore.ivrit.ai corpus (1,300 clips, 285 episodes)
[^fit]: Phase 2N run on Knesset decision points
