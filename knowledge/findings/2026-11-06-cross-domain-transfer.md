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

# Reverse direction and agreement of independent fits

Phase 1 v2 on all 1,300 explore clips: 895 contained digits, 803 (90%) gave a clean word path, 13 degenerate. Phase 2N was then fit on
**400 explore points only** (run `phase2n-20261106-145905-dd42`) and compared with the Knesset-only fit (91 points):

| agreement of two independent fits (74,160 units) | value |
|---|---:|
| Pearson correlation of mean per-unit attribution | 0.87 (Spearman 0.59) |
| top-100 units by effect: overlap / chance / sign agreement | 71 / 0.1 / 100% |
| top-1000 units: overlap / chance / sign agreement | 663 / 13.5 / 95% |
| best steering layer | 10 and 10 |
| cosine between steering directions per decoder layer | 0.72 to 0.86 (peak 0.86 at layers 7-9) |

The same units top both lists (decoder layer 10 cross-attention heads 3 and 9, layer 10 self-attention head 1, layer 11 neuron 1746,
layer 9 neuron 2193). The decision is localised, not an artifact of one corpus.

**Explore-fit edits evaluated on 60 held-out Knesset clips** (baseline digit share 53%), strength / digit share / drift; controls are
shuffled values and same values on random parameters:

| edit | 0.3 | 1 | 3 |
|---|---|---|---|
| 64 neurons -> words | 48% / 0.02 | **39%** / 0.03 | **17%** / 0.12 |
| 64 neurons -> digits | 56% / 0.03 | 59% / 0.07 | 63% / 0.13 |
| steering -> words | **33%** / 0.08 | **10%** / 0.17 | 0%, collapses (95% degenerate) |
| steering -> digits | 57% / 0.07 | 75% / 0.22 | 98%, collapses (98% degenerate) |
| shuffled controls (range) | 51-55% / 0.01-0.04 | 52-58% / 0.02-0.11 | 52-64% / 0.05-0.32 |
| random-parameter controls (range) | 53-55% / 0.00-0.01 | 52-55% / 0.00-0.05 | 38-51% / 0.00-0.70* |

*the only non-flat random control is the strength-3 steering vector placed on random parameters (a 6-norm bias on arbitrary
`fc2.bias`/other entries damages the model); it reflects the size of the edit, not the direction.

* The neuron edit is the cleanest tool: at strength 1 it moves the share by 14 points with drift 0.03, at 3 by 36 points with drift 0.12,
  and no collapse. It moves toward words much more easily than toward digits (the model already prefers digits on most numbers).
* **Steering must stay <= 1** (it adds the same vector at every position); at 3 the decoder collapses.
* Both fits transfer in both directions across domains, so a single edit can be shipped for general Hebrew.
