---
type: Hypothesis
title: Digit vs cardinal output is steerable by a small set of weights
description: The owner's theory that Hebrew Whisper's choice between digit and spelled-out number output is localised enough to bias with small weight adapters.
tags: [whisper, hebrew, ivrit-ai, interpretability, numbers, itn]
status: draft
generated: { by: human:yoad, at: 2026-11-06T08:00:00Z }
---

# Statement

Hebrew-trained Whisper models (ivrit.ai) can output a spoken number either as **digits** (`ב-2024`, `17 דקות`)
or as **cardinal text** (`אלפיים עשרים וארבע`, `שבע עשרה דקות`). The choice is not spread uniformly across the
network: a small, identifiable set of parameters ("knobs", most likely in the text decoder) controls it.
Nudging those parameters by a fixed delta (a "weight adapter") should bias the model toward one form
without retraining.

# Definitions

| Term | Meaning |
|---|---|
| decision point | decoder step at which the model picks between a digit-form token and a cardinal-form token |
| digit side | any digit-related token (all vocab tokens whose text contains a digit) |
| cardinal side | the first token of the spelled-out alternative (found by banning digits and re-decoding) |
| `to_digits` nudge | parameter change that moves digit-vs-cardinal log-odds at the decision point to `1 - mistake_prob` |
| `to_cardinal` nudge | the same, toward `mistake_prob` (i.e. away from digits) |
| knob | a fully addressable parameter: `(module name, index)` |
| mistake prob. thresh | hyperparameter; target probability is `1 - mistake_prob` (0.99 by default) |

> Terminology note: the original brief called the digit-pushing nudge "cardinal nudge". To avoid ambiguity the
> harness names nudges by **direction** (`to_digits`, `to_cardinal`). See [Review](/theory/review-of-approach.md).

# Three phases

1. [Phase 1](/plan/phase-1-existence-proof.md) - prove the model knows both forms.
2. [Phase 2](/plan/phase-2-upstream-discovery.md) - discover upstream decision points (knobs).
3. [Phase 3](/plan/phase-3-control-experiments.md) - control experiments with the nudges applied.

# Falsifiers

* Phase 1: suppressing digits mostly yields garbage or loops instead of number words.
* Phase 2: after the null/FDR test no param set survives, or candidates are unstable across split halves.
* Phase 3: nudged output is no different from the shuffled-value control, or only works by degrading transcription.
