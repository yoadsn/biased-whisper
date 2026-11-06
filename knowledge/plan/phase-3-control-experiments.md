---
type: Plan
title: Phase 3 - control experiments
description: Apply the candidate nudge at several strengths with matched controls and measure the digit share of number mentions on held-out audio.
tags: [plan, phase3, controls, evaluation]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:35:00Z }
---

# Procedure (job `phase3`, code `phase3.py`)

1. Load a Phase 2 run's candidates (`candidate_limit` optionally keeps the top-N by votes) and a statistic (`stat`:
   `mean_all`, `mean_present`, `median_present`).
2. For each nudge type and `strength` apply `delta = strength * stat` to the candidate params (sparse in-place edit, undone after).
3. Controls with identical size and magnitudes: `shuffled_values` (values permuted across the same params) and
   `random_indices` (same values on random params).
4. Decode held-out corpus utterances (excluding Phase 2 samples) with the greedy decoder.
5. Metrics per condition: digit share of number *mentions*, share of outputs containing digits, degenerate rate, `text_drift`
   (change in non-number words vs baseline), and an in-sample probe of mean p(digit) at the Phase 2 decision points.

`strength = 1` means the average per-sample min-norm nudge restricted to the candidate set; because a subset delivers only part of the
intended change, useful strengths are > 1.

# Read the table like this

* The nudge row should move digit share in the intended direction; the controls should not.
* A real effect that comes with large `text_drift` or degenerate outputs is not usable.
* The probe is in-sample; the digit share is held-out.

# Not yet implemented

Digit-logit-bias and prompt baselines, steering-vector-at-one-layer, WER after number normalisation, confidence intervals.
See [Roadmap](/plan/roadmap.md).
