---
type: Finding
title: Budget sweep on whisper-small - the signal needs far fewer than 30% of parameters
description: Over all 200M non-embedding parameters, 1% holds 74% of the gradient energy, and a consensus edit on only 200 to 20,000 parameters transfers to held-out utterances at the predicted size, against a flat shuffled control.
tags: [finding, budget, whisper-small, storage, hyperparameters]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T13:10:00Z }
sources:
  - id: run
    resource: data/runs/budget_sweep-20261106-121425-e2b8
    title: budget_sweep, whisper-small, all_no_embed, v1 points, objective mass, to_cardinal, 24 train / 12 test (local only)
---

# Setup[^run]

`yoad/whisper-small`, `param_filter=all_no_embed` (200,406,528 knobs), v1 decision points (objective `mass`), direction
`to_cardinal`, 24 train and 12 held-out decision points, per-tensor RMS selection. Weight RMS of the model is 0.043.

# Results

| fraction | k | energy (raw top-k) | energy (selection rule) | self effect | held-out effect (log-odds) | shuffled | per-param edit RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | 200 | 0.03 | 0.01 | 0.85 | -3.33 +/- 1.30 | -0.12 | 1.2e-2 |
| 1e-5 | 2,004 | 0.08 | 0.05 | 0.90 | -3.10 +/- 1.14 | 0.00 | 2.2e-3 |
| 1e-4 | 20,041 | 0.20 | 0.15 | 0.92 | -3.03 +/- 1.02 | 0.01 | 4.1e-4 |
| 1e-3 | 200,407 | 0.43 | 0.36 | 0.93 | -2.93 +/- 1.00 | 0.00 | 9.2e-5 |
| 1e-2 | 2,004,065 | 0.74 | 0.65 | 0.92 | -2.78 +/- 0.97 | 0.00 | 2.3e-5 |
| 5e-2 | 10,020,326 | 0.91 | 0.86 | 0.92 | -2.67 +/- 0.93 | 0.00 | 8.9e-6 |
| 3e-1 | 60,121,958 | 0.995 | 0.99 | 0.92 | -2.59 +/- 0.89 | 0.00 | 3.3e-6 |

*Held-out effect*: the consensus direction (mean normalised gradient over train points), restricted to the top-k, scaled so its
first-order prediction on *train* points is -3, then applied and measured on *held-out* points.

# Reading

* **Gradient energy is concentrated:** 1% of the parameters holds 74%, 5% holds 91%. Keeping 30% of a union is unnecessary.
* **The cross-utterance direction is very low-dimensional in effect:** a 200-parameter consensus edit moves held-out log-odds by the
  predicted amount, and shuffled values do nothing. Smaller supports just need larger per-parameter changes (at k = 200, 27% of the
  typical weight size; at k = 20k, 1%).
* **Self effect** (a sample's own min-norm edit on its own top-k) is 85-93% at every k: one decision can be flipped by almost any
  small support, so per-sample sets say little on their own. Consensus and transfer are what matter.
* **Caveat:** this is the log-odds at decision points, not full decoding. Large sparse edits may have side effects that only Phase 3
  (drift, degenerate rate) can show.

# Decision

* `sample_k` default 1,000,000 (0.5% on small, 8 MB per sample per nudge type) is ample. On bigger models start at 1e-3 to 5e-3 of
  the knobs and re-run this sweep.
* Use `common_k` of 1e5 to 1e6 rather than 30% of the union.
* Re-run the sweep per model (large-v3 has 1.5B knobs) and for `objective=seq_contrast`.

[^run]: budget_sweep, whisper-small, all_no_embed, 24 train / 12 test
