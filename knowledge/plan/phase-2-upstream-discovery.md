---
type: Plan
title: Phase 2 - upstream discovery
description: Compute per-decision-point nudges, keep top-% params, intersect across samples, vote over random groups against a permutation null, and select knobs by FDR.
tags: [plan, phase2, gradients, statistics]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:35:00Z }
---

# Pipeline (job `phase2`; `phase2.py`, `aggregate.py`, `paramspace.py`)

1. **Knob space.** `param_filter` preset (+ include/exclude regex) selects parameters; they are flattened in a stable order
   so a knob is `(module, index)` and a flat int.
2. **Per decision point** (verified Phase 1 row): forward the decoder over `prefix_ids`; objective `d` = digit-vs-cardinal log-odds
   (`mass`, `pair` or `mass_vs_card`); backprop to the knob space; `g = grad d`.
3. **Two nudges.** Target `d* = ln((1-eps)/eps)`; `to_digits` needs `delta = d* - d0`, `to_cardinal` needs `delta = -d* - d0`.
   `linear`: `n = delta * g / |g|^2`. `iterative`: damped Gauss-Newton on the real model. `skip_satisfied` skips samples already past
   the target.
4. **Per-sample set.** Top `sample_pct` of `|n|` after per-tensor RMS scaling (`normalize=rms`); stored sparse
   `(idx, raw value, per-tensor rms)` under `data/runs/<id>/vectors/<type>/`.
5. **Common set.** Counter over samples; union; top `common_pct` of the union by count (stable tie-break by flat index).
6. **Vote.** `rounds` times: draw `group_size` samples, vector over the common set (0 where absent), divide by `max|v|`, sum;
   the top `agree_top_pct` (non-zero only) get +1.
7. **Null.** Same with every sample's vector randomly permuted over the common set.
8. **Select.** Smallest vote count with empirical FDR <= `fdr`. If none: "no significant knobs".
9. **Report.** Per candidate: votes, null votes, mean (all / present), median, std, count, sign consistency. Tower/layer/module
   breakdowns of gradient energy and candidates; vote histogram vs null; Gini and top-share; split-half Jaccard.

Outputs: `summary.json`, `candidates_<type>.npz`, `votes_<type>.npz`, `points.jsonl`, `space.json` in the run directory.

# Design notes

See [Review](/theory/review-of-approach.md) for why the objective is a log-odds, why the nudge types are near-mirrors, why the
statistics are permutation-based, and what to try next. Parameters: [Hyperparameters](/reference/hyperparameters.md).
