---
type: Reference
title: Hyperparameters
description: Every phase hyperparameter, its default and meaning, mapped to the original brief's names.
tags: [reference, hyperparameters, config]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:40:00Z }
resource: src/bwhisper/config.py
---

Source of truth: `src/bwhisper/config.py` (`PHASE1_DEFAULTS`, `PHASE2_DEFAULTS`, `PHASE3_DEFAULTS`). The UI forms are generated
from these; `uv run bw job-types` prints them.

# Phase 1

| key | default | meaning |
|---|---|---|
| `model` | active model | HF id or local path |
| `limit` / `sample_ids` | 50 / null | how many corpus samples (random) or an explicit id list |
| `require_ref_digits` | true | only samples whose reference text has digits |
| `max_new_tokens` | 96 | decode length cap |
| `extra_digit_chars` | "" | extra characters that make a token "digit-related" (e.g. `%`) |

# Phase 2

| key | default | brief's name / meaning |
|---|---|---|
| `phase1_runs`, `verdicts` | - , cardinal + human_good | which decision points are used |
| `max_points` | 100 | cap on decision points |
| `objective` | `mass` | scalar being nudged: `mass`, `pair`, `mass_vs_card` |
| `mistake_prob` | 0.01 | **mistake prob. thresh**; target prob = 1 - this |
| `nudge_mode`, `iter_steps`, `iter_lr` | linear, 15, 0.5 | min-norm 1st order vs damped Gauss-Newton |
| `skip_satisfied` | true | only compute a nudge from samples that need it |
| `param_filter` | `bitfit` | knob space preset (`bitfit`, `decoder_no_embed`, `decoder`, `encoder`, `all_no_embed`, `all`) |
| `include_regex`, `exclude_regex` | "" | further restrict knob names |
| `normalize` | `rms` | per-tensor scaling for selection and voting (`none` to ablate) |
| `sample_pct` | 5 | **upstream params sample pct** |
| `common_pct` | 30 | **Common upstream params percent** |
| `agree_top_pct` | 50 | **top agreement params percent** |
| `group_size` | 3 | vectors summed per voting round |
| `rounds` | 2000 | voting rounds (also used for the null) |
| `fdr` | 0.05 | empirical false-discovery rate for the vote threshold |
| `max_candidates` | 50000 | cap on selected knobs |
| `seed`, `max_store_gb` | 0, 8 | randomness; storage guard (estimate shown in the log before capture) |

# Phase 3

| key | default | meaning |
|---|---|---|
| `phase2_run` | - | run whose candidates are applied |
| `nudge_types` | both | `to_digits`, `to_cardinal` |
| `strengths` | 2, 5, 10, 20 | multipliers of the statistic (bitfit on small needs >= 5 to show an effect) |
| `stat` | `mean_all` | `mean_all`, `mean_present`, `median_present` |
| `candidate_limit` | null | keep top-N candidates by votes |
| `eval_limit` | 50 | held-out utterances |
| `exclude_phase2_samples` | true | keep eval disjoint from fitting data |
| `controls` | shuffled_values, random_indices | matched controls run for every nudge condition |
