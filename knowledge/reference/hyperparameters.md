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
| `devices` | [] | explicit CUDA device list for independent Phase 1 replicas; [] preserves single-device behavior |
| `limit` / `sample_ids` | 50 / null | how many corpus samples (random) or an explicit id list |
| `require_ref_digits` | true | only samples whose reference text has digits |
| `max_new_tokens` | 96 | decode length cap |
| `extra_digit_chars` | "" | extra characters that make a token "digit-related" (e.g. `%`) |
| `span_mode` | true | v2: rewind to the start of the style span (prefix + hyphen), ban digits, `%`-like symbols and a hyphen after a one-letter prefix; decision = first divergent token |
| `satellites_run` | null | `phase1_5` run whose `satellites.json` adds satellite tokens; null = latest, else built-in rules |

# Phase 1.5 (`phase1_5`, CPU, tokenizer only)

| key | default | meaning |
|---|---|---|
| `phase1_runs` | all done | which Phase 1 runs to mine |
| `window` | 3 | token offsets examined before/after each digit span |
| `min_count`, `z_min`, `min_share` | 3, 3.0, 0.05 | satellite selection rule (count, enrichment z, share of spans) |
| `include_reference` | true | also tabulate the human reference texts |

# Phase 2

| key | default | brief's name / meaning |
|---|---|---|
| `capture_backend` | `auto` | device-native linear capture on CUDA; `reference` retains original CPU path; `device` explicitly chooses native selection |
| `resume_run` | null | stopped compatible Phase 2 run whose complete captures are reused without modifying originals |
| `release_unused_models` | true | evict cached replicas other than the active model |
| `phase1_runs`, `verdicts` | - , cardinal + human_good | which decision points are used |
| `max_points` | 100 | cap on decision points |
| `objective` | `auto` | `auto` = `seq_contrast` for span-mode points (log P(digit-style span) - log P(word-style span), teacher forced), else `mass`; also `pair`, `mass_vs_card` |
| `mistake_prob` | 0.01 | **mistake prob. thresh**; target prob = 1 - this |
| `nudge_mode`, `iter_steps`, `iter_lr`, `iter_tol` | linear, 15, 0.5, 0.05 | min-norm 1st order vs damped Gauss-Newton until within `iter_tol` of the target |
| `skip_satisfied` | true | only compute a nudge from samples that need it |
| `param_filter` | `all_no_embed` | knob space: all weights and biases except token/position embeddings. Others: `all`, `decoder_no_embed`, `decoder`, `encoder`, `bitfit` |
| `include_regex`, `exclude_regex` | "" | further restrict knob names |
| `normalize` | `rms` | per-tensor scaling for selection and voting (`none` to ablate) |
| `sample_pct` | 5 | **upstream params sample pct** (used only when `sample_k` is null) |
| `sample_k` | 1,000,000 | absolute per-sample budget (8 MB per sample per nudge type); pick it from the budget sweep |
| `common_pct` | 30 | **Common upstream params percent** |
| `common_k` | null | absolute cap on the common set |
| `agree_top_pct` | 50 | **top agreement params percent** |
| `group_size` | 3 | vectors summed per voting round |
| `rounds` | 2000 | voting rounds (also used for the null) |
| `fdr` | 0.05 | empirical false-discovery rate for the vote threshold |
| `max_candidates` | 50000 | cap on selected knobs |
| `seed`, `max_store_gb` | 0, 8 | randomness; storage guard (estimate shown in the log before capture) |

# Budget sweep (`budget_sweep`)

Measures, for budgets k = fraction x total knobs, the energy captured, the real self-effect, and the held-out transfer of a consensus
edit restricted to k params (with a shuffled control). Keys: `fractions`, `train_points`, `test_points`, `direction`, `xfer_target`
(predicted mean log-odds change the transfer edit is scaled to), plus `param_filter`, `objective`, `normalize`.

| key | default | meaning |
|---|---|---|
| `budget_backend` | auto | CUDA uses device-native gradients, nested rankings and sorted sparse edits; `reference` retains the legacy CPU implementation |
| `resume_run` | null | stopped compatible device sweep with `plan.json` and per-budget measurements; legacy sweeps cannot resume |
| `release_unused_models` | true | release idle cached replicas before the sweep |
| `release_model_after_run` | true | unload models and device allocator buffers after job frames unwind, including cancellation/failure; also applies to `budget_benchmark` |

The device backend checkpoints each completed budget/calibration point, the train
consensus, and NumPy control RNG state. Resume validates the scientific config,
model signature and point order. Interrupted training before consensus was saved
replays gradients but reuses completed metrics. The shuffled control retains NumPy
permutation ordering, so its selected values still visit CPU.

On the 740M-weight turbo space, the unchanged default 0.3 endpoint means ~222M edited
weights. Run `budget_benchmark` with the same config (or the UI's smallest/largest
benchmark button) to check endpoint timing, peak CUDA allocation and exact full-weight
restoration. Memory guards fail rather than silently truncating the grid. This is
still a single-device, unbatched sweep. See [throughput findings](/findings/2026-11-06-budget-sweep-throughput.md).

# Phase 2N (`phase2n`)

| key | default | meaning |
|---|---|---|
| `towers` | decoder, encoder | which towers' neurons and heads are attributed |
| `map` | `full` | `full` (all positions) or `decision` (decoder units at the decision step only) |
| `fdr` | 0.05 | Benjamini-Hochberg level for the per-unit t-test |
| `top_neurons`, `top_heads` | 64, 8 | significant units turned into gain edits |
| `gain` | 0.1 | output-gain change per unit at Phase 3 strength 1 (strength 10 = remove or double) |
| `steer_target` | 1.0 | steering edit sized for this predicted mean log-odds change at strength 1 |
| `vote_rounds`, `group_size`, `agree_top_pct` | 1000, 3, 5 | the owner's voting procedure, run on unit attributions |

Phase 3 uses its outputs with `phase2_run=<phase2n run>` and `nudge_types` from the summary's `phase3_nudge_types`
(`neurons_to_cardinal`, `heads_to_digits`, `steer_to_cardinal`, ...).

# Phase 3

CUDA defaults to same-condition batched inference. No encoded states or KV caches
survive an edit/condition boundary: encoder outputs are recomputed under the current
weights. Only weight-independent CPU mel features are reused. Precision, suppression,
candidate ordering, control RNG, strengths and token limits are unchanged.
`phase3_benchmark` compares serial reference tokens, metrics and probe values against
batches (default 1/4/8) on Phase 2 discovery audio only, including nudges/controls
and exact full-weight restoration.

| key | default | meaning |
|---|---|---|
| `inference_backend` | auto | CUDA uses tokens-only batching; `reference` retains serial diagnostic decoding; `batched` explicitly selects the new path |
| `batch_size` | 8 | utterances per encoder/decoder/probe batch; lower explicitly if device memory is constrained; no automatic precision/grid reduction |
| `release_model_after_run` | true | unload cached models and unused GPU memory on success/cancellation/failure, recording `model_cleanup.json` |

Stage timings are in `preparation_timings.json` and `condition_timings.jsonl`;
`inference.json` records backend, peak allocation and cache policy. Progress counts
probe work as well as transcription. There is still no Phase 3 resume facility.
Use explicit recording-held-out `eval_sample_ids` when required; excluding only
Phase 2 sample IDs is not a recording-disjoint holdout guarantee.

Interactive `test_transcribe` deliberately differs from scientific jobs:
`release_model_after_run=false` retains the restored baseline for later audio
comparisons. Its model/statistic/candidate limit/token cap are pinned by the
selected completed Phase 3 run; each `edit_configs` row specifies an edit name,
finite strength and optional label. Baseline is always included, including an
empty config list. CUDA is used when available. See the
[Test reference](/reference/interactive-test.md).

| key | default | meaning |
|---|---|---|
| `phase2_run` | - | run whose candidates are applied |
| `nudge_types` | both | `to_digits`, `to_cardinal` |
| `strengths` | 2, 5, 10, 20 | multipliers of the statistic (bitfit on small needs >= 5 to show an effect) |
| `stat` | `mean_all` | `mean_all`, `mean_present`, `median_present` |
| `candidate_limit` | null | keep top-N candidates by votes |
| `eval_limit` | 50 | held-out utterances |
| `eval_sample_ids` | null | restrict to frozen recording-held-out IDs written by `setup_scale` |
| `exclude_phase2_samples` | true | keep eval disjoint from fitting data |
| `controls` | shuffled_values, random_indices | matched controls run for every nudge condition |
