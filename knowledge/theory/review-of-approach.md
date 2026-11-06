---
type: Review
title: Review of the approach
description: Critique of the three-phase plan - what is sound, what was wrong or ambiguous, what the harness changed, and which risks and alternatives remain open.
tags: [review, methodology, statistics, attribution, whisper]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:00:00Z }
---

# Verdict

The skeleton is sound and is the classic **attribute → intervene** loop: use a gradient to localise, then test the
localisation by editing weights and measuring behaviour. Phase 1 (a *minimal pair* at one decoder step) is the
right foundation. Most issues below are about definitions and statistics, not about the idea. Items marked
**[fixed]** are already handled in the harness; **[open]** items are risks or experiments to schedule.

# Phase 1 - existence proof

* **Sound.** Rewinding to the step before the first digit token and banning digit tokens gives a clean minimal pair
  `(digit continuation, cardinal continuation)` with identical history. We also get the cardinal-side token for free,
  which Phase 2 needs. **[fixed]** Phase 1 rows are stored as Phase 2 decision points.
* **"All digit-related tokens" is large.** whisper-small's vocabulary has ~427 tokens containing a digit
  (`" 20"`, `"25"`, `"2024"`, `"1"` ...). **[fixed]** The ban set is computed from the tokenizer, not hand-listed;
  `extra_digit_chars` can add `%` etc.
* **Success needs more than "no digits".** Banning digits can also give garbage. **[fixed]** Auto-verdict checks that the
  replaced span contains Hebrew number words (lexicon + prefix stripping), that the rest of the sentence is unchanged
  (`rest_match`), and flags repetition loops. A human good/bad override feeds Phase 2.
* **Empirical surprise - the decision is sometimes made one token earlier.** Hebrew attaches prefixes to digits with a
  hyphen: `ב-2025`. Whisper emits the token `ב-` *and then* the digits. Once `ב-` is emitted the model is committed to
  the digit style, and banning digits afterwards produces loops (`ב-ב-ב-ב-...`): 8 of 36 digit samples in the first run
  (see [findings](/findings/2026-11-06-smoke-run.md)). **[fixed]** such rows are marked `hyphen_committed` and the
  `degenerate` verdict keeps them out of Phase 2 by default. **[open]** model the earlier decision (`ב-` vs `ב` + number
  word) as its own decision point - arguably the real "upstream" decision for prefixed numbers.
* **Multi-token numbers.** `2025` is tokenised `20|25`. Only the first token is the decision; later tokens are conditional.
  The harness records the first digit token only. **[open]** study later tokens (do number *words* get continued consistently?).
* **The reverse direction is missing.** Some utterances already produce number words (4/40 in the first run). They are the
  natural `to_digits` cohort, but they have no digit token to define a decision point. **[open]** Phase 1b: ban number-word
  tokens and check digits appear, giving symmetric minimal pairs.

# Phase 2 - nudges and gradients

## Definition of the two nudges

* A single "target token logit" is the wrong scalar: digits are many tokens, and a "99% sampler" statement is about a
  *distribution*. **[fixed]** The objective is a log-odds `d`: `mass` (`logsumexp(digit tokens) - logsumexp(others)`, default),
  `pair` (digit token vs cardinal token) or `mass_vs_card`. 99% means `d* = ln(0.99/0.01) = 4.6`. The required logit change is
  `delta = ±d* - d0`.
* The "cardinal"/"non-cardinal" naming was ambiguous (the sentence describes choosing the *digit* token). **[fixed]** Named by
  direction: `to_digits`, `to_cardinal`.

## The two nudges are the same vector (important)

With one objective, both nudges are `delta * grad(d) / |grad(d)|^2`: **identical direction, opposite sign**. Their
top-5% sets coincide and agreement between them proves nothing. Real differences come only from:

1. **cohort** - which samples need the nudge. A sample already at 99% digits needs no `to_digits` nudge; computing it anyway
   gives a vector pointing the *wrong way* and wrecks cross-sample sign agreement (observed in the first run: samples with
   `d0 = +10` produced negative `to_digits` vectors). **[fixed]** `skip_satisfied` (default on): a nudge is computed only
   from samples that need it, so the two types use different cohorts (16 vs 25 in the first run).
2. **nonlinearity** - **[fixed]** `nudge_mode=iterative` runs damped Gauss-Newton steps on the *real* model until the target
   is reached; the accumulated change differs per direction.

Validation of the knobs therefore has to come from [Phase 3](/plan/phase-3-control-experiments.md) with controls, not from
agreement between the two sets.

## Gradient magnitude is a weak proxy for causal importance

First-order sensitivity ignores saturation (softmax, GELU, LayerNorm) and scale. Options, cheapest first:

* gradient x weight (Taylor attribution, scale-aware) - **[open]**, trivial to add
* integrated gradients along the nudge path - **[open]**
* activation-level attribution ("attribution patching"): `d(d)/d(activation) * activation` per decoder neuron / head at the
  decision position gives a *signed scalar per neuron*, tiny to store, and is what "the set of neurons" in the hypothesis
  literally means. A neuron's weights (its `fc1` row/bias, `fc2` column) are then the knobs. **[open]**, recommended next.
* causal confirmation by ablation/patching (zero or swap candidate neurons, watch `d`) - **[open]**; Phase 3 is the weight-space analogue.

## Should gradients be normalised by layer?

Yes - but call it **per-tensor scaling**, not LayerNorm. Raw gradient magnitude depends on fan-in, activation scale and
distance to the loss, so a global top-5% concentrates in a few tensors. **[fixed]** `normalize=rms` divides each tensor's
nudge by its own RMS for *selection and voting*; raw values are kept for statistics and application. `normalize=none` is
the ablation. Alternatives to sweep: per-tensor top-k (stratified, every tensor contributes 5%) and rank-percentile.
Compare candidate sets and Phase 3 effect for each.

## Tied embeddings

`proj_out.weight` is tied to `embed_tokens.weight`. The gradient on the digit tokens' embedding rows is trivially the
largest (it *is* the readout) and equals a logit bias. **[fixed]** presets exclude embeddings except `all`/`decoder`. Treat the
readout as a separate baseline, not an "upstream" knob.

## Scale and storage

whisper-small has 244M params; 5% = 12M per sample per type; a dense "30% of the union" vector per sample is not storable for
1000 samples (and large-v3 is 6x bigger). **[fixed]**

* sparse storage `(idx, value, per-tensor rms)`; common set via a `uint16` counter; two-pass streaming; storage estimate with a
  hard guard (`max_store_gb`) shown before the run starts;
* `param_filter` presets. `bitfit` (biases + LayerNorm, 272k knobs on small) is the laptop default and the natural adapter
  family: a bias delta on `out_proj.bias`/`fc2.bias` is exactly an additive **residual-stream steering vector**. (In the first
  run the highest gradient-energy tensors were precisely these residual-write biases.)
* **[open]** coarser units: whole rows/neurons or attention heads as knobs with a signed gain scalar, which is also a better match
  to "adapters" (LoRA/BitFit) than single scalar weights.

## Vector construction and the zero-fill

Filling absent params with 0 is fine for "does this param show up", but with sparse vectors most entries of a 3-vector sum
are exactly 0, so "top 50% by magnitude" would pick arbitrary zeros. **[fixed]** only non-zero entries can vote; `agree_top_pct`
is a cap relative to the common set (effective set = `min(cap, nonzeros)`). "Rescale to [-1, 1]" is implemented as division by
`max|v|` (zero stays zero, signs preserved).

# Phase 2 - choosing the leading bunch (the statistics question)

What the random-group voting estimates is a *cross-sample consistency* statistic. Recommended reading, all implemented:

1. **Null by permutation.** Run the identical voting on the same vectors with each sample's values randomly permuted over the
   common set. This destroys cross-sample alignment but keeps each sample's magnitude profile. Compare real vs null vote counts.
2. **Empirical FDR threshold.** Smallest vote count `t` with `#null>=t / #real>=t <= fdr`. If none, the harness reports
   "no significant knobs" rather than a top-N list. (A binomial null is anti-conservative: groups overlap and samples are few.)
3. **Stopping rule - split-half reliability.** Votes from even vs odd rounds; Jaccard of top-1% sets. Stop when it plateaus (~0.9).
4. **Descriptive concentration.** Gini / top-1% / top-10% vote share, real vs null. The "exponential, top 10% hold 90%" intuition
   is not a test: under the null votes are ~Binomial, so the informative thing is the *excess* heavy tail over the null.
5. **Direct estimator.** Per-param mean, std, sign-consistency and `mean/std*sqrt(n)` across samples are computed alongside, since
   random triples are a Monte-Carlo way of estimating the same thing. With `N` points there are only `C(N,3)` distinct triples
   (N=25 -> 2300), so large `rounds` just re-sample the same groups; more *points* add power, more rounds do not.
6. **[open]** Cross-validate: pick candidates on half the decision points, check sign-consistency on the other half.

# Phase 3 - control experiments

* **Controls are essential.** **[fixed]** every nudge is paired with `shuffled_values` (same params, same magnitudes, values
  permuted) and optionally `random_indices` (same values at random params).
* **Strength calibration.** The mean min-norm nudge *restricted to a candidate subset* delivers only a fraction of the intended
  logit shift. `strength` multiplies it; sweep it and watch side effects (drift, loops).
* **Held-out evaluation.** Eval samples exclude Phase 2 points. The "probe" at decision points is in-sample by construction.
* **Trivial baselines to beat [open]:** (a) a constant logit bias on digit tokens (readout), (b) a decoder prompt/prefix with
  number words or digits, (c) one residual steering vector at one layer. If a baseline matches the knobs, the knobs are not
  interesting. (d) fine-tune delta (LoRA on digit-normalised vs word-normalised text) as an upper reference.
* **Confound - style vs number.** The "knobs" may encode a transcript *normalisation style* (punctuation, `%` vs `אחוז`, hyphen
  prefixes, dates), inherited from mixed training sources. Measure drift on non-number text (implemented as `text_drift`) and
  compare per corpus source.

# Corpus

* Regex `\d` also matches identifiers, `A4`, list numbers. **[open]** add a "number-like" filter and a per-source breakdown.
* The reference transcripts use digits *inconsistently across sources*; that inconsistency is what the model learned.

# Priority list

1. Grow decision points (N >= 100) and rerun Phase 2/3 on `bitfit` - cheap, decides whether anything is there.
2. Ablate `normalize`, `objective`, `nudge_mode` and compare candidate overlap and Phase 3 effect.
3. Add activation-level attribution and trivial baselines.
4. Model the `ב-` decision point and the reverse-direction cohort.
5. Only then scale to bigger models / all parameters.
