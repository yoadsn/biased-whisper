---
type: Plan
title: Approach v2 - style spans, sequence contrast, activation-level attribution
description: Revised plan after the first smoke run and owner feedback - number style is a span-level decision, discovery moves from weight gradients to component attribution, and the deliverable stays a weight edit on any parameters.
tags: [plan, attribution, steering, activation-patching, style, prefixes]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T11:20:00Z }
sources:
  - id: rome
    resource: https://arxiv.org/abs/2202.05262
    title: Meng et al. 2022, Locating and Editing Factual Associations in GPT (causal tracing, rank-one edits)
  - id: atp
    resource: https://arxiv.org/abs/2310.10348
    title: Syed, Rager, Conmy 2023, Attribution Patching Outperforms Automated Circuit Discovery
  - id: kn
    resource: https://arxiv.org/abs/2104.08696
    title: Dai et al. 2022, Knowledge Neurons in Pretrained Transformers
  - id: ig
    resource: https://arxiv.org/abs/1703.01365
    title: Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks (integrated gradients)
  - id: caa
    resource: https://arxiv.org/abs/2312.06681
    title: Rimsky et al. 2024, Steering Llama 2 via Contrastive Activation Addition
  - id: actadd
    resource: https://arxiv.org/abs/2308.10248
    title: Turner et al. 2023, Activation Addition - steering language models without optimization
  - id: repe
    resource: https://arxiv.org/abs/2310.01405
    title: Zou et al. 2023, Representation Engineering
  - id: iti
    resource: https://arxiv.org/abs/2306.03341
    title: Li et al. 2023, Inference-Time Intervention
  - id: kv
    resource: https://arxiv.org/abs/2012.14913
    title: Geva et al. 2021, Transformer Feed-Forward Layers Are Key-Value Memories
  - id: ioi
    resource: https://arxiv.org/abs/2211.00593
    title: Wang et al. 2022, Interpretability in the Wild (IOI circuit, path patching)
  - id: reft
    resource: https://arxiv.org/abs/2404.03592
    title: Wu et al. 2024, ReFT - Representation Finetuning
---

# What changed and why

| v1 assumption | what we learned | v2 |
|---|---|---|
| the style decision is the first digit token | `ב-2025` is `[' ב','-','20','25']`; `באלפיים` starts `' בא'`. The choice is made at the prefix, and also covers `%` vs `אחוז`, units, gender agreement | **style span**: prefix + number + unit is one decision |
| knobs = individual weights | gradient on 244M weights smears one decision over thousands of entries (6-7k bias knobs were significant in v1) | discover in **component space** (neurons, heads, residual directions), then translate to weights |
| "bias" = bias vectors (laptop default `bitfit`) | the owner meant *influence*, over any parameter | default knob space = all weights and biases except embeddings (`all_no_embed`) |
| `to_digits` and `to_cardinal` give different sets | they are exactly antiparallel to first order; on the real model cos -0.75 to -0.96 | keep both, but validate only through interventions with controls |
| pick 5% / 30% by intuition | | [budget sweep](/reference/hyperparameters.md) measures the knee per model |

# 1 - The style span (Phase 1.5 and Phase 1 v2)

* **Phase 1.5 (co-occurrence mining):** for every digit span in the model's outputs, the tokens at offsets -3..+3 are tested for
  enrichment against the background (log-odds ratio with informative Dirichlet prior). The paired Phase 1 outputs (digits vs digits
  banned) are aligned and abstracted into **edit patterns** such as `ב-<D>` => `ב<N>` and `<D>%` => `<N> אחוז`. The output
  includes `satellites.json`, the token set that belongs to the digit style (hyphen, one-letter prefixes before it, `%`, ...).
* **Phase 1 v2:** rewind to the *start of the style span* (the first satellite token, not the first digit) and ban the digit style
  family (digit tokens + hyphen satellites + `%`). The model is then free to write `באלפיים עשרים וחמש` or `שמונים אחוז`.
  This removes the `ב-ב-ב` loops.
* **Paired transcripts:** each verified sample yields `Y_D` (digit style, the free decode) and `Y_C` (word style, the v2 suppressed
  decode) for the same audio.

## Capturing `%` -> `אחוז` and prefix fusion

They are captured by making the objective *sequence-level* over the span instead of one token:

    d_seq = log P(Y_D | audio) - log P(Y_C | audio)        (teacher forced, summed over the divergent region)

The divergent region runs from the first token where `Y_D` and `Y_C` differ until they re-converge. It contains the prefix
decision, the number, the `%`/`אחוז` choice and agreement, so the gradient of `d_seq` covers all of them. No single "probe token"
has to be chosen. Phase 3 additionally scores **style coherence**: digit-style markers (hyphen prefix, `%`) must co-occur with digits,
and word-style markers (fused prefix, `אחוז`) with number words. An edit producing `ב-אלפיים` or `שמונים%` is counted as incoherent.

Reverse direction (natural word-style outputs -> `Y_D`) cannot be produced by banning: Hebrew number words are split into generic
subword tokens (`' ש','מ','ונים'`). It needs inverse text normalisation (rules or an LLM). Open.

# 2 - From weight gradients to component attribution

## Why weights are the wrong unit for discovery

A transformer computes with activations: MLP neurons, attention heads, and directions in the residual stream. A weight matrix entry
only matters through the activation it feeds. One functional unit (for example an MLP neuron that fires for "numeric context, word
style") is implemented by many weights in every layer: its `fc1` row (768 input weights) and bias, and its `fc2` column (768 output
weights), plus everything upstream that builds its input. So a gradient over weights
spreads one decision over many entries. The v1 result of 6-7k significant bias knobs with near-uniform votes is consistent with that.

Gradient magnitude is also *sensitivity*, not *contribution*: it says how much `d` would move per unit change, not how much a component
is actually doing for this decision right now.

## Common practice

| method | what it computes | cost | refs |
|---|---|---|---|
| activation patching / causal tracing | run a clean and a counterfactual input; copy one component's activation from one run into the other; measure the change in the metric | 1 forward per component | [^rome] [^ioi] |
| attribution patching | first-order approximation of patching: `(a_cf - a_clean) . d(metric)/d(a)` for all components at once | 2 forwards + 1 backward | [^atp] |
| gradient x activation / integrated gradients | contribution of each neuron relative to a baseline (zero or mean) along a path | 1 backward (x steps for IG) | [^ig] [^kn] |
| contrastive activation addition (steering vectors) | mean residual activation difference between the two behaviours at a layer; add it at inference | forwards only | [^caa] [^actadd] [^repe] |
| head-level probing + shifting | find heads whose outputs separate the behaviours; shift them along the probe direction | forwards + probes | [^iti] |
| rank-one model editing | rewrite an MLP's key->value association with a rank-one weight update | small solve | [^rome] |
| representation fine-tuning | learn a low-rank intervention on hidden states | training | [^reft] |

Knowledge neurons[^kn] is the closest to the owner's "set of neurons" idea: integrated gradients over FFN neuron activations to find
neurons expressing a relation, then amplify or suppress them. FFN neurons act as key->value memories[^kv], which is why editing
`fc2` columns works.

## The counterfactual problem (important for our data)

At the decision position the digit run and the word run have **identical input** (same audio, same prefix), so their activations are
identical: there is nothing to patch. Contrast must come from an input that changes the choice:

1. **Prompt counterfactual.** Whisper accepts previous-text conditioning (`<|startofprev|> ...`). A prompt written in word style vs
   digit style is a clean counterfactual on the same audio. To test: does it flip the style?
2. **Style carry-over.** Teacher-force the first number of an utterance in word style and check whether later numbers follow without
   any ban. If yes, the model keeps a style state, and the activation at the second decision point differs between the two histories.
   That difference *is* the style representation.
3. **Natural cohorts.** Utterances the model spells out naturally vs ones it writes in digits (in v1, 43 of 106 decodes contained no
   digits). Confounded by content, but usable for a mean-difference direction when averaged over many samples.
4. **No counterfactual:** gradient x activation or IG against a zero/mean baseline, per decision point. Weaker but needs nothing extra.

## Translating components back to weights (the deliverable stays a weight adapter)

| finding | weight edit |
|---|---|
| neuron j in decoder layer L matters | scale `fc2.weight[:, j]` (output gain) or shift `fc1.bias[j]` (firing threshold) |
| attention head h matters | scale the head's slice of `out_proj.weight` |
| residual direction v at layer L (steering vector) | add `alpha * v` to `fc2.bias` or `out_proj.bias` of layer L. This is exactly equivalent to activation addition |
| a key->value association | rank-one update of `fc2.weight` (ROME-style) |

So "influence over the results" is realised by editing weights *or* biases, chosen by what the component analysis finds.

## Effect on the statistics

The decoder of whisper-small has 12 x 3072 = 36,864 MLP neurons, 144 heads and 12 x 768 residual dimensions. Per-sample attribution is
a dense vector of about 40k numbers (160 KB). The existing pipeline (common set, voting, permutation null, FDR) applies unchanged and
gets far more power, because there are about 6,000x fewer hypotheses than weights.

# Revised phase list

1. **Phase 1 v2**: span-level rewind and ban (needs Phase 1.5 satellites); paired transcripts `Y_D`, `Y_C`.
2. **Phase 1.5**: co-occurrence mining (implemented).
3. **Phase 1c (cheap experiments)**: prompt counterfactual; style carry-over. They decide which counterfactual Phase 2N uses.
4. **Phase 2 (weights, kept)**: now with `all_no_embed`, absolute budgets, and `objective=seq_contrast`.
5. **Phase 2N (components, new)**: attribution per decision span for MLP neurons, heads and residual directions; same aggregation;
   plus a contrastive steering vector per layer.
6. **Phase 3**: compare edits - weight candidates, neuron gains, steering-vector bias, rank-one edit - against trivial baselines
   (digit-logit bias, prompting) and matched controls, with the style-coherence metric and quality (drift / WER).

[^rome]: Meng et al. 2022, Locating and Editing Factual Associations in GPT
[^atp]: Syed, Rager, Conmy 2023, Attribution Patching Outperforms Automated Circuit Discovery
[^kn]: Dai et al. 2022, Knowledge Neurons in Pretrained Transformers
[^ig]: Sundararajan et al. 2017, Axiomatic Attribution for Deep Networks
[^caa]: Rimsky et al. 2024, Steering Llama 2 via Contrastive Activation Addition
[^actadd]: Turner et al. 2023, Activation Addition
[^repe]: Zou et al. 2023, Representation Engineering
[^iti]: Li et al. 2023, Inference-Time Intervention
[^kv]: Geva et al. 2021, Transformer Feed-Forward Layers Are Key-Value Memories
[^ioi]: Wang et al. 2022, Interpretability in the Wild
[^reft]: Wu et al. 2024, ReFT: Representation Finetuning
