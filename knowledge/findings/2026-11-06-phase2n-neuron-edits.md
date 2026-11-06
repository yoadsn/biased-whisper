---
type: Finding
title: Phase 2N on whisper-small - 64 neuron gains steer digit vs word style on held-out audio
description: Gradient x activation attribution over 74k neurons and heads (91 span-level decision points) localises the style decision to decoder MLPs in layers 6-9 and a few attention heads in layers 9-10, and a 49k-parameter weight edit on 64 neurons moves held-out digit share from 58% to 0% or 82% against flat shuffled controls.
tags: [finding, phase2n, phase3, neurons, attention-heads, steering, whisper-small]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T13:45:00Z }
sources:
  - id: p1
    resource: data/runs/phase1-20261106-125046-8f42
    title: Phase 1 v2 (span mode), 146 samples, whisper-small (local only)
  - id: p2n
    resource: data/runs/phase2n-20261106-125436-0ef2
    title: phase2n, 91 decision points, objective seq_contrast (local only)
  - id: p3
    resource: data/runs/phase3 run on phase2n-20261106-125436-0ef2
    title: Phase 3, 40 held-out utterances incl. 200 new knesset-committee windows (local only)
---

# Phase 1 v2 (span mode)[^p1]

| | v1 | v2 |
|---|---:|---:|
| `cardinal` (clean word path) | 73 (two runs, 146 samples) | 91 |
| `degenerate` | ~22% of digit samples | 3 |

The word path now fuses prefixes and swaps symbols together: `ל-90%` -> `לתשעים אחוז`, `מ-24 שנה` -> `מעשרים וארבע שנה`,
`ב-2003` -> `באלפיים ושלוש`. For 30 of 99 digit spans the decision is 1-2 tokens before the first digit (prefix and hyphen).

Fixes that made this work:
* prefixes whose fused number words start with a different token (`' ב'` -> `' בא'`, `' בש'`; also `ל`, `מ`, `ה`) are banned at
  the span start; `ו` and `מה` are not (their fused forms start with the bare token);
* maqaf `־` and niqqud are byte-split by the BPE (byte 0xD6); those byte tokens are banned on the word path. Without this the
  model escaped into maqaf and niqqud garbage.

Open: the verdict does not check the *value* (`2024` -> `בארבעים ועשרים` counts as cardinal). Needs inverse text normalisation.

# Phase 2N - where the decision lives[^p2n]

Gradient x activation for 74,160 units (decoder and encoder MLP neurons, self- and cross-attention heads), objective
`seq_contrast` over the style span. Sanity check: for 46 neurons, zeroing a neuron changed the objective as predicted
(r = 0.97, sign agreement 98%, median real/predicted 0.98).

* Share of |mean attribution|: decoder MLP 47%, encoder MLP 36% (mostly the last encoder layer), decoder cross-attention 7%,
  encoder self-attention 5%, decoder self-attention 4%.
* Decoder MLP layers 6-9 carry the most (|mean| per layer rises from about 2.4 at layer 1 to 13.0 at layer 9).
* Largest single units are **attention heads**: decoder layer 10 cross-attention heads 9 and 3 push toward digits (+1.35, +1.22);
  decoder self-attention layer 10 head 1 and layer 9 head 9 push toward words (-1.02, -0.71). Decoder self-attention reads the text
  written so far, so these are candidates for a "keep the style consistent" mechanism (to test with the carry-over experiment).
* **Two methods agree:** decoder layer-9 neurons 2193 (+0.60) and 2240 (-0.51) are top units here and were also top v1 Phase 2 weight
  candidates (`decoder.layers.9.fc1.bias[2193]`, `[2240]`), found with a different objective (`mass`) and from weight gradients.
* 18,093 units pass BH-FDR 5%: the decision is consistent but distributed. The top 100 units hold 16% of |mean attribution|; votes are
  strongly concentrated (Gini 0.83 vs null 0.38).
* Selection for edits uses significance as a gate and **effect size** to rank (ranking by t picked consistent but negligible units).

# Phase 3 - edits on held-out audio[^p3]

40 held-out utterances (not used in Phase 2N), baseline digit share 58%, digit-output rate 62%.

| edit | params | strength | digit share | digit-output rate | text drift | degenerate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | - | - | 58% | 62% | 0 | 0% |
| 64 neurons, gain -> words | 49,152 | 3 (30%) | **29%** | 33% | 0.15 | 0% |
| 64 neurons, gain -> words | 49,152 | 10 (100%) | **0%** | 0% | 0.30 | 0% |
| shuffled control | 49,152 | 10 | 54% | 57% | 0.20 | 3% |
| 64 neurons, gain -> digits | 49,152 | 3 | 68% | 85% | 0.14 | 5% |
| 64 neurons, gain -> digits | 49,152 | 10 | 82% | 72% | 0.53 | 12% |
| 8 heads, gain -> words | 393,216 | 10 | 11% | 12% | 0.25 | 5% |
| 8 heads, gain -> digits | 393,216 | 10 | 71% | 88% | 0.18 | 3% |
| steering bias, layer 10 | 768 | 3 | - | - | 0.87 | 85% |
| steering bias -> words | 768 | 0.3 | 40% | 42% | 0.09 | 0% |
| steering bias -> words | 768 | 1 | **9%** | 15% | 0.23 | 3% |
| shuffled control | 768 | 1 | 65% | 70% | 0.12 | 3% |
| steering bias -> digits | 768 | 1 | 80% | 88% | 0.25 | 7% |

* **Neuron gains work in both directions** with flat controls. Toward words they reach 0% digits at strength 10 without loops.
* **Cost:** text drift (non-number words changed vs baseline) is 0.15 at strength 3, but the shuffled control also drifts 0.10-0.20,
  so part of the drift is any perturbation of these weights. A real quality metric (WER after number normalisation) is still missing.
* **A single bias vector steers about as well as 64 neurons:** the layer-10 `fc2.bias` steering edit (768 numbers, the mean
  residual gradient direction at the decision step) reaches 9% digit share at drift 0.23, against 29% at drift 0.15 for neurons at
  strength 3. Its default size was mis-calibrated (sized for one log-odds unit at the decision step, but added at every position),
  so useful strengths are 0.1-1, not 3-10. Same eval set (seed 0, knesset source) as the neuron rows.
* For comparison (different eval set, so indicative only): v1 bias-only knobs (6,892 biases) reached 18% digit share at strength 20
  with drift 0.16; 64 neurons reach 29% at drift 0.15 and 0% at drift 0.30.

# Next

1. Position-limited steering (only around number spans) as an inference-time comparison; steering at other layers; combine
   steering with a few neurons.
2. Fewer units: does a top-8 or top-16 neuron edit keep most of the effect?
3. WER after number normalisation; value-correct cardinal verdict.
4. Style carry-over and prompt counterfactual experiments (approach v2), to explain the layer 9-10 heads.
5. The 1000-sample explore.ivrit.ai corpus (other domains than Knesset), then larger models.

[^p1]: Phase 1 v2 (span mode), 146 samples, whisper-small
[^p2n]: phase2n, 91 decision points, objective seq_contrast
[^p3]: Phase 3, 40 held-out utterances
