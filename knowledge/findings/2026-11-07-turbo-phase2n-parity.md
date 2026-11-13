---
type: Finding
title: Turbo Phase 2N produces coherent edits and full inference parity; steering strength needs calibration
description: Owner-authorized component attribution on 200 turbo decision points found concentrated, directionally coherent units; a discovery-only six-edit benchmark passed all 190 token/metric/probe/restoration checks, while strength-2 steering showed output-length side effects.
tags: [finding, phase2n, phase3, turbo, parity, steering, calibration]
status: draft
generated: { by: opencode/gpt-6.1-sol, at: 2026-11-07T07:39:52Z }
sources:
  - id: p2n
    resource: data/runs/phase2n-20261107-093026-e404
    title: Phase 2N on large-v3-turbo, 200 mixed-source discovery points
  - id: bench
    resource: data/runs/phase3_benchmark-20261107-093321-ed32
    title: Discovery-only Phase 3 benchmark, six Phase 2N edits at strength 2
---

# Owner-authorized rerun and stopping rule

After reviewing Phase 1, the owner directed the project to make component-level
Phase 2N the default discovery path, abandon legacy weight-voting Phase 2, and
rerun turbo phases while stopping on problems. Phase 1 was **not repeated**: all
16,112 discovery samples had already completed with span-mode v2 and zero runtime
errors. Its cardinal branch is produced by hard digit-family token exclusion; it
is a strong paired-decoding baseline, not a weight adapter.

UI/defaults were changed first: the Phase 2 tab is now Phase 2N, old Phase 2 is
read-only legacy, Phase 3 selects only completed Phase 2N runs, and interactive
Test rows default to strength 1. Tests: 136 passed / 1 skipped; JavaScript and
OKF checks clean.

# Phase 2N result[^p2n]

`phase2n-20261107-093026-e404`, local large-v3-turbo, first 200 accepted
decision points across the three completed Phase 1 runs (all span-mode
`seq_contrast`; mean digit-vs-cardinal contrast **+4.851**):

| Statistic | Value |
|---|---:|
| Units | 185,120 |
| BH-FDR significant | 63,822 (34.5%) |
| Significant MLP / self / cross | 63,517 / 266 / 39 |
| Voting Gini real / null | 0.807 / 0.250 |
| Vote top-10% share real / null | 64.3% / 18.6% |

Edits written:

- `neurons_to_{digits,cardinal}`: 64 units, 81,920 parameters, L2 0.3253.
- `heads_to_{digits,cardinal}`: 8 units, 655,360 parameters, L2 0.9457.
- `steer_to_{digits,cardinal}`: decoder layer 2 `fc2.bias`, 1,280 parameters,
  alpha 0.6983, best per-layer projection t = 11.55.

Attribution is concentrated at the encoder/decoder interface: encoder MLPs
layers 25-31 and decoder L3 dominate the absolute profile; decoder L3
cross-attention heads 2, 14 and 19 are among the largest signed units. Several
top units have sign consistency 0.85-0.98. This pattern is a causal
**hypothesis**, not a localization proof; generation edits still decide it.

The saved candidate vectors are finite and mirrored by direction.

# Discovery-only Phase 3 parity benchmark[^bench]

`phase3_benchmark-20261107-093321-ed32` used only discovery audio, 8 samples,
8 in-sample probes, all six Phase 2N edits and both controls at strength 2,
with reference and batch sizes 1/4/8. Result: **all 190 checks passed**, covering
exact token identity, metric equality, probe allclose and exact full-weight
restoration. Model cleanup released cached memory afterward.

# Problem found before held-out evaluation

Neuron and head edits at strength 2 decoded with baseline-like timing, but the
real steering edits were much slower:

| Reference condition | Decoder s | Output s |
|---|---:|---:|
| baseline | 2.24 | 0.240 |
| `neurons_to_digits` ×2 | 1.99 | 0.239 |
| `neurons_to_cardinal` ×2 | 2.03 | 0.240 |
| `heads_to_digits` ×2 | 2.00 | 0.235 |
| `heads_to_cardinal` ×2 | 2.04 | 0.240 |
| `steer_to_digits` ×2 | **5.17** | **0.623** |
| `steer_to_cardinal` ×2 | **5.16** | **0.622** |
| steering shuffled controls ×2 | 1.81-2.77 | 0.215-0.332 |
| steering random-index controls ×2 | ~2.00 | ~0.237 |

The benchmark does not retain generated benchmark texts or degeneration metrics,
so this timing/output-length pattern is a warning rather than proof of degraded
language. It is consistent with small's finding that every-position residual
steering is useful around strength 1 and collapses at higher strength. Therefore
the rerun **stops before frozen held-out Phase 3** until steering strengths are
calibrated on discovery audio (for example 0.1/0.3/1) and neuron/head strengths
are selected from measured probe/decode behavior, not inherited from the old
weight-voting 2/5/10/20 grid.

# Next gated step

Run a discovery-only calibration comparison at lower strengths. Proceed to the
50 recording-held-out evaluation only if real edits move the intended objective,
controls stay near baseline, outputs remain finite/coherent, and non-number drift
and degeneration are acceptable.
