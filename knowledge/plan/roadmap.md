---
type: Plan
title: Roadmap and status
description: Phases, current status, and the ordered list of next experiments.
tags: [plan, status]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:35:00Z }
---

# Status (2026-11-06)

| Item | State |
|---|---|
| Harness (corpus, jobs, UI, CLI, OKF knowledge) | built, smoke-tested end to end |
| Offline corpus (HF cache) | working, 146 utterances imported |
| explore.ivrit.ai corpus | client written, **untested** - needs the owner's login cookie (see [Corpus sources](/reference/corpus-sources.md)) |
| [Phase 1](/plan/phase-1-existence-proof.md) | done on whisper-small: digits can be suppressed into cardinals; ~22% degenerate (hyphen prefix) |
| [Phase 2](/plan/phase-2-upstream-discovery.md) | done on `bitfit` space; significant distributed candidate sets |
| [Phase 3](/plan/phase-3-control-experiments.md) | first signal: bidirectional, dose-dependent, controls flat - see [findings](/findings/2026-11-06-smoke-run.md) |

# Next (ordered)

1. Pull the real corpus from explore.ivrit.ai (1000 samples) - needs cookie.
2. Re-run on more decision points; confidence intervals; held-out *sources*.
3. Ablations: `normalize`, `objective`, `nudge_mode=iterative`, `param_filter` (`decoder_no_embed`, `all_no_embed`).
4. Trivial baselines: digit logit bias, prompting, single-layer steering vector.
5. Neuron-level activation attribution as an alternative knob definition ([Review](/theory/review-of-approach.md)).
6. Model the hyphen-prefix decision (`ב-`) and the reverse cohort (spelled-out -> digits).
7. Larger ivrit.ai checkpoints on the server (same code; set `model` and `device`).
