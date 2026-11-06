---
type: Plan
title: Roadmap and status
description: Phases, current status, and the ordered list of next experiments.
tags: [plan, status]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T13:50:00Z }
---

The current design is [Approach v2](/plan/approach-v2.md); v1 phase docs remain valid for their parts.

# Status (2026-11-06, afternoon)

| Item | State |
|---|---|
| Harness (corpus, jobs, UI, CLI, OKF knowledge) | built; UI tabs for every job type |
| Offline corpus (HF cache) | 346 knesset utterances (4 extra committee sessions downloaded) |
| explore.ivrit.ai corpus | cookie imported, API format known (episodes -> hits); adapter and 1000-sample pull in progress |
| [Phase 1 v2](/plan/phase-1-existence-proof.md) (span mode) | done: 91 clean word paths of 99 digit samples, prefixes fused, `%` -> `אחוז` |
| [Phase 1.5](/findings/2026-11-06-phase1-5-style-patterns.md) | done: hyphen prefixes are the main digit-style satellite |
| [Budget sweep](/findings/2026-11-06-budget-sweep-small.md) | done on small: 1% of params = 74% of gradient energy; 200-param consensus edits transfer |
| [Phase 2](/plan/phase-2-upstream-discovery.md) (weights) | v1 done on bitfit; to re-run with all params + `seq_contrast` |
| Phase 2N (components) | done: decoder MLP L6-9 + heads L9-10; see [findings](/findings/2026-11-06-phase2n-neuron-edits.md) |
| [Phase 3](/plan/phase-3-control-experiments.md) | 64-neuron gain edit: digit share 58% -> 0% / 82%, controls flat; steering mis-calibrated |

# Next (ordered)

1. 1000-sample explore.ivrit.ai corpus; rerun Phase 1 v2 / 2N / 3 on it (other domains than Knesset).
2. Phase 3 refinements: steering strengths 0.1-1, fewer neurons (8/16/32), random-neuron control, WER after number normalisation,
   value-correct verdicts, confidence intervals.
3. Phase 2 (weights) with `all_no_embed` + `seq_contrast` + `sample_k` from the budget sweep; compare its candidates with Phase 2N units.
4. Style carry-over and prompt counterfactual experiments; activation patching of the layer 9-10 heads.
5. Trivial baselines (digit-logit bias, prompting).
6. Reverse cohort (natural word style -> digits) via inverse text normalisation.
7. Larger ivrit.ai checkpoints on the server.
