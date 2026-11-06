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
| Offline corpus (HF cache) | 346 knesset utterances (4 extra committee sessions downloaded); 1,646 clips in total |
| explore.ivrit.ai corpus | done: 1,300 clips, 285 episodes, 0.17 s/sample fetch ([sources](/reference/corpus-sources.md)) |
| [Phase 1 v2](/plan/phase-1-existence-proof.md) (span mode) | done: 91 clean word paths of 99 digit samples, prefixes fused, `%` -> `אחוז` |
| [Phase 1.5](/findings/2026-11-06-phase1-5-style-patterns.md) | done: hyphen prefixes are the main digit-style satellite |
| [Budget sweep](/findings/2026-11-06-budget-sweep-small.md) | done on small: 1% of params = 74% of gradient energy; 200-param consensus edits transfer |
| [Phase 2](/plan/phase-2-upstream-discovery.md) (weights) | v1 done on bitfit; to re-run with all params + `seq_contrast` |
| Phase 2N (components) | done: decoder MLP L6-9 + heads L9-10; see [findings](/findings/2026-11-06-phase2n-neuron-edits.md) |
| [Phase 3](/plan/phase-3-control-experiments.md) | 64-neuron gain edit: digit share 58% -> 0% / 82%, controls flat; transfers across domains in both directions ([finding](/findings/2026-11-06-cross-domain-transfer.md)); steering usable at strength <= 1 |

# Next (ordered)

1. Phase 3 refinements: fewer neurons (8/16/32) and a random-neuron control (same layers, same size), WER after number normalisation,
   value-correct verdicts, bootstrap confidence intervals, a held-out *episode* split for explore.
3. Phase 2 (weights) with `all_no_embed` + `seq_contrast` + `sample_k` from the budget sweep; compare its candidates with Phase 2N units.
4. Style carry-over and prompt counterfactual experiments; activation patching of the layer 9-10 heads.
5. Trivial baselines (digit-logit bias, prompting).
6. Reverse cohort (natural word style -> digits) via inverse text normalisation.
7. Larger ivrit.ai checkpoints on the server.
