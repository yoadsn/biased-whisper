---
type: Plan
title: Roadmap and status
description: Phases, current status, and the ordered list of next experiments.
tags: [plan, status]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T13:50:00Z }
---

The current design is [Approach v2](/plan/approach-v2.md); v1 phase docs remain valid for their parts.

# Server preparation (2026-11-06)

Eight A40 GPUs and the local large-v3-turbo checkpoint are validated. The first
5,799-clip import incorrectly filtered normalized manifest text; its original-digit
ceiling claim is retracted. The [provenance correction](/findings/2026-11-06-original-text-correction.md)
restores true original references; repaired imports and regenerated recording-disjoint
cohorts are now complete. The earlier Phase 1 config was written
but **not submitted**; do not use it as the final prepared cohort.

Explore authentication is configured and both corpora are complete: 10,000 local
and 10,000 Explore clips. Crowd-transcribe has original-digit references in its
source index, but none matches the available aligned subset; no crowd digit
samples were fabricated.

The owner has now **authorized Phase 1 only**. Run
`phase1-20261106-180130-0a91` completed 32 corrected discovery samples on all eight
GPUs without runtime errors (18 cardinal / 8 no_number_words / 6 no_digits).
After a SQLite-lock fix and resumed download, all Phase 1 increments completed:
32 validation + 8,072 current-cohort + 8,008 expanded Explore samples = **16,112
unique discovery samples**, 10,940 automatic cardinal verdicts, zero runtime errors.
The 3,888 recording-disjoint evaluation samples remain untouched.
`setup_scale-20261106-192355-9dca/phase1-runs.json` lists the three disjoint runs.
No further automatic jobs remain. The owner questioned the sample budget and GPU
utilization, but no reduction was approved before completion. Future work should
use smaller reviewed subsets and profile/batch the decoder rather than repeating
this oversized run. See [turbo results](/findings/2026-11-06-turbo-phase1-validation.md)
and [throughput review](/findings/2026-11-06-phase1-throughput.md).
Subsequently, the owner ran Phase 1.5 (`phase1_5-20261106-200747-48a7`, done) and
started weight-space Phase 2 (`phase2-20261106-200924-2731`, 100 Explore decision
points, all_no_embed). A [throughput review](/findings/2026-11-06-phase2-throughput.md)
identified dense CPU gradient export/selection and idle cached replicas. The owner
then approved optimization: the source job was stopped with 31 complete captures
preserved; a real-model benchmark passed numerical parity with 12–13x faster
selection. The same 100-point experiment resumed as `phase2-20261106-203005-7d37`,
reusing all 31 captures and releasing unused GPU models. See
[GPU capture benchmark](/findings/2026-11-06-phase2-gpu-capture.md).
Phase 2/2N and Phase 3
are still single-device. No new budget, Phase 2N or Phase 3 job has been authorized
or submitted by this session.

# Laptop status (2026-11-06, afternoon)

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
