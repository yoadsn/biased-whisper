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
selection. The same 100-point experiment completed as `phase2-20261106-203005-7d37`,
reusing all 31 captures and releasing unused GPU models. Its 69 new captures
averaged 2.91 s; both directions saved 50,000 capped candidates. CPU aggregation
then dominated (~35 of 38.8 resumed wall minutes), making voting/null/statistics
the next performance target rather than backprop. No Phase 3 was started. See
[GPU capture benchmark](/findings/2026-11-06-phase2-gpu-capture.md).
Phase 2/2N and Phase 3 are still single-device. No new Phase 2N or Phase 3 job
has been authorized or submitted by this session.

The owner subsequently started `budget_sweep-20261106-211412-59e7` and requested
another performance check. It still uses the legacy CPU gradient/full-sort path,
ten repeated selections and expensive large-support application; the default
largest budget is ~222M weights. See [budget review](/findings/2026-11-06-budget-sweep-throughput.md).
The owner then approved pausing/optimizing/restarting it. Device-native nested
ranking and sorted sparse edits passed parity/resume tests and the real endpoint
benchmark `budget_benchmark-20261106-213235-c809`: exact full-weight restoration,
22.94 GB peak allocation, 0.248 s largest edit/eval/undo. The preserved NumPy
shuffle still costs 10.33 s at the largest budget. The unchanged ten-budget,
24-train/12-test sweep restarted as `budget_sweep-20261106-213326-bff5`, now with
per-budget metrics, timing and resume checkpoints. The old files remain intact;
the legacy sweep had no partial metrics to resume. Still single-GPU/unbatched;
no automatic Phase 3 follows.
The restarted sweep completed in 7.80 minutes with 384 checkpoint records and no
runtime error. Mean held-out cardinal log-odds shift was already about -2 at small
supports, versus -1.85 at the 222M endpoint; shuffled controls were near zero.
Small-support variance is higher; these objective changes still require separate
generated-output validation. NumPy control shuffling/transfers consumed ~41% wall.

Before the owner starts Phase 3, its CUDA path was optimized with same-condition
batching (default 8), tokens-only decoding, reusable CPU mel features, and fresh
encoder outputs under every condition's actual weights. The strength-2 real-model
benchmark passed token/metric/probe/restoration parity, with 4.48x decoder and
1.29x measured overall-inference speedup; encoder computation dominates. Budget
and Phase 3 jobs/benchmarks now unload models and allocator buffers after their
frames unwind. A repeat largest-budget diagnostic confirmed cleanup from 22.94 GB
peak to ~18 MB live tensors (~38 MB reserved), excluding CUDA context overhead.
See [Phase 3 validation](/findings/2026-11-06-phase3-throughput.md). Full Phase 3
evaluation has not been started; frozen evaluation audio remains unused.
The remaining default strengths 5/10/20 also passed discovery-only token/metric/
probe/restoration parity at batch 8 (146 checks total across the two benchmarks).
The validated UI/API code is loaded, no jobs remain active, and cleanup leaves
only ~375 MiB CUDA context/library residency rather than the old ~30 GB cache.

The owner then explicitly authorized Phase 3. Run `phase3-20261106-220319-1429`
completed all 25 conditions on 50 frozen recording-held-out clips in 7.35 minutes,
with 1,250 outputs and successful model cleanup. Effects were modest: baseline
digit-mention share 62.67%, strongest cardinal edit 58.67%, strongest digit edit
64.00%; controls and lower-strength metrics flat. No detected degeneration. See
[results and limitations](/findings/2026-11-06-turbo-phase3-results.md).
The owner also requested a [Test tab](/reference/interactive-test.md) for ≤30 s
microphone/upload audio, independent configurable edits plus baseline, using one
cached CUDA model with exact rollback and fresh encoding under each edit.

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
