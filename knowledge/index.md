---
okf_version: "0.2"
---

# Start here

* [Hypothesis](theory/hypothesis.md) - the theory being tested, in the owner's words, with definitions
* [Review of the approach](theory/review-of-approach.md) - critique, fixes made, open risks and alternatives
* [Latin vs transliteration hypothesis](theory/latin-script-hypothesis.md) - second axis: English words in Latin letters vs Hebrew transliteration, mapped onto the same harness

# Plan

* [Roadmap](plan/roadmap.md) - phases, status and next steps
* [Approach v2](plan/approach-v2.md) - style spans, sequence contrast, component attribution, translation to weight edits
* [Phase 1 - existence proof](plan/phase-1-existence-proof.md) - does the model know both forms?
* [Phase 2 - legacy weight discovery](plan/phase-2-upstream-discovery.md) - deprecated; new discovery is Phase 2N in Approach v2
* [Phase 3 - control experiments](plan/phase-3-control-experiments.md) - do the knobs steer output?
* [Phase 4 - online LoRA training](plan/phase-4-lora.md) - independent frozen-teacher pseudo-label training, natural cardinal targets, monitoring UI and unsuppressed evaluation

# Reference

* [Controller validation](reference/controller-validation.md) - ten-sentence correctness checks, a shared hard 100-sentence budget, calibration/confirmation separation and exact uncertainty reporting
* [Phase 4 parallel training](reference/phase4-parallel.md) - eight-GPU teacher cache, DDP with fixed global batch, resumable checkpoints, monitoring and A40 NCCL routing
* [Hyperparameters](reference/hyperparameters.md) - every knob, default, and what it means
* [Harness architecture](reference/architecture.md) - modules, jobs, data layout, CLI/API
* [Corpus sources](reference/corpus-sources.md) - explore.ivrit.ai and the offline HF-cache source
* [A40 server preparation](reference/server-setup.md) - aligned manifests, CUDA environment, eight-device Phase 1 and frozen recording-disjoint cohorts
* [GPU-native Phase 2 capture](reference/phase2-gpu-capture.md) - sparse device selection, CPU-reference benchmark, preserved-capture resume and timing artifacts
* [Interactive Test tab](reference/interactive-test.md) - local mic/upload comparisons, independent edit configs, baseline restoration and a reused CUDA model

# Findings

* [2026-11-13 Latin-axis Phase 1 POC](findings/2026-11-13-latin-phase1-poc.md) - banning Latin tokens yields Hebrew spellings for short words and most acronyms, not phrases; 332 clips, turbo, laptop; 90% of short words convert
* [2026-11-13 Cloud Run LoRA demo](findings/2026-11-13-cloud-run-lora-demo.md) - selected digit/cardinal checkpoints and deployment semantics

* [2026-11-07 Phase 4 parallel validation](findings/2026-11-07-phase4-parallel-validation.md) - eight-rank NCCL/update checks, serial/parallel frozen-teacher equivalence and NVLink-only P2P workaround; synthetic only
* [2026-11-07 controller establishment](findings/2026-11-07-controller-establishment.md) - bounded correctness sampling, rejected digit-only wins, balanced rediscovery and conservative integer-value filtering of contrast targets
* [2026-11-07 microphone nudge audit](findings/2026-11-07-microphone-nudge-audit.md) - saved owner failure uses correct nudges; held-out digit share overstates correctness; head gain ×10 correctly converts the saved 1948 sentence in one bounded follow-up
* [2026-11-07 turbo Phase 2N held-out](findings/2026-11-07-turbo-phase2n-heldout.md) - calibrated 50-clip control: decoder steering reaches 21.7% digit share at strength 0.3 with no degeneration; unit gains are gentler
* [2026-11-07 turbo Phase 2N parity](findings/2026-11-07-turbo-phase2n-parity.md) - 200-point component attribution, six coherent edits, 190/190 discovery-only parity checks, and a steering-strength warning before held-out evaluation
* [2026-11-07 turbo steering diagnosis](findings/2026-11-07-turbo-steering-diagnosis.md) - small/turbo method mismatch, weak sign consistency, uncalibrated final edits and candidate-count versus gradient localization
* [2026-11-06 turbo Phase 3 results](findings/2026-11-06-turbo-phase3-results.md) - 25 conditions on 50 recording-held-out clips; modest strength-20 steering, flat controls and clean completion
* [2026-11-06 Phase 3 throughput](findings/2026-11-06-phase3-throughput.md) - same-condition CUDA batching, fresh encoding under every edit, discovery-only parity benchmark and post-job memory cleanup
* [2026-11-06 budget-sweep throughput](findings/2026-11-06-budget-sweep-throughput.md) - device-native replacement validated at 222M-weight endpoint (22.94 GB peak); unchanged-grid sweep restarted with per-budget checkpoints
* [2026-11-06 Phase 2 GPU capture](findings/2026-11-06-phase2-gpu-capture.md) - real 12–13x selection speedup, numerical parity and 31 preserved captures reused
* [2026-11-06 Phase 2 throughput](findings/2026-11-06-phase2-throughput.md) - CUDA backprop followed by huge CPU gradient export/top-k; cached GPU replicas are not active workers
* [2026-11-06 Phase 1 throughput](findings/2026-11-06-phase1-throughput.md) - low utilization from single-sample threaded decoding; batching feasibility and a smaller Phase 1 budget recommendation
* [2026-11-06 turbo Phase 1 validation](findings/2026-11-06-turbo-phase1-validation.md) - real eight-GPU validation, corrected 10,000-clip local corpus and resumed full Phase 1 / Explore fetch
* [2026-11-06 original-text correction](findings/2026-11-06-original-text-correction.md) - prepared transcripts were normalized; restore original references before digit filtering
* [2026-11-06 server-scale preparation](findings/2026-11-06-server-scale-preparation.md) - valid A40/checkpoint checks; initial text inventory deprecated by the correction above
* [2026-11-06 smoke run](findings/2026-11-06-smoke-run.md) - first end-to-end results on whisper-small
* [2026-11-06 Phase 1.5 style patterns](findings/2026-11-06-phase1-5-style-patterns.md) - prefixes, hyphens, units and magnitude words around digits
* [2026-11-06 budget sweep](findings/2026-11-06-budget-sweep-small.md) - how many parameters the signal needs
* [2026-11-06 Phase 2N neuron edits](findings/2026-11-06-phase2n-neuron-edits.md) - where the decision lives, and 64-neuron edits on held-out audio
* [2026-11-06 cross-domain transfer](findings/2026-11-06-cross-domain-transfer.md) - Knesset-fit edits evaluated on explore.ivrit.ai audio
