---
okf_version: "0.2"
---

# Start here

* [Hypothesis](theory/hypothesis.md) - the theory being tested, in the owner's words, with definitions
* [Review of the approach](theory/review-of-approach.md) - critique, fixes made, open risks and alternatives

# Plan

* [Roadmap](plan/roadmap.md) - phases, status and next steps
* [Approach v2](plan/approach-v2.md) - style spans, sequence contrast, component attribution, translation to weight edits
* [Phase 1 - existence proof](plan/phase-1-existence-proof.md) - does the model know both forms?
* [Phase 2 - upstream discovery](plan/phase-2-upstream-discovery.md) - find the knobs
* [Phase 3 - control experiments](plan/phase-3-control-experiments.md) - do the knobs steer output?

# Reference

* [Hyperparameters](reference/hyperparameters.md) - every knob, default, and what it means
* [Harness architecture](reference/architecture.md) - modules, jobs, data layout, CLI/API
* [Corpus sources](reference/corpus-sources.md) - explore.ivrit.ai and the offline HF-cache source

# Findings

* [2026-11-06 smoke run](findings/2026-11-06-smoke-run.md) - first end-to-end results on whisper-small
* [2026-11-06 Phase 1.5 style patterns](findings/2026-11-06-phase1-5-style-patterns.md) - prefixes, hyphens, units and magnitude words around digits
* [2026-11-06 budget sweep](findings/2026-11-06-budget-sweep-small.md) - how many parameters the signal needs
* [2026-11-06 Phase 2N neuron edits](findings/2026-11-06-phase2n-neuron-edits.md) - where the decision lives, and 64-neuron edits on held-out audio
* [2026-11-06 cross-domain transfer](findings/2026-11-06-cross-domain-transfer.md) - Knesset-fit edits evaluated on explore.ivrit.ai audio
