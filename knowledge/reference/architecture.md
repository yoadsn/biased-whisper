---
type: Reference
title: Harness architecture
description: Modules, job system, data layout, API and CLI of the biased-whisper harness.
tags: [reference, architecture, harness]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:40:00Z }
resource: src/bwhisper
---

# Layout

```
src/bwhisper/
  config.py        paths, settings.json, phase default hyperparameters
  model.py         load Whisper, digit-token set, explicit greedy decoder (parity-tested against generate)
  hebrew_numbers.py number-word lexicon (prefix stripping), digit runs, word diff, auto verdicts
  audio.py         ffmpeg decode/crop -> 16 kHz mono
  corpus/          store.py (SQLite), hf_cache.py (offline source), explore.py (explore.ivrit.ai), jobs.py
  jobs.py          job registry/runner; run dir = data/runs/<id>/{job.json,log.txt,*.jsonl,...}
  phase1.py phase2.py phase3.py   the three phases (each registers a job type)
  paramspace.py    stable flat addressing of a parameter subset, sparse edit/undo
  aggregate.py     counts -> common set -> voting -> permutation null -> FDR (pure numpy, unit-tested)
  api.py           FastAPI: settings, corpus, jobs, phase views, knowledge
  cli.py           `bw serve | corpus-import | run <type> <json> | jobs | job-types`
web/               index.html, app.js, style.css (Alpine.js from CDN, no bundler; RTL for Hebrew)
knowledge/         this OKF bundle
tests/             decoder parity, number lexicon, aggregation statistics
data/              gitignored: corpus.db, audio, runs, secrets, settings.json
```

# Jobs

`@register(name, kind, defaults)` adds a job type. `kind='gpu'` jobs run one at a time, `'io'` jobs on a separate pool.
Each run writes `job.json` (state), `log.txt`, and phase outputs; the UI polls `/api/jobs` and reads the same files, so a run started
from the CLI shows up in the UI and vice versa. Job state survives restarts (active jobs are marked `interrupted`).

# Decoder

`model.greedy` is a cached greedy decoder with explicit control of the forced prefix, ban sets and rewinds. Tests assert it
matches `WhisperForConditionalGeneration.generate` token for token, and that rewinding reproduces the full decode.
Whisper's `suppress_tokens` are applied; digits are never in that list.

# API (selection)

`GET/POST /api/settings`, `/api/corpus*`, `/api/explore/*`, `GET/POST /api/jobs`, `/api/jobs/{id}/{log,rows,file/<name>.json}`,
`POST /api/phase1/{id}/verdict`, `/api/phase2/{id}/candidates`, `/api/knowledge*`.

# Environment

Python 3.13 via `uv`; torch picks CUDA > MPS > CPU (`device` setting). `ffmpeg` must be on PATH. Hugging Face models load from the local cache first.
