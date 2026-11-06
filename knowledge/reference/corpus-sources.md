---
type: Data Source
title: Corpus sources
description: Where utterances come from - the explore.ivrit.ai regex search (login required) and the offline HF-cache source.
tags: [corpus, explore-ivrit-ai, data, auth]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:40:00Z }
resource: https://explore.ivrit.ai/search?q=%5Cd&mode=regex&seed=234608111
---

# explore.ivrit.ai (target source)

Regex search `\d` finds transcribed segments containing digit characters; each hit has a time range so audio can be fetched by range.
The site requires **Google login**; unauthenticated requests redirect to `/xhost-auth/login`.

**Workflow (UI -> corpus -> "explore.ivrit.ai")**

1. Log in with a browser, open DevTools -> Network, run the search, *Copy as cURL* for the search request and for one audio request.
2. Paste both; the harness extracts the cookie, headers and URL templates (`{q} {mode} {seed} {page}` substituted) and stores them in
   `data/secrets/explore.json` (mode 0600, gitignored).
3. *Check auth*, optionally *Probe API paths* (crawls the site's same-origin JS for `/api/...` paths), adjust the JSON field mapping
   (`results_path`, `text`, `id`, `start`, `end`, `audio_url_key`), then *Fetch batch* (`seed`, `limit`).
4. Seed strategy: first batch of ~40 for a seed, change seed, repeat; duplicates are skipped by `(source, id@start-end)`.

Status: client implemented, **not yet exercised against the live API** (no cookie available when it was written).

# Offline HF-cache source (`hf_cache`)

Datasets already in `~/.cache/huggingface/hub` (`yoad/knesset-plenums`, `yoad/knesset-committees`, ...) provide `audio.m4a` and
`transcript.aligned.json`. The importer builds windows of merged adjacent segments (6-25 s) around segments containing digits,
crops them with ffmpeg to 16 kHz wav and stores them in `data/corpus/`. Deterministic with `seed`. Used for the first smoke run.

# Caveats

* Reference digit usage differs by source; the model inherits that inconsistency.
* `\d` also matches identifiers and list numbers; no number-like filter yet.
* Audio and cookies are never committed (`data/` is ignored).
