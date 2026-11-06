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

Status: working. Real API (verified):

* `GET /api/search?q=%5Cd&mode=regex&page=<1-based>&per_page=50&shuffle=true&seed=<n>` returns `{has_more, total_hits, episodes: [...]}`;
  each episode has `uuid, source, title, date, duration_sec, hits: [...]`; **one hit per matched character**, so segments repeat and are
  deduplicated by `(uuid, segment_idx)`. `HEAD` on audio is not allowed (405); `GET /audio/<uuid>.opus` supports Range (206).
* Episodes are 3-60 minutes of Ogg/Opus (~31 kbit/s, ~12 MB per hour). `ffmpeg -ss ... -i <url>` costs 9-12 s per crop (bisection over
  many round-trips), so the adapter groups segments by episode, runs 4 episodes in parallel on keep-alive connections, Range-fetches
  the Ogg headers plus a ~300 KB window around each segment, parses Ogg page granules itself, assembles a mini Ogg and decodes it
  locally. Result: 0.17 s and 230 KB per sample (354 samples/min); the old path was 9 s per sample. `mode="episode"` (whole-file
  download, 3.9 MB/sample, 0.45 s/sample) is the fallback. Validated against the slow path on 5 clips (correlation 1.000).
* Job `corpus_fetch_explore`: `seeds`, `per_seed`, `target`, `per_episode` (6), `pad`, `mode`, `workers`. 1,300 clips, 285 episodes,
  median 3.5 s, all with digits and Hebrew text.

# Offline HF-cache source (`hf_cache`)

Datasets already in `~/.cache/huggingface/hub` (`yoad/knesset-plenums`, `yoad/knesset-committees`, ...) provide `audio.m4a` and
`transcript.aligned.json`. The importer builds windows of merged adjacent segments (6-25 s) around segments containing digits,
crops them with ffmpeg to 16 kHz wav and stores them in `data/corpus/`. Deterministic with `seed`. Used for the first smoke run.

# PocketTTS aligned-manifest source (`pockettts_*`)

Job `corpus_import_pockettts` reads only plenum, committee and crowd_transcribe_v5
aligned train/valid manifests from `/mlspeech/hebrew/pockettts-data`. Digit eligibility
comes from recovered pre-normalization text, NOT the prepared `transcript`.
Knesset originals are recorded in inverse `alt_words.norms[*].replace` spans;
older crowd rows join `clip` to original source-index `text`. See the
[provenance correction](/findings/2026-11-06-original-text-correction.md). It crops
the refined `start` / `duration` and records recording identity for held-out splitting.
See [A40 server preparation](/reference/server-setup.md) for configuration, data
shortfall reporting, setup-only preflight and the explicit phase execution gate.

# Caveats

* Reference digit usage differs by source; the model inherits that inconsistency.
* `\d` also matches identifiers and list numbers; no number-like filter yet.
* Audio and cookies are never committed (`data/` is ignored).
