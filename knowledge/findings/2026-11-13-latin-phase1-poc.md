---
type: Finding
title: Latin-axis Phase 1 POC (turbo, local Mac, 332 clips)
description: Banning Latin-letter tokens makes ivrit-ai turbo write short English words (90%) and most acronyms in Hebrew letters while mostly preserving the sentence; English phrases are not transliterated. Heuristic verdicts, one checkpoint.
tags: [finding, phase1, latin, transliteration, turbo, poc]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-13T16:00:00Z }
sources:
  - id: hypothesis
    resource: /theory/latin-script-hypothesis.md
    title: Latin vs transliteration hypothesis
---

# Setup

Job `phase1_latin` (`src/bwhisper/phase1_latin.py`, helpers in `latin_script.py`, defaults `PHASE1_LATIN_DEFAULTS`).
Same paired decoding as the digits Phase 1: free greedy decode, rewind to the start of the first Latin span
(prefix + hyphen via `StyleTokens`), ban every Latin-letter token (plus maqaf bytes), continue, compare.
`ivrit-ai/whisper-large-v3-turbo` on MPS (Apple laptop), greedy, 96 new tokens.

Data: the first 32 local clips with a Latin run, plus **300 clips fetched from explore.ivrit.ai** with the regex
`[א-ת]\S* \S*[A-Za-z]{3,}\S* [א-ת]|[א-ת]-[A-Za-z]{3,}` (Hebrew word next to a Latin word; 58 s). Of the 300:
160 first-span ordinary words, 138 acronyms, 2 variable-like symbols. Run `phase1_latin-20261113-152501-5077`,
332 clips, ~8 min wall. The search index is small (about 60-80 hits per pattern) and clips are 1.5-3 s snippets.

# Result

Baseline output contains Latin in 219 of 332 clips (66%); the other 113 already transliterate. Of the 219:

| First Latin span | n | translit | other Hebrew | no Hebrew |
|---|---|---|---|---|
| ordinary word, <=3 words | 73 | **66 (90%)** | 5 | 2 |
| acronym (ALLCAPS) | 127 | 92 (72%) | 34 | 1 |
| variable (`v1`) | 7 | 6 | 1 | 0 |
| multi-word English phrase | 12 | 3 | 7 | 2 |

* Clean examples: `ה-understatement`->`האנדרסטייטמנט`, `ב-memories`->`בממוריז`, `minutes`->`מינטס`,
  `diversity`->`דייברסיטי`, `Channel Go-To-Market Agency`->`צ'אנל גו טו מרקט אייג'נסי`. Prefix+hyphen fuses (`ב-guide`->`בגייד`).
* Failures: English phrases (translated or garbled), short acronyms misheard, one repetition loop (`html`),
  a URL read out letter by letter. Some "other Hebrew" words are real transliterations the heuristic misses (`once`->`וואנס`).
* Preservation: for converted words the tokens after the changed span are identical to the free decode in 48 of 68
  (acronyms 81 of 93); the rest differ, often because later Latin words are banned too. One of 219 hit the token limit; one looped.
* Mean Latin probability at the first Latin token in the free decode: 0.90, so the model is confident in Latin there.

# Reading

The model *knows* a Hebrew spelling at these decision points, the minimal-pair premise holds, and the span/prefix machinery
transfers unchanged. The Latin choice is a high-probability mode of the free decode, so steering it is a real shift
(compare digits at ~0.6). This does not yet show localisation or steerability by weights.

# Caveats

* Verdicts are a heuristic (consonant skeleton and letter names), reviewed by eye on ~50 pairs, not scored against ground truth.
* One checkpoint, greedy decoding, no recording-disjoint split (needed before Phase 2N/3); explore clips come from few episodes' search hits.
* Only the first Latin span per clip is analysed; later Latin words are banned on the same path.
* The opposite direction (transliterated baseline -> Latin) needs a forced-Latin contrast; not built.

# Next

Phase 2N component attribution on the 150+ translit pairs with a frozen recording-disjoint evaluation cohort, then Phase 3 controls
(random-neuron control, WER on non-English words, translation check).
