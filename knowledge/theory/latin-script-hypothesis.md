---
type: Hypothesis
title: Latin vs Hebrew-transliteration output is steerable like digits vs cardinals
description: Second axis for the same harness - an English word inside a Hebrew sentence can be written in Latin letters or transliterated into Hebrew letters, and a small set of weights may choose between them.
tags: [whisper, hebrew, ivrit-ai, code-switching, transliteration, latin]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-13T15:30:00Z }
---

# Statement

A Hebrew Whisper checkpoint writes an English word spoken inside a Hebrew sentence either in **Latin letters**
(`ב-Facebook`, `eBay`) or **transliterated into Hebrew** (`בפייסבוק`, `אי-ביי`). As with
[digits vs cardinals](/theory/hypothesis.md), the choice may be localised enough to bias with a small adapter.

# Mapping onto the digits harness

| Digits axis | Latin axis |
|---|---|
| digit tokens (any token containing a digit) | Latin tokens (any token containing `[A-Za-z]`) |
| ban digits, re-decode, get cardinals | ban Latin letters, re-decode, get Hebrew spelling |
| verdict: number words + integer value check | verdict: consonant-skeleton / letter-name similarity (**heuristic**, no canonical spelling) |
| prefix + hyphen style span (`ב-2024`) | same span logic (`ב-Facebook` -> `בפייסבוק`) |

# Differences that matter

* There is no exact verdict. [`latin_script.span_verdict`](/reference/architecture.md) screens pairs; every pair is
  stored so they can be reviewed by eye. A transliteration has several valid spellings.
* Scope is single borrowed words, brand names and spelled-out acronyms. Whole English phrases have no
  transliteration; the banned path then translates or hallucinates.
* Script choice may be entangled with language-ID / English decoding, so it may be less local than digit style.
  Controls must include WER on non-English words and a check that nothing is translated.
* Data is thin: Latin text is rare in the ivrit.ai references (32 of 1,646 local clips). A Hebrew-next-to-Latin regex query on explore.ivrit.ai yields a few hundred short clips.

# Falsifiers

* Phase 1: banning Latin letters mostly yields translations, garbage or dropped words for ordinary English words.
* Later phases: same as digits - edits no better than shuffled/random controls, or only work by degrading Hebrew.

See the [Phase 1 POC finding](/findings/2026-11-13-latin-phase1-poc.md).
