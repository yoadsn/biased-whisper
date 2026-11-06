---
type: Finding
title: Phase 1.5 - digit style leaks into prefixes, units and magnitude words
description: Co-occurrence mining over 146 whisper-small decodes shows hyphenated one-letter prefixes are the main digit-style satellite, prefixes fuse into the first number word in word style, and v1 rewinding was too late for hyphen-prefixed numbers.
tags: [finding, phase1.5, prefixes, style-span, tokenization, hebrew]
status: draft
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T12:30:00Z }
sources:
  - id: run
    resource: data/runs/phase1_5-20261106-122047-7cef
    title: phase1_5 run over the two Phase 1 runs (146 samples, 141 digit spans, 73 valid pairs; local only, not committed)
---

# Tokenization facts (whisper-small BPE)

| text | tokens |
|---|---|
| `ב-2025` | `' ב'`, `'-'`, `'20'`, `'25'` |
| `באלפיים עשרים וחמש` | `' בא'`, ... (prefix fused into the first subword) |
| `80%` | `' 80'`, `'%'` |
| `שמונים אחוז` | `' ש'`, `'מ'`, `'ונים'`, `' אח'`, `'וז'` |
| `ו-9` / `ותשעה` | `' ו'`, `'-'`, `'9'` / `' ו'`, `'ת'`, ... |

So for `ב`, the style decision is the **first token** of the span (`' ב'` vs `' בא'`); for `ו` it is the **second** (`'-'` vs `'ת'`).
Number words are split into generic subword tokens, so "number-word tokens" cannot be listed or banned. Banning is only possible on
the digit side.

# Results[^run]

* **31% of digit spans** are immediately preceded by a hyphen. The most common two-token lead-ins are `' ב' '-'` (13%), `' מ' '-'`,
  `' ל' '-'`, `' ה' '-'` (3-4% each).
* **Satellites** (enrichment z >= 3): `'-'` at offset -1 (z = 10.0), `' ב'` at offset -2 (z = 5.9). `'%'` at +1 has z = 4.0 but
  share 4.96%, just under the 5% cutoff. `' מ'` at +1 (z = 6.7) is the first subword of `מיליון`.
* **Prefix realization** in the model's free decodes: digits take hyphenated prefixes (`ב-` 18, `מ-` 6, `ל-` 6, `ה-` 4); number words
  take fused prefixes (`ב` 9, `ה` 14, `ו` 8, `מ` 4). References follow the same split, with more digits overall (161 bare digits vs 79
  bare number words).
* **Edit patterns** (digits vs digits banned, 96 spans): `<D> => <N>` 67%, `<D>% => <N> אחוז` 3%, `<D> אחוז => <N> אחוז` 2%.
  Hyphen cases split by *where the rewind happened*:
  * hyphen already emitted before the rewind point: `ב-<D> => ב- <N>`, `מה-<D> => מה-<N>` (awkward, hyphen kept);
  * hyphenated number *after* the rewind point: `ל-<D> => ל<N>` (`ל-100` -> `למאה`), `ו-<D> => ו<N>` (`ו-9` -> `ותשעה`).
  This confirms that the model fuses the prefix by itself when the decision is not pre-committed, and that v1 rewound too late.
* **Magnitude words are part of the digit style:** `5 מיליון`, `150 אלף`. Mixed `<D> מיליון` is normal Hebrew digit style and must
  not be scored as incoherent.

# Consequences (implemented)

* Phase 1 `span_mode` (default on) rewinds to the start of the style span (prefix + hyphen), bans digits, digit-style symbols (`%`) and
  a hyphen right after a one-letter prefix, and records the decision at the first token where the two paths diverge.
* `objective=auto` uses the sequence contrast over the divergent region, so `%`/`אחוז`, the prefix and agreement are all part of the
  objective. See [Approach v2](/plan/approach-v2.md).
* Only symbol satellites are banned; word satellites such as `מ` (`מיליון`) are never banned.

[^run]: phase1_5 run over the two Phase 1 runs (146 samples, 141 digit spans, 73 valid pairs)
