---
type: Plan
title: Phase 1 - existence proof
description: Prove the model can produce both digits and cardinal text at the same decision point by rewinding and banning digit tokens.
tags: [plan, phase1, minimal-pair]
status: stable
generated: { by: claude-agent/claude-sonnet-5-5, at: 2026-11-06T09:35:00Z }
---

# Procedure (job `phase1`, code `src/bwhisper/phase1.py`)

1. Take corpus utterances whose reference text contains digits.
2. Greedy-decode freely (forced prefix `<|startoftranscript|><|he|><|transcribe|><|notimestamps|>`, Whisper's own
   `suppress_tokens` applied).
3. If a digit-containing token is generated, rewind to just before the **first** one.
4. Ban *every* token whose text contains a digit (computed from the tokenizer; `extra_digit_chars` can add e.g. `%`) and continue decoding.
5. Judge the result.

# Verdicts (`src/bwhisper/hebrew_numbers.py`)

| verdict | meaning |
|---|---|
| `cardinal` | every digit span was replaced by Hebrew number words |
| `partial` | some spans became number words, others did not |
| `no_number_words` | digits gone but replaced by non-number text |
| `degenerate` | repetition loop (e.g. `ב-ב-ב-ב`); usually the hyphen prefix `ב-` already committed to digits |
| `still_digits` / `empty` | should not happen with the ban in place |
| `no_digits` | the free decode contained no digits (natural cardinal sample; not usable as a decision point) |

Also recorded: `rest_match` (fraction of non-number words unchanged), `p_digit` at the decision step, top-k tokens with and
without the ban, `hyphen_committed`, the digit token and the first cardinal token. A human *good/bad* verdict in the UI
overrides the automatic one.

# Output = Phase 2 input

Each row with a `cardinal_token` is a decision point: `prefix_ids` (full decoder prefix), `digit_token`, `cardinal_token`.
`decision_points()` filters by effective verdict (`cardinal`, `human_good` by default).

# Success criterion

A large fraction of digit samples are `cardinal` with high `rest_match`. Observed on whisper-small: see [findings](/findings/2026-11-06-smoke-run.md).
