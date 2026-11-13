"""Conservative integer-value checks for discovery-pair filtering, not ITN.

Reject dates, fractions, decimals, vague quantities, digit-by-digit readings
and ambiguous number grammar instead of guessing. This never rewrites ASR
output and is not a replacement for sampled correctness judgments.
"""
from __future__ import annotations

import re

from .hebrew_numbers import classify_word, normalize_word, words

SMALL = {
    "אפס": 0, "אחד": 1, "אחת": 1, "ראשון": 1, "ראשונה": 1,
    "שניים": 2, "שנים": 2, "שני": 2, "שתיים": 2, "שתים": 2, "שתי": 2,
    "שנייה": 2, "שניה": 2, "שלוש": 3, "שלושה": 3, "שלושת": 3, "שלישי": 3, "שלישית": 3,
    "ארבע": 4, "ארבעה": 4, "ארבעת": 4, "רביעי": 4, "רביעית": 4,
    "חמש": 5, "חמישה": 5, "חמשת": 5, "חמישי": 5, "חמישית": 5,
    "שש": 6, "שישה": 6, "ששת": 6, "שישי": 6, "שישית": 6,
    "שבע": 7, "שבעה": 7, "שבעת": 7, "שביעי": 7, "שביעית": 7,
    "שמונה": 8, "שמונת": 8, "שמיני": 8, "שמינית": 8,
    "תשע": 9, "תשעה": 9, "תשעת": 9, "תשיעי": 9, "תשיעית": 9,
    "עשר": 10, "עשרה": 10, "עשרת": 10, "עשירי": 10, "עשירית": 10,
    "עשרים": 20, "שלושים": 30, "ארבעים": 40, "חמישים": 50,
    "שישים": 60, "שבעים": 70, "שמונים": 80, "תשעים": 90,
    "אחדעשר": 11, "אחתעשרה": 11,
}
ATOMIC = {**SMALL, "מאה": 100, "מאתיים": 200, "מאתים": 200, "אלפיים": 2000}
SCALE = {"אלף": 1000, "אלפים": 1000, "מיליון": 10**6, "מיליונים": 10**6,
         "מיליארד": 10**9, "מיליארדים": 10**9}
KNOWN = set(ATOMIC) | set(SCALE) | {"מאות"}
PREFIXES = "והבכלמש"


def _atom(token):
    # Numeric punctuation must be checked before normalize_word strips it.
    if re.search(r"\d", token):
        match = re.fullmatch(r"(?:[והבכלמש]-?)?(\d+|\d{1,3}(?:,\d{3})+)[.,!?;:]?", token)
        return int(match[1].replace(",", "")) if match else None
    word = normalize_word(token)
    for n in range(4):
        if n and (len(word) <= n + 1 or any(c not in PREFIXES for c in word[:n])):
            continue
        candidate = word[n:]
        if candidate in KNOWN:
            return candidate
    return None


def _under_thousand(tokens):
    if not tokens:
        return None
    if len(tokens) == 1:
        t = tokens[0]
        return t if isinstance(t, int) else ATOMIC.get(t)
    values = [t if isinstance(t, int) else ATOMIC.get(t) for t in tokens]
    hundred, start = 0, 0
    if values[0] in (100, 200):
        hundred, start = values[0], 1
    elif len(tokens) >= 2 and values[0] in range(1, 10) and tokens[1] == "מאות":
        hundred, start = values[0] * 100, 2
    rest = values[start:]
    if not rest:
        return hundred
    if len(rest) == 1 and rest[0] is not None and 0 <= rest[0] < 100:
        return hundred + rest[0]
    if len(rest) == 2 and all(v is not None for v in rest):
        a, b = rest
        if 1 <= a <= 9 and b == 10:
            return hundred + a + b
        if a in range(20, 100, 10) and 1 <= b <= 9:
            return hundred + a + b
    return None


def _parse(tokens):
    total, chunk, last_scale = 0, [], float("inf")
    for token in tokens:
        scale = SCALE.get(token) if isinstance(token, str) else None
        if token == "אלפיים":
            if chunk or last_scale <= 1000:
                return None
            total += 2000
            last_scale = 1000
        elif scale:
            if scale >= last_scale:
                return None
            if chunk:
                coefficient = _under_thousand(chunk)
            else:
                # A bare plural is vague ('thousands'), not a known value.
                coefficient = 1 if token in ("אלף", "מיליון", "מיליארד") else None
            if coefficient is None or not 0 < coefficient < 1000:
                return None
            total += coefficient * scale
            chunk, last_scale = [], scale
        else:
            chunk.append(token)
    tail = _under_thousand(chunk) if chunk else 0
    if tail is None or (last_scale != float("inf") and tail >= last_scale):
        return None
    return total + tail


def value_sequence(text):
    """Ordered recognized integer mentions, or None if any span is unclear.

    False negatives are intentional: e.g. the century idiom 'מאה העשרים'
    and year shorthand 'עשרים עשרים ושתיים' are not interpreted specially.
    """
    values, span = [], []
    for token in words(text) + [""]:
        atom = _atom(token)
        if atom is not None:
            span.append(atom)
        else:
            if classify_word(token) is not None:
                return None  # a detected but unsupported numeric expression
            if span:
                value = _parse(span)
                if value is None:
                    return None
                values.append(value)
                span = []
    return values


def same_values(baseline, cardinal):
    """High-precision discovery gate. Unknown is NOT equal/correct."""
    a, b = value_sequence(baseline), value_sequence(cardinal)
    return bool(a) and a == b
