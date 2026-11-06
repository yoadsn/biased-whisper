"""Detect Hebrew cardinal/ordinal number words and digit runs in text.

This is a *detector*, not a converter: it answers "does this span contain
number words" so Phase 1 can auto-judge that suppressing digits produced a
spelled-out number rather than garbage.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_PUNCT = re.compile(r"[\"'׳״`.,;:!?()\[\]{}<>…–—־\-/\\]+")
DIGIT_RE = re.compile(r"\d")
DIGIT_RUN_RE = re.compile(r"\d[\d.,:/\-%]*")

CARDINAL_STEMS = {
    # units (m/f, absolute/construct)
    "אחד", "אחת", "אחד", "שניים", "שנים", "שני", "שתיים", "שתים", "שתי",
    "שלוש", "שלושה", "שלושת", "ארבע", "ארבעה", "ארבעת", "חמש", "חמישה", "חמשת",
    "שש", "שישה", "ששת", "שבע", "שבעה", "שבעת", "שמונה", "שמונת", "תשע", "תשעה", "תשעת",
    "עשר", "עשרה", "עשרת", "עשרה",
    # teens spelled as single tokens in some writings
    "אחדעשר", "עשתיים", "עשתי", "אחתעשרה",
    # tens
    "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים",
    # hundreds / thousands / millions
    "מאה", "מאות", "מאתיים", "מאתים", "מאת", "אלף", "אלפים", "אלפיים", "אלפי", "אלפיים",
    "מיליון", "מיליוני", "מיליונים", "מיליארד", "מיליארדי", "מיליארדים", "טריליון",
    "אפס", "חצי", "רבע",
}
ORDINAL_STEMS = {
    "ראשון", "ראשונה", "שני", "שנייה", "שניה", "שלישי", "שלישית", "רביעי", "רביעית",
    "חמישי", "חמישית", "שישי", "שישית", "שביעי", "שביעית", "שמיני", "שמינית",
    "תשיעי", "תשיעית", "עשירי", "עשירית",
}
_PREFIXES = "והבכלמש"


def normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", w)
    w = _NIQQUD.sub("", w)
    w = _PUNCT.sub("", w)
    return w.strip()


def _strip_candidates(w: str):
    yield w
    for n in (1, 2, 3):
        if len(w) > n + 1 and all(c in _PREFIXES for c in w[:n]):
            yield w[n:]


def classify_word(word: str) -> str | None:
    """Return 'cardinal', 'ordinal', 'digit' or None for a single token."""
    if DIGIT_RE.search(word):
        return "digit"
    w = normalize_word(word)
    if not w:
        return None
    for cand in _strip_candidates(w):
        if cand in CARDINAL_STEMS:
            return "cardinal"
        if cand in ORDINAL_STEMS:
            return "ordinal"
    return None


def words(text: str) -> list[str]:
    return [t for t in re.split(r"\s+|(?<=\d)(?=[^\d\s.,:/\-%])|(?<=[^\d\s.,:/\-%])(?=\d)", text.strip()) if t]


def number_words(text: str) -> list[str]:
    return [w for w in words(text) if classify_word(w) in ("cardinal", "ordinal")]


def digit_chars(text: str) -> int:
    return len(DIGIT_RE.findall(text))


def digit_groups(text: str) -> list[str]:
    return DIGIT_RUN_RE.findall(text)


def count_mentions(text: str) -> dict[str, int]:
    """Count number *mentions*: runs of adjacent number words count once;
    runs of digits count once."""
    counts = {"digit": 0, "word": 0}
    prev = None
    for w in words(text):
        c = classify_word(w)
        kind = "digit" if c == "digit" else ("word" if c else None)
        if kind and kind != prev:
            counts[kind] += 1
        prev = kind
    return {"digit_mentions": counts["digit"], "word_mentions": counts["word"]}


def diff_number_spans(orig: str, new: str) -> dict:
    """Word-level alignment of the original (digit) output vs the
    digit-suppressed output. Reports, for every original span containing
    digits, what replaced it and whether the replacement is number words."""
    a, b = words(orig), words(new)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    spans = []
    equal_words = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            equal_words += i2 - i1
            continue
        o, n = a[i1:i2], b[j1:j2]
        if not any(DIGIT_RE.search(w) for w in o):
            continue
        kinds = [classify_word(w) for w in n]
        nw = [w for w, k in zip(n, kinds) if k in ("cardinal", "ordinal")]
        spans.append(
            {
                "orig": " ".join(o),
                "new": " ".join(n),
                "op": op,
                "number_words": nw,
                "is_number_words": bool(nw) and not any(k == "digit" for k in kinds),
            }
        )
    non_digit_orig = sum(1 for w in a if not DIGIT_RE.search(w))
    return {
        "spans": spans,
        "rest_match": (equal_words / non_digit_orig) if non_digit_orig else 1.0,
        "ratio": sm.ratio(),
    }


def is_degenerate(text: str, min_repeat: int = 4) -> bool:
    """True when a token (or short n-gram) repeats back-to-back, e.g. 'ב-ב-ב-ב-ב'."""
    toks = [t for t in re.split(r"[\s\-־]+", text.strip()) if t]
    for n in (1, 2, 3):
        run = 1
        for i in range(n, len(toks), 1):
            if toks[i : i + n] == toks[i - n : i] and len(toks[i : i + n]) == n:
                run += 1
                if run >= min_repeat * 1:
                    return True
            else:
                run = 1
    return False


def auto_verdict(orig: str, new: str) -> tuple[str, dict]:
    """cardinal | partial | no_number_words | still_digits | degenerate | empty"""
    if not new.strip():
        return "empty", {}
    if is_degenerate(new):
        return "degenerate", {}
    d = diff_number_spans(orig, new)
    if DIGIT_RE.search(new):
        return "still_digits", d
    spans = d["spans"]
    if not spans:
        return "no_number_words", d
    ok = sum(1 for s in spans if s["is_number_words"])
    if ok == len(spans):
        return "cardinal", d
    if ok:
        return "partial", d
    return "no_number_words", d


def split_number_prefix(word: str) -> tuple[str, str] | None:
    """Split a Hebrew number word (cardinal or ordinal) into (prefix, stem),
    mirroring _strip_candidates / classify_word. Returns None if not a number word."""
    if DIGIT_RE.search(word):
        return None
    w = normalize_word(word)
    if not w:
        return None
    if w in CARDINAL_STEMS or w in ORDINAL_STEMS:
        return ("", w)
    for n in (1, 2, 3):
        if len(w) > n + 1 and all(c in _PREFIXES for c in w[:n]):
            cand = w[n:]
            if cand in CARDINAL_STEMS or cand in ORDINAL_STEMS:
                return (w[:n], cand)
    return None

