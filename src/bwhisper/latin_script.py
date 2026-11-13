"""Latin-script vs Hebrew-transliteration detection for the Latin axis.

The digits axis has an exact detector (digit characters, number words, integer values). The Latin axis
has no canonical transliteration spelling, so this module only offers a *heuristic* check: both the
Latin word and its candidate Hebrew spelling are reduced to a coarse consonant skeleton and compared.
It is a screening filter for Phase 1 verdicts, never ground truth; review the pairs by eye.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

LATIN_RE = re.compile(r"[A-Za-z]")
LATIN_RUN_RE = re.compile(r"[A-Za-z]{2,}")
# What a corpus sample must contain to be useful for each experiment axis.
CONTENT_PATTERNS = {"digits": re.compile(r"\d"), "latin": LATIN_RUN_RE}


def content_pattern(content: str) -> "re.Pattern[str]":
    try:
        return CONTENT_PATTERNS[content]
    except KeyError:
        raise ValueError(f"content must be one of {sorted(CONTENT_PATTERNS)}, got {content!r}") from None


def latin_run_count(text: str) -> int:
    return len(LATIN_RUN_RE.findall(text))
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'.\-]*")
HEBREW_RE = re.compile(r"[א-ת]")
HEBREW_PREFIXES = "והבכלמש"

# Hebrew letters -> coarse phoneme class. Vowel letters (alef, he, vav, yod, ayin) are dropped.
_HEB = {
    "ב": "b", "ג": "g", "ד": "d", "ז": "s", "ח": "h", "ט": "t", "כ": "k", "ך": "k", "ל": "l",
    "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s", "פ": "p", "ף": "p", "צ": "s", "ץ": "s",
    "ק": "k", "ר": "r", "ש": "s", "ת": "t",
}
# Latin digraphs first, then single letters; vowels, h and apostrophes drop out.
_LAT_DIGRAPHS = (("sh", "s"), ("ch", "s"), ("ts", "s"), ("ph", "p"), ("th", "t"), ("ck", "k"), ("qu", "k"))
_LAT = {
    "b": "b", "v": "b", "w": "b", "c": "k", "k": "k", "q": "k", "x": "ks", "d": "d", "f": "p", "p": "p",
    "g": "g", "j": "g", "l": "l", "m": "m", "n": "n", "r": "r", "s": "s", "z": "s", "t": "t",
}


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def latin_words(text: str) -> list[str]:
    return LATIN_WORD_RE.findall(text)


def latin_kind(word: str) -> str:
    """word: ordinary lower/Capitalized word; acronym: ALLCAPS or short mixed; symbol: has digits."""
    letters = re.sub(r"[^A-Za-z]", "", word)
    if any(c.isdigit() for c in word):
        return "symbol"
    if len(letters) < 3 or (letters.isupper() and len(letters) <= 6):
        return "acronym"
    return "word"


def latin_skeleton(text: str) -> str:
    s = re.sub(r"[^a-z]", "", text.lower())
    for a, b in _LAT_DIGRAPHS:
        s = s.replace(a, b)
    out = "".join(_LAT.get(c, "") for c in s)
    return re.sub(r"(.)\1+", r"\1", out)


def hebrew_skeleton(text: str) -> str:
    out = "".join(_HEB.get(c, "") for c in text)
    return re.sub(r"(.)\1+", r"\1", out)


def translit_score(latin: str, hebrew: str) -> float:
    """Best skeleton similarity in [0,1] between a Latin phrase and a Hebrew phrase, allowing the Hebrew
    side to start with up to two prefix letters (ב ה ל ...) that belong to the Hebrew grammar, not the word."""
    ls = latin_skeleton(latin)
    words = [w for w in re.split(r"\s+", hebrew) if HEBREW_RE.search(w)]
    if not ls or not words:
        return 0.0
    best = 0.0
    letters = re.sub(r"[^א-ת]", "", words[0])
    rest = "".join(re.sub(r"[^א-ת]", "", w) for w in words[1:])
    for strip in range(0, 3):
        if strip and not all(c in HEBREW_PREFIXES for c in letters[:strip]):
            break
        hs = hebrew_skeleton(letters[strip:] + rest)
        if hs:
            best = max(best, SequenceMatcher(a=ls, b=hs, autojunk=False).ratio())
    return best


# How Hebrew text spells the English letter names (acronyms and variables are read letter by letter).
_LETTER_NAMES = {
    "a": ("אי", "איי", "אה"), "b": ("בי", "ביי"), "c": ("סי", "סיי"), "d": ("די", "דיי"),
    "e": ("אי", "איי", "איה"), "f": ("אף", "אפ", "אפף"), "g": ("גי", "גיי", "ג'י", "ג'יי"),
    "h": ("אייץ", "אייטש", "איץ", "אייצ'"), "i": ("איי", "אי"), "j": ("ג'יי", "גיי", "ג'י", "גי"),
    "k": ("קיי", "קי", "קיי"), "l": ("אל", "אלל"), "m": ("אם", "אמ"), "n": ("אן", "אנ"), "o": ("או", "אוו"),
    "p": ("פי", "פיי"), "q": ("קיו", "קיוו"), "r": ("אר", "אאר"), "s": ("אס", "אסס"), "t": ("טי", "טיי"),
    "u": ("יו", "יוו"), "v": ("וי", "ווי", "ויי"), "w": ("דאבליו", "דבליו", "דאבל"), "x": ("אקס", "איקס", "אכס"),
    "y": ("וואי", "ווי", "וואיי"), "z": ("זד", "זי", "זט"),
}
_NAME_LETTERS: dict[str, set[str]] = {}
for _l, _names in _LETTER_NAMES.items():
    for _n in _names:
        _NAME_LETTERS.setdefault(re.sub(r"[^א-ת]", "", _n), set()).add(_l)


def letter_name_score(latin: str, hebrew: str) -> float:
    """Fraction of the Latin letters read out as Hebrew letter names, in order (LCS over letter tokens)."""
    letters = re.sub(r"[^a-z]", "", latin.lower())
    toks = []
    for w in re.split(r"[\s\-.,]+", hebrew):
        w = re.sub(r"[^א-ת]", "", w)
        if w in _NAME_LETTERS:
            toks.append(_NAME_LETTERS[w])
        elif len(w) > 2 and w[0] in HEBREW_PREFIXES and w[1:] in _NAME_LETTERS:
            toks.append(_NAME_LETTERS[w[1:]])
    if not letters or not toks:
        return 0.0
    dp = [[0] * (len(toks) + 1) for _ in range(len(letters) + 1)]
    for i, c in enumerate(letters):
        for j, t in enumerate(toks):
            dp[i + 1][j + 1] = dp[i][j] + 1 if c in t else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1] / max(len(letters), len(toks))


def span_verdict(orig_span: str, new_span: str, threshold: float = 0.6) -> tuple[str, float]:
    """Classify what replaced a Latin span once Latin letters were banned. The consonant-skeleton path
    needs a Latin span of at least three letters (shorter ones match by accident); the letter-name path covers
    acronyms and variables."""
    if has_latin(new_span):
        return "latin_leak", 0.0
    if not HEBREW_RE.search(new_span):
        return "no_hebrew", 0.0
    latin = " ".join(latin_words(orig_span))
    score = letter_name_score(latin, new_span)
    if len(re.sub(r"[^A-Za-z]", "", latin)) >= 3:
        score = max(score, translit_score(latin, new_span))
    return ("translit" if score >= threshold else "other_hebrew"), score
