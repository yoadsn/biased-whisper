"""Phase 1.5 - statistical discovery of Hebrew number-style satellites and edit patterns.

Hebrew number style leaks beyond digit characters:
  ' ב-2025' -> [' ב', '-', '20', '25']   vs   ' באלפיים עשרים וחמש'
  ' 80%'    -> [' 80', '%']              vs   ' שמונים אחוז'
  ' ו-9'    -> [' ו', '-', '9']          vs   ' ותשעה'

This IO-only job analyses completed Phase 1 runs (and reference texts) at both
the Whisper BPE token level and the word level, producing `summary.json` and
`satellites.json`.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

from . import model as M
from .config import RUNS_DIR, load_settings
from .hebrew_numbers import (
    DIGIT_RE,
    classify_word,
    normalize_word,
    split_number_prefix,
    words,
)
from .jobs import Ctx, read_jsonl, register
from .phase1 import load_verdicts

PHASE15_DEFAULTS: dict[str, Any] = {
    "phase1_runs": [],
    "model": None,
    "window": 3,
    "min_count": 3,
    "z_min": 3.0,
    "min_share": 0.05,
    "include_reference": True,
}

SEP_TOKEN_TEXTS = {",", ".", ":", "/"}

_HYPHEN_DIGIT_RE = re.compile(r"^(?P<prefix>[והבכלמש]{1,3})[-־](?=\d)")
_GLUED_DIGIT_RE = re.compile(r"^(?P<prefix>[והבכלמש]{1,3})(?=\d)")
_DIGIT_PART_RE = re.compile(r"\d(?:[\d.,:/]*\d)?")
_OUTER_PUNCT_RE = re.compile(r"^[\"'׳״`.,;:!?()\[\]{}<>…–—]+|[\"'׳״`.,;:!?()\[\]{}<>…–—]+$")
_GLUED_SYMBOL_RE = re.compile(r"([%₪$€])")

UNIT_WORDS = {
    "%", "₪", "$", "€",
    "אחוז", "אחוזים", "אחוזי",
    "שקל", "שקלים", "שח",
    "דולר", "דולרים", "יורו", "אירו",
    "קמ", "מטר", "מטרים", "קילומטר", "קילומטרים",
    "קילו", "קילוגרם", "גרם", "טון", "ליטר", "מעלות",
}


def load_tokenizer(name: str):
    """Load only the WhisperTokenizer without loading neural weights."""
    from transformers import WhisperTokenizer

    try:
        return WhisperTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return WhisperTokenizer.from_pretrained(name)


def separator_token_ids(tokenizer, vocab: int | None = None) -> set[int]:
    """Token IDs whose decoded text is exactly one of ',', '.', ':', '/'."""
    n = vocab if vocab is not None else len(tokenizer)
    out: set[int] = set()
    for i in range(n):
        s = tokenizer.decode([i], skip_special_tokens=False)
        if s in SEP_TOKEN_TEXTS:
            out.add(i)
    return out


def find_digit_spans(
    tokens: list[int],
    digit_ids: set[int],
    sep_ids: set[int] | None = None,
) -> list[tuple[int, int]]:
    """Find maximal runs [start, end) of digit tokens in `tokens`, absorbing
    single separator tokens (',', '.', ':', '/') when they sit immediately
    between two digit tokens."""
    sep = sep_ids or set()
    spans: list[tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i] not in digit_ids:
            i += 1
            continue
        start = i
        end = i + 1
        while end < n:
            if tokens[end] in digit_ids:
                end += 1
            elif tokens[end] in sep and (end + 1) < n and tokens[end + 1] in digit_ids:
                end += 2
            else:
                break
        spans.append((start, end))
        i = end
    return spans


def fightin_words_z(
    fg_counts: dict[int, int] | Counter,
    bg_counts: dict[int, int] | Counter,
    alpha0: float = 1000.0,
) -> dict[int, tuple[float, float]]:
    """Monroe, Colaresi & Quinn (2008) 'Fightin' Words' log-odds ratio with an
    informative Dirichlet prior (background scaled to `alpha0`).

    Returns {token_id: (log_odds_ratio, z_score)} for all tokens in `fg_counts`.
    """
    n_fg = float(sum(fg_counts.values()))
    n_bg = float(sum(bg_counts.values()))
    if n_fg <= 0 or n_bg <= 0:
        return {k: (0.0, 0.0) for k in fg_counts}

    all_keys = set(bg_counts) | set(fg_counts)
    eps = 1e-6
    denom_prior = n_bg + eps * len(all_keys)

    out: dict[int, tuple[float, float]] = {}
    for w, y_fg in fg_counts.items():
        if y_fg <= 0:
            continue
        y_bg = float(bg_counts.get(w, 0))
        alpha_w = alpha0 * ((y_bg + eps) / denom_prior)
        # Clamp alpha_w slightly below alpha0 to avoid division by zero in degenerate 1-token vocab
        alpha_w = min(alpha_w, alpha0 - eps)

        num_fg = float(y_fg) + alpha_w
        den_fg = max(eps, n_fg + alpha0 - num_fg)
        num_bg = y_bg + alpha_w
        den_bg = max(eps, n_bg + alpha0 - num_bg)

        delta = math.log(num_fg / den_fg) - math.log(num_bg / den_bg)
        var = (1.0 / num_fg) + (1.0 / den_fg) + (1.0 / num_bg) + (1.0 / den_bg)
        z = delta / math.sqrt(var)
        out[w] = (delta, z)
    return out


def analyse_tokens(
    token_rows: list[list[int]],
    digit_ids: set[int],
    sep_ids: set[int],
    tok_decode: dict[int, str],
    window: int = 3,
    min_count: int = 3,
    z_min: float = 3.0,
    min_share: float = 0.05,
    top_k: int = 25,
) -> dict[str, Any]:
    """Compute token_offsets, pre_digit_chains, hyphen_before_share, and satellites."""
    bg_counts: Counter[int] = Counter()
    offset_counts: dict[int, Counter[int]] = {
        r: Counter() for r in list(range(-window, 0)) + list(range(1, window + 1))
    }
    chains: Counter[tuple[str, str]] = Counter()
    hyphen_before = 0
    n_spans = 0

    for seq in token_rows:
        bg_counts.update(seq)
        spans = find_digit_spans(seq, digit_ids, sep_ids)
        for start, end in spans:
            n_spans += 1
            for r in range(-window, 0):
                pos = start + r
                if 0 <= pos < len(seq):
                    offset_counts[r][seq[pos]] += 1
            for r in range(1, window + 1):
                pos = end - 1 + r
                if 0 <= pos < len(seq):
                    offset_counts[r][seq[pos]] += 1

            if start >= 1:
                prev_txt = tok_decode.get(seq[start - 1], "")
                if prev_txt in ("-", "־") or prev_txt.endswith(("-", "־")) or prev_txt.rstrip().endswith(("-", "־")):
                    hyphen_before += 1
                t1 = prev_txt
                t2 = tok_decode.get(seq[start - 2], "") if start >= 2 else "<BOS>"
                chains[(t2, t1)] += 1
            else:
                chains[("<BOS>", "<BOS>")] += 1

    token_offsets: dict[str, list[dict[str, Any]]] = {}
    all_offset_rows: dict[int, list[dict[str, Any]]] = {}

    for r in sorted(offset_counts):
        fw = fightin_words_z(offset_counts[r], bg_counts, alpha0=1000.0)
        items = []
        for tid, cnt in offset_counts[r].items():
            if cnt < min_count:
                continue
            _, z = fw.get(tid, (0.0, 0.0))
            items.append(
                {
                    "id": int(tid),
                    "tok": tok_decode.get(tid, ""),
                    "offset": int(r),
                    "count": int(cnt),
                    "share": round(cnt / n_spans, 4) if n_spans else 0.0,
                    "z": round(z, 3),
                }
            )
        items.sort(key=lambda x: (-x["z"], -x["count"], x["id"]))
        all_offset_rows[r] = items
        key = f"{r:+d}"
        token_offsets[key] = [
            {k: v for k, v in it.items() if k != "offset"} for it in items[:top_k]
        ]

    pre_digit_chains = [
        {
            "chain": [c[0], c[1]],
            "count": int(cnt),
            "share": round(cnt / n_spans, 4) if n_spans else 0.0,
        }
        for c, cnt in chains.most_common(30)
    ]

    before_sats = []
    for r in (-1, -2):
        for it in all_offset_rows.get(r, []):
            if it["z"] >= z_min and it["count"] >= min_count and it["share"] >= min_share:
                before_sats.append(it)
    before_sats.sort(key=lambda x: (-x["z"], -x["count"]))

    after_sats = []
    for it in all_offset_rows.get(1, []):
        if it["z"] >= z_min and it["count"] >= min_count and it["share"] >= min_share:
            after_sats.append(it)
    after_sats.sort(key=lambda x: (-x["z"], -x["count"]))

    return {
        "n_spans": n_spans,
        "token_offsets": token_offsets,
        "pre_digit_chains": pre_digit_chains,
        "hyphen_before_share": round(hyphen_before / n_spans, 4) if n_spans else 0.0,
        "satellites": {
            "before": before_sats,
            "after": after_sats,
            "rule": {
                "z_min": z_min,
                "min_count": min_count,
                "min_share": min_share,
                "before_offsets": [-1, -2],
                "after_offsets": [1],
                "spans": n_spans,
            },
        },
    }


def classify_text_prefix_realizations(text: str) -> list[tuple[str, str]]:
    """Return a list of (prefix, form) for every number mention in `text`.
    Forms: 'hyphen_digit', 'glued_digit', 'bare_digit', 'fused_word', 'bare_word'.
    """
    out: list[tuple[str, str]] = []
    raw_tokens = text.strip().split()
    prev_digit = False
    for raw in raw_tokens:
        clean = _OUTER_PUNCT_RE.sub("", raw)
        if DIGIT_RE.search(clean):
            m_hyp = _HYPHEN_DIGIT_RE.match(clean)
            if m_hyp:
                out.append((m_hyp.group("prefix"), "hyphen_digit"))
            else:
                m_glu = _GLUED_DIGIT_RE.match(clean)
                if m_glu:
                    out.append((m_glu.group("prefix"), "glued_digit"))
                elif not prev_digit:
                    out.append(("", "bare_digit"))
            prev_digit = True
        else:
            prev_digit = False

    # Word-level number-word runs (first word of each contiguous cardinal/ordinal run)
    prev_word_num = False
    for w in words(text):
        c = classify_word(w)
        if c in ("cardinal", "ordinal"):
            if not prev_word_num:
                sp = split_number_prefix(w)
                prefix = sp[0] if sp else ""
                form = "fused_word" if prefix else "bare_word"
                out.append((prefix, form))
            prev_word_num = True
        else:
            prev_word_num = False
    return out


def word_prefix_realization(sources: dict[str, list[str]]) -> dict[str, dict[str, dict[str, int]]]:
    """Compute {source: {prefix: {form: count}}} across text sources."""
    forms_order = ("hyphen_digit", "glued_digit", "bare_digit", "fused_word", "bare_word")
    result: dict[str, dict[str, dict[str, int]]] = {}
    for src_name, texts in sources.items():
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for t in texts:
            for prefix, form in classify_text_prefix_realizations(t):
                table[prefix][form] += 1
        # Sort prefixes by total descending, keeping '' first if present
        sorted_prefixes = sorted(
            table.keys(),
            key=lambda p: (0 if p == "" else 1, -sum(table[p].values()), p),
        )
        result[src_name] = {
            p: {f: int(table[p].get(f, 0)) for f in forms_order if table[p].get(f, 0) > 0}
            for p in sorted_prefixes
        }
    return result


def extract_units_after(text: str) -> list[tuple[str, str]]:
    """Extract (style, follower) pairs where style in ('digit', 'word') and
    follower is either a glued symbol ('%', '₪', '$', '€') or the normalized next word."""
    out: list[tuple[str, str]] = []
    ws = words(text)
    n = len(ws)
    i = 0
    while i < n:
        c = classify_word(ws[i])
        if c == "digit":
            j = i + 1
            while j < n and classify_word(ws[j]) == "digit":
                j += 1
            span_str = " ".join(ws[i:j])
            m_sym = _GLUED_SYMBOL_RE.search(span_str)
            if m_sym:
                out.append(("digit", m_sym.group(1)))
            elif j < n:
                nxt = ws[j]
                m_nxt_sym = _GLUED_SYMBOL_RE.search(nxt)
                norm = normalize_word(nxt)
                if m_nxt_sym and not norm:
                    out.append(("digit", m_nxt_sym.group(1)))
                elif norm:
                    out.append(("digit", norm))
            i = j
        elif c in ("cardinal", "ordinal"):
            j = i + 1
            while j < n and classify_word(ws[j]) in ("cardinal", "ordinal"):
                j += 1
            if j < n:
                nxt = ws[j]
                m_nxt_sym = _GLUED_SYMBOL_RE.search(nxt)
                norm = normalize_word(nxt)
                if m_nxt_sym and not norm:
                    out.append(("word", m_nxt_sym.group(1)))
                elif norm:
                    out.append(("word", norm))
            i = j
        else:
            i += 1
    return out


def compute_unit_after(sources: dict[str, list[str]], top_k: int = 30) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Distribution of what follows digit mentions and number-word runs per source."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for src_name, texts in sources.items():
        counts: dict[str, Counter[str]] = {"digit": Counter(), "word": Counter()}
        totals: Counter[str] = Counter()
        for t in texts:
            for style, unit in extract_units_after(t):
                counts[style][unit] += 1
                totals[style] += 1
        result[src_name] = {}
        for style in ("digit", "word"):
            tot = totals[style]
            result[src_name][style] = [
                {
                    "unit": u,
                    "count": int(cnt),
                    "share": round(cnt / tot, 4) if tot else 0.0,
                }
                for u, cnt in counts[style].most_common(top_k)
            ]
    return result


def _continues_number_word_run(w: str) -> bool:
    """True if `w` can continue an existing number-word run (bare number word or 'ו'-prefixed)."""
    if classify_word(w) not in ("cardinal", "ordinal"):
        return False
    sp = split_number_prefix(w)
    return sp is not None and sp[0] in ("", "ו")
def _is_unit_word(w: str) -> bool:
    clean = _OUTER_PUNCT_RE.sub("", w)
    if clean in UNIT_WORDS or any(s in clean for s in ("%₪$€")):
        return True
    norm = normalize_word(clean)
    return norm in UNIT_WORDS


def abstract_span(span_text: str | list[str]) -> str:
    """Abstract a phrase into a number-style pattern:
    - Each digit run becomes `<D>` (preserving prefix letters, hyphens, and `%`/`₪`/`$`).
    - Each maximal contiguous run of number words (cardinal/ordinal) becomes `<prefix><N>`,
      where `<prefix>` is the prefix of the first word in the run.
    - Other words and prefixes are kept verbatim.
    """
    if isinstance(span_text, list):
        raw_words = [w for s in span_text for w in s.split() if w]
    else:
        raw_words = [w for w in span_text.strip().split() if w]

    out_tokens: list[str] = []
    i = 0
    n = len(raw_words)
    while i < n:
        w = _OUTER_PUNCT_RE.sub("", raw_words[i])
        if not w:
            i += 1
            continue

        if DIGIT_RE.search(w):
            ab = _DIGIT_PART_RE.sub("<D>", w)
            # Collapse multiple <D> within the same token (e.g. 4.11.2024 or 15:45)
            ab = re.sub(r"<D>(?:[.,:/]<D>)+", "<D>", ab)
            if out_tokens and out_tokens[-1] == "<D>" and ab == "<D>":
                pass
            else:
                out_tokens.append(ab)
            i += 1
            continue

        # Check if this word (or after a hyphen prefix like ב-אלפיים) is a number word
        hyph_prefix = ""
        check_w = w
        if "-" in w or "־" in w:
            parts = re.split(r"([-־])", w, maxsplit=1)
            if len(parts) == 3 and parts[0] and all(c in "והבכלמש" for c in parts[0]):
                if classify_word(parts[2]) in ("cardinal", "ordinal"):
                    hyph_prefix = parts[0] + parts[1]
                    check_w = parts[2]

        c = classify_word(check_w)
        if c in ("cardinal", "ordinal"):
            if hyph_prefix:
                prefix = hyph_prefix
            else:
                sp = split_number_prefix(check_w)
                prefix = sp[0] if sp else ""
            # Consume maximal contiguous run of number words (continuations only have '' or 'ו' prefix)
            j = i + 1
            while j < n:
                w_next = _OUTER_PUNCT_RE.sub("", raw_words[j])
                if not w_next:
                    j += 1
                    continue
                if _continues_number_word_run(w_next):
                    j += 1
                else:
                    break
            out_tokens.append(f"{prefix}<N>")
            i = j
        else:
            out_tokens.append(w)
            i += 1

    return " ".join(out_tokens)


def abstract_edit_pair(orig: str | list[str], new: str | list[str]) -> tuple[str, str]:
    """Abstract an aligned (orig, new) span pair into (orig_pattern, new_pattern)."""
    return abstract_span(orig), abstract_span(new)


def extract_edit_spans(orig_text: str, new_text: str) -> list[tuple[str, str, str, str]]:
    """Align `orig_text` and `new_text` at word level using SequenceMatcher on
    `hebrew_numbers.words()`, extend changed digit-containing regions by 1 word on
    each side if that neighbouring word is a unit/percent word (changed neighbours
    are already inside the non-equal opcode), and return
    [(orig_span, new_span, orig_pattern, new_pattern), ...]."""
    a = words(orig_text)
    b = words(new_text)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    opcodes = sm.get_opcodes()
    results: list[tuple[str, str, str, str]] = []

    for idx, (op, i1, i2, j1, j2) in enumerate(opcodes):
        if op == "equal":
            continue
        if not any(DIGIT_RE.search(w) for w in a[i1:i2]):
            continue

        ext_i1, ext_i2 = i1, i2
        ext_j1, ext_j2 = j1, j2

        # Trim boundary words that only differed by attached punctuation (e.g. 'היום' vs 'היום,')
        while (
            ext_i2 - ext_i1 > 1
            and ext_j2 - ext_j1 > 1
            and not DIGIT_RE.search(a[ext_i1])
            and _OUTER_PUNCT_RE.sub("", a[ext_i1]) == _OUTER_PUNCT_RE.sub("", b[ext_j1])
            and not _is_unit_word(a[ext_i1])
        ):
            ext_i1 += 1
            ext_j1 += 1
        while (
            ext_i2 - ext_i1 > 1
            and ext_j2 - ext_j1 > 1
            and not DIGIT_RE.search(a[ext_i2 - 1])
            and _OUTER_PUNCT_RE.sub("", a[ext_i2 - 1]) == _OUTER_PUNCT_RE.sub("", b[ext_j2 - 1])
            and not _is_unit_word(a[ext_i2 - 1])
        ):
            ext_i2 -= 1
            ext_j2 -= 1

        # Extend left by 1 word if the preceding equal word is a unit/percent word
        if idx > 0 and opcodes[idx - 1][0] == "equal":
            _, pi1, pi2, pj1, pj2 = opcodes[idx - 1]
            if pi2 > pi1 and pj2 > pj1 and _is_unit_word(a[pi2 - 1]):
                ext_i1 -= 1
                ext_j1 -= 1

        # Extend right by 1 word if the following equal word is a unit/percent word
        if idx + 1 < len(opcodes) and opcodes[idx + 1][0] == "equal":
            _, ni1, ni2, nj1, nj2 = opcodes[idx + 1]
            if ni2 > ni1 and nj2 > nj1 and _is_unit_word(a[ni1]):
                ext_i2 += 1
                ext_j2 += 1

        orig_str = " ".join(a[ext_i1:ext_i2])
        new_str = " ".join(b[ext_j1:ext_j2])
        orig_pat, new_pat = abstract_edit_pair(a[ext_i1:ext_i2], b[ext_j1:ext_j2])
        if orig_pat or new_pat:
            results.append((orig_str, new_str, orig_pat, new_pat))

    return results


def compute_edit_patterns(valid_rows: list[dict], top_k: int = 50) -> list[dict[str, Any]]:
    """Aggregate edit patterns across valid-pair rows (baseline vs suppressed)."""
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for r in valid_rows:
        base = r.get("baseline_text") or ""
        sup = r.get("suppressed_text") or ""
        if not base or not sup:
            continue
        sid = r.get("sample_id")
        for orig_s, new_s, orig_pat, new_pat in extract_edit_spans(base, sup):
            key = (orig_pat, new_pat)
            counts[key] += 1
            if len(examples[key]) < 2:
                examples[key].append({"sample_id": sid, "orig": orig_s, "new": new_s})

    total = sum(counts.values())
    out = []
    for (orig_pat, new_pat), cnt in counts.most_common(top_k):
        out.append(
            {
                "orig_pattern": orig_pat,
                "new_pattern": new_pat,
                "pattern": f"{orig_pat} => {new_pat}",
                "count": int(cnt),
                "share": round(cnt / total, 4) if total else 0.0,
                "examples": examples[(orig_pat, new_pat)],
            }
        )
    return out


def _resolve_phase1_runs(spec: list[str] | None) -> list[str]:
    if spec:
        return [str(r) for r in spec if r]
    found: list[tuple[float, str]] = []
    for p in RUNS_DIR.glob("phase1-*/job.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("type") == "phase1" and d.get("status") == "done":
            found.append((float(d.get("created") or 0.0), d["id"]))
    found.sort()
    return [rid for _, rid in found]


@register("phase1_5", kind="io", defaults=PHASE15_DEFAULTS)
def run_phase1_5(ctx: Ctx, cfg: dict) -> dict:
    run_ids = _resolve_phase1_runs(cfg.get("phase1_runs"))
    if not run_ids:
        raise RuntimeError("No completed phase1 runs found")

    ctx.log(f"phase1_5: reading {len(run_ids)} phase1 run(s): {run_ids}")
    rows: list[dict] = []
    valid_rows: list[dict] = []
    for rid in run_ids:
        hv = load_verdicts(rid)
        r_list = read_jsonl(RUNS_DIR / rid / "results.jsonl")
        for r in r_list:
            if r.get("verdict") == "error" or "baseline_tokens" not in r:
                continue
            r = dict(r)
            r["run_id"] = rid
            sid = r.get("sample_id")
            h = hv.get(sid) if sid is not None else None
            eff = "human_good" if h == "good" else ("human_bad" if h == "bad" else r.get("verdict"))
            r["effective_verdict"] = eff
            rows.append(r)
            if r.get("has_digits") and r.get("suppressed_text") and eff in ("cardinal", "human_good"):
                valid_rows.append(r)

    if not rows:
        raise RuntimeError(f"No usable rows in phase1 runs: {run_ids}")

    model_name = cfg.get("model") or next((r.get("model") for r in rows if r.get("model")), None) or load_settings()["model_active"]
    ctx.log(f"loading tokenizer for {model_name} (rows={len(rows)}, valid_pairs={len(valid_rows)})")
    ctx.progress(1, 5, f"tokenizer {model_name}")
    ctx.check()

    tokenizer = load_tokenizer(model_name)
    vocab_len = len(tokenizer)
    dig_ids = set(M.digit_token_ids(tokenizer, vocab_len))
    sep_ids = separator_token_ids(tokenizer, vocab_len)

    # Decode every token ID that actually appears in baseline_tokens (plus digits/separators)
    used_ids: set[int] = set(dig_ids) | set(sep_ids)
    for r in rows:
        used_ids.update(r.get("baseline_tokens") or [])
    tok_decode = {i: tokenizer.decode([i], skip_special_tokens=False) for i in used_ids}

    ctx.progress(2, 5, "token offsets & satellites")
    ctx.check()
    tok_res = analyse_tokens(
        token_rows=[r["baseline_tokens"] for r in rows if r.get("baseline_tokens")],
        digit_ids=dig_ids,
        sep_ids=sep_ids,
        tok_decode=tok_decode,
        window=int(cfg.get("window", 3)),
        min_count=int(cfg.get("min_count", 3)),
        z_min=float(cfg.get("z_min", 3.0)),
        min_share=float(cfg.get("min_share", 0.05)),
    )
    tok_res["satellites"]["rule"]["model"] = model_name

    ctx.progress(3, 5, "word prefix realization & units")
    ctx.check()
    text_sources: dict[str, list[str]] = {
        "model_baseline": [r.get("baseline_text") or "" for r in rows if r.get("baseline_text")],
        "model_suppressed": [r.get("suppressed_text") or "" for r in valid_rows if r.get("suppressed_text")],
    }
    if cfg.get("include_reference", True):
        text_sources["reference"] = [r.get("ref_text") or "" for r in rows if r.get("ref_text")]

    prefix_rel = word_prefix_realization(text_sources)
    units = compute_unit_after(text_sources, top_k=30)

    ctx.progress(4, 5, "edit patterns")
    ctx.check()
    edit_pats = compute_edit_patterns(valid_rows, top_k=50)

    summary = {
        "phase1_runs": run_ids,
        "model": model_name,
        "samples": len(rows),
        "valid_pairs": len(valid_rows),
        "digit_spans": tok_res["n_spans"],
        "hyphen_before_share": tok_res["hyphen_before_share"],
        "token_offsets": tok_res["token_offsets"],
        "pre_digit_chains": tok_res["pre_digit_chains"],
        "word_prefix_realization": prefix_rel,
        "unit_after": units,
        "edit_patterns": edit_pats,
        "satellites": tok_res["satellites"],
    }

    ctx.write_json("summary.json", summary)
    ctx.write_json("satellites.json", tok_res["satellites"])
    ctx.progress(5, 5, "done")
    ctx.log(
        f"done: spans={tok_res['n_spans']} hyphen_before_share={tok_res['hyphen_before_share']} "
        f"edit_patterns={len(edit_pats)} satellites_before={len(tok_res['satellites']['before'])} "
        f"satellites_after={len(tok_res['satellites']['after'])}"
    )

    return {
        "model": model_name,
        "phase1_runs": len(run_ids),
        "samples": len(rows),
        "valid_pairs": len(valid_rows),
        "digit_spans": tok_res["n_spans"],
        "hyphen_before_share": tok_res["hyphen_before_share"],
        "satellites_before": len(tok_res["satellites"]["before"]),
        "satellites_after": len(tok_res["satellites"]["after"]),
        "top_edit_pattern": edit_pats[0]["pattern"] if edit_pats else None,
    }
