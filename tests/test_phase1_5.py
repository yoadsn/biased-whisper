from collections import Counter

import pytest

from bwhisper import model as M
from bwhisper.hebrew_numbers import split_number_prefix
from bwhisper.phase1_5 import (
    abstract_edit_pair,
    abstract_span,
    analyse_tokens,
    classify_text_prefix_realizations,
    extract_edit_spans,
    extract_units_after,
    fightin_words_z,
    find_digit_spans,
    load_tokenizer,
    separator_token_ids,
    word_prefix_realization,
)


def test_split_number_prefix():
    assert split_number_prefix("באלפיים") == ("ב", "אלפיים")
    assert split_number_prefix("ותשעה") == ("ו", "תשעה")
    assert split_number_prefix("שלושה") == ("", "שלושה")
    assert split_number_prefix("מהשלישי") == ("מה", "שלישי")
    assert split_number_prefix("בנובמבר") is None
    assert split_number_prefix("2025") is None


def test_find_digit_spans_absorbs_inter_digit_separators():
    digit_ids = {10, 11, 12}
    sep_ids = {20, 21}  # e.g. ',', ':'
    other = 99
    # [other, dig, ',', dig, ':', dig, ',', other, dig]
    seq = [other, 10, 20, 11, 21, 12, 20, other, 10]
    spans = find_digit_spans(seq, digit_ids, sep_ids)
    # Trailing separator 20 at index 6 is NOT between two digit tokens, so first span is (1, 6)
    assert spans == [(1, 6), (8, 9)]


def test_pattern_abstraction_examples():
    assert abstract_edit_pair("ב-2025", "באלפיים עשרים וחמש") == ("ב-<D>", "ב<N>")
    assert abstract_edit_pair("80%", "שמונים אחוז") == ("<D>%", "<N> אחוז")
    assert abstract_edit_pair("ו-9", "ותשעה") == ("ו-<D>", "ו<N>")
    assert abstract_edit_pair("מה-24", "מהעשרים וארבעה") == ("מה-<D>", "מה<N>")
    assert abstract_span("2025") == "<D>"

    spans = extract_edit_spans("זה קרה ב-2025 בישראל", "זה קרה באלפיים עשרים וחמש בישראל")
    assert len(spans) == 1
    orig_s, new_s, orig_pat, new_pat = spans[0]
    assert (orig_pat, new_pat) == ("ב-<D>", "ב<N>")

    spans_pct = extract_edit_spans("יש 80% הצלחה", "יש שמונים אחוז הצלחה")
    assert len(spans_pct) == 1
    assert spans_pct[0][2:] == ("<D>%", "<N> אחוז")


def test_fightin_words_z_ordering():
    # Token 1 is strongly over-represented in fg vs bg; token 2 matches bg; token 3 is under-represented
    bg = Counter({1: 50, 2: 500, 3: 500, 4: 1000})
    fg = Counter({1: 40, 2: 25, 3: 5, 4: 50})
    fw = fightin_words_z(fg, bg, alpha0=500.0)
    assert fw[1][1] > fw[2][1] > fw[3][1]
    assert fw[1][1] > 3.0
    assert fw[3][1] < 0.0


def test_prefix_realization_and_units():
    rels = classify_text_prefix_realizations("בשנת ב-2025 או ב2026 היו 80% וגם באלפיים עשרים וחמש או ושלושה אנשים")
    assert ("ב", "hyphen_digit") in rels
    assert ("ב", "glued_digit") in rels
    assert ("", "bare_digit") in rels
    assert ("ב", "fused_word") in rels
    assert ("ו", "fused_word") in rels

    tbl = word_prefix_realization({"src": ["ב-2025 באלפיים עשרים וחמש 10"]})
    assert tbl["src"]["ב"]["hyphen_digit"] == 1
    assert tbl["src"]["ב"]["fused_word"] == 1
    assert tbl["src"][""]["bare_digit"] == 1

    units = extract_units_after("יש 80% הצלחה ו-50 שקלים או שמונים אחוז הצלחה")
    assert ("digit", "%") in units
    assert ("digit", "שקלים") in units
    assert ("word", "אחוז") in units


def test_tokenizer_and_analyse_tokens():
    try:
        tok = load_tokenizer("yoad/whisper-tiny")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")

    dig_ids = set(M.digit_token_ids(tok, len(tok)))
    sep_ids = separator_token_ids(tok, len(tok))
    assert len(dig_ids) > 100
    assert len(sep_ids) >= 4

    seq = tok.encode(" שלום ב-2025 יש 80% הצלחה ב-2024 וב-2023", add_special_tokens=False)
    filler = tok.encode(" אנחנו מדברים היום בדיון חשוב מאוד בכנסת על נושאים רבים ומגוונים", add_special_tokens=False)
    spans = find_digit_spans(seq, dig_ids, sep_ids)
    assert len(spans) == 4
    tok_decode = {i: tok.decode([i], skip_special_tokens=False) for i in set(seq + filler)}
    res = analyse_tokens([seq + filler * 3] * 4, dig_ids, sep_ids, tok_decode, window=2, min_count=3, z_min=1.0, min_share=0.05)
    assert res["n_spans"] == 16
    assert res["hyphen_before_share"] == pytest.approx(0.75)
    assert any(s["tok"] == "-" for s in res["satellites"]["before"])
