import pytest

from bwhisper.number_values import same_values, value_sequence


@pytest.mark.parametrize("digits,cardinal", [
    ("24 שעות", "עשרים וארבע שעות"),
    ("בשנת 1948 הוקמה המדינה", "בשנת אלף תשע מאות ארבעים ושמונה הוקמה המדינה"),
    ("ב-2024", "באלפיים עשרים וארבע"),
    ("800,000 נשים", "שמונה מאות אלף נשים"),
    ("60 אלף", "שישים אלף"),
    ("13 חברים", "שלושה עשר חברים"),
    ("12 נשים", "שתים עשרה נשים"),
    ("200 שקלים", "מאתיים שקלים"),
    ("2 מיליון", "שני מיליון"),
    ("0", "אפס"),
])
def test_known_integer_values(digits, cardinal):
    assert same_values(digits, cardinal)


@pytest.mark.parametrize("digits,wrong", [
    ("14 ביוני", "עשר ביוני"),
    ("1948", "אלף תשע מאות תשע מאות שמונה"),
    ("1865", "אלף שמונה מאות שישים וחמ"),
    ("2005", "חמש"),
    ("1 בינואר 2025", "האחד בינואר"),
    ("2024", "אלפיים וארבע"),
    ("24 שעות", "שעות"),
])
def test_wrong_or_missing_values_are_not_accepted(digits, wrong):
    assert not same_values(digits, wrong)


@pytest.mark.parametrize("text", ["3.14", "1/2", "01.01.2025", "אלפי אנשים", "מאות אנשים",
                                 "עשרים עשרים ושתיים", "תשע שתיים תשע", "חצי", "25%"])
def test_unsupported_is_unknown_not_a_guessed_value(text):
    assert value_sequence(text) is None


def test_mention_order_and_unchanged_word_numbers_are_preserved():
    assert value_sequence("בין שני אנשים בכביש 60 ובכביש 24") == [2, 60, 24]
    assert not same_values("24 ואז 60", "שישים ואז עשרים וארבע")
    assert not same_values("בלי מספרים", "בלי מספרים")
