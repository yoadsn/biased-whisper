from bwhisper.latin_script import latin_kind, latin_skeleton, letter_name_score, span_verdict, translit_score


def test_words_transliterate():
    assert span_verdict(" eBay", " אי-ביי")[0] == "translit"
    assert span_verdict("ב-Facebook", "בפייסבוק")[0] == "translit"
    assert span_verdict(" Channel Go-To-Market Agency", " צ'אנל גו טו מרקט אייג'נסי")[0] == "translit"


def test_translation_is_not_transliteration():
    assert span_verdict(" Facebook", " חברתית")[0] == "other_hebrew"
    assert translit_score("computer", "מחשב") < 0.6


def test_letter_names():
    assert span_verdict(" v1 עד vn", " וי אחד עד וי אן")[0] == "translit"
    assert span_verdict("ה-PPP", "הפי פי פי")[0] == "translit"
    assert letter_name_score("FM", "אף אם") == 1.0


def test_short_skeleton_does_not_match_by_accident():
    assert span_verdict("מ-n", "מן")[0] == "other_hebrew"


def test_verdict_edges():
    assert span_verdict("YC", "...")[0] == "no_hebrew"
    assert span_verdict("YC", "יו YC")[0] == "latin_leak"


def test_kind_and_skeleton():
    assert latin_kind("eBay") == "word" and latin_kind("PPP") == "acronym" and latin_kind("v1") == "symbol"
    assert latin_skeleton("Facebook") == "pkbk"
