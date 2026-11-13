import sqlite3
from pathlib import Path

import pytest

from bwhisper.corpus import explore, hf_cache, pockettts, store


def row(text):
    return {"path": "/audio/a.wav", "start": 1.0, "duration": 3.0, "transcript": text, "words": []}


def test_pockettts_content_filter():
    assert pockettts.candidate(row("הלכתי ל-Facebook"), "plenum", Path("t.jsonl"), 1, 1.5, 28) is None
    c = pockettts.candidate(row("הלכתי ל-Facebook"), "plenum", Path("t.jsonl"), 1, 1.5, 28, None, "latin")
    assert c and c["text"] == "הלכתי ל-Facebook"
    assert pockettts.candidate(row("בשנת 2024"), "plenum", Path("t.jsonl"), 1, 1.5, 28, None, "latin") is None
    assert pockettts.candidate(row("בשנת 2024"), "plenum", Path("t.jsonl"), 1, 1.5, 28) is not None
    with pytest.raises(ValueError):
        pockettts.candidate(row("x"), "plenum", Path("t.jsonl"), 1, 1.5, 28, None, "bogus")


def test_hf_windows_follow_content():
    segs = [{"start": 0, "end": 4, "text": "שלום"}, {"start": 4, "end": 8, "text": "ב-eBay קניתי"},
            {"start": 8, "end": 12, "text": "ב-2024 היה"}, {"start": 12, "end": 16, "text": "סוף"}]
    assert hf_cache.build_windows(segs, 3, 10, "latin") == [(1, 1)]
    assert hf_cache.build_windows(segs, 3, 10, "digits") == [(2, 2)]


def test_explore_query_selection():
    assert explore.search_query("digits") == r"\d"
    assert "A-Za-z" in explore.search_query("latin")
    assert explore.search_query("digits", " foo ") == "foo"
    assert explore.search_query("saved") is None
    with pytest.raises(ValueError):
        explore.search_query("bogus")


def test_search_term_in_copied_curl_is_templated():
    url = "https://explore.ivrit.ai/api/search?q=%D7%91%D7%A8&mode={mode}&page={page1}"
    assert explore.template_search_url(url) == "https://explore.ivrit.ai/api/search?q={q}&mode={mode}&page={page1}"
    assert explore.template_search_url("https://x/api/search?mode=a&q={q}") == "https://x/api/search?mode=a&q={q}"
    assert explore.template_search_url("") == ""


def test_import_curl_templates_any_search_term(tmp_path, monkeypatch):
    monkeypatch.setattr(explore, "CFG_PATH", tmp_path / "explore.json")
    cmd = ("curl 'https://explore.ivrit.ai/api/search?q=%D7%91%D7%A8%D7%99%D7%90%D7%95%D7%AA&mode=exact&page=1"
           "&per_page=50&shuffle=true&seed=5' -H 'cookie: a=b'")
    cfg = explore.import_curl(cmd)
    assert "q={q}" in cfg["search_url"] and "mode={mode}" in cfg["search_url"]


def test_latin_counts_filter_and_migration(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)  # database created before the Latin axis existed
    old.executescript("""CREATE TABLE samples (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
        source_key TEXT NOT NULL, text TEXT NOT NULL, duration REAL, audio_path TEXT, start REAL, end REAL,
        ref_digits INTEGER DEFAULT 0, meta TEXT DEFAULT '{}', batch TEXT, created REAL, UNIQUE(source, source_key));""")
    old.execute("INSERT INTO samples(source,source_key,text) VALUES ('t','1','ה-eBay מצוין')")
    old.execute("INSERT INTO samples(source,source_key,text) VALUES ('t','2','שנת 2024')")
    old.commit(); old.close()
    monkeypatch.setattr(store, "DB_PATH", path)
    monkeypatch.setattr(store, "_local", type("L", (), {})())
    assert [s["text"] for s in store.query(latin_only=True)] == ["ה-eBay מצוין"]
    assert store.count(latin_only=True) == 1 and store.count(digits_only=True) == 0
    sid = store.add_sample("t", "3", "x2 ו-Google", "a.wav", 2)
    assert store.get(sid)["ref_latin"] == 1
    store.update_text(sid, "רק עברית")
    assert store.get(sid)["ref_latin"] == 0
    assert store.stats()["sources"][0]["with_latin"] == 1
