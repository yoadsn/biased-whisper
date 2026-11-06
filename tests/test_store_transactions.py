import json
import sqlite3

import pytest

from bwhisper.corpus import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "corpus.db"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(store.SCHEMA)
    monkeypatch.setattr(store, "conn", lambda: connection)
    yield path, connection
    connection.close()


def test_duplicate_insert_rolls_back_and_releases_write_lock(db):
    path, connection = db
    assert store.add_sample("test", "one", "מספר 1", "one.wav", 3)
    assert store.add_sample("test", "one", "מספר 1", "one.wav", 3) is None
    assert not connection.in_transaction
    with sqlite3.connect(path, timeout=0.1) as other:
        other.execute("UPDATE samples SET batch='other writer'")


def test_reference_batch_commits_before_returning_and_another_writer_can_write(db):
    path, connection = db
    sid = store.add_sample("test", "one", "מספר אחד", "one.wav", 3)
    store.update_references([(sid, "מספר 1", {"reference_version": 2})])
    assert not connection.in_transaction
    with sqlite3.connect(path, timeout=0.1) as other:
        assert other.execute("SELECT text,ref_digits,meta FROM samples").fetchone() == (
            "מספר 1", 1, json.dumps({"reference_version": 2}))
        other.execute("UPDATE samples SET batch='other writer'")
