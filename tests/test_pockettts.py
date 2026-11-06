import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bwhisper.corpus import pockettts


def test_original_text_and_aligned_boundaries():
    row = {"path": "/audio/session.wav", "start": 10.25, "duration": 3.5,
           "original_start": 9.0, "original_duration": 5.0, "transcript": "היום ארבעים ושניים אנשים",
           "words": [{"word": w} for w in ["היום", "ארבעים", "ושניים", "אנשים"]],
           "alt_words": {"norms": [{"from": 1, "to": 2, "replace": ["42"]}]}}
    c = pockettts.candidate(row, "plenum", Path("/data/train_aligned.jsonl"), 7, 1.5, 28)
    assert c["text"] == "היום 42 אנשים"
    assert c["meta"]["text_origin"] == "restored_pre_normalization"
    assert c["start"] == 10.25 and c["duration"] == 3.5
    assert c["meta"]["recording"] == "/audio/session.wav"
    row.pop("alt_words")
    assert pockettts.candidate(row, "plenum", Path("train_aligned.jsonl"), 7, 1.5, 28) is None


def test_crowd_uses_original_source_text_not_normalized_transcript():
    row = {"path": "/audio/ep/1.flac", "clip": "show/ep/1", "start": 0.1,
           "duration": 4, "transcript": "המאה התשע עשרה"}
    c = pockettts.candidate(row, "crowd_transcribe_v5", Path("train_aligned.jsonl"), 1,
                            1.5, 28, {"show/ep/1": "המאה ה-19"})
    assert c["text"] == "המאה ה-19"
    assert c["meta"]["text_origin"] == "crowd_source_index"
    with pytest.raises(ValueError, match="source index"):
        pockettts.original_text(row, "crowd_transcribe_v5", {})


def test_inverse_spans_are_applied_right_to_left_and_validated():
    row = {"words": [{"word": w} for w in ["חמש", "עשרה", "ועשרים", "ושלוש"]],
           "alt_words": {"norms": [{"from": 0, "to": 1, "replace": ["15"]},
                                    {"from": 2, "to": 3, "replace": ["ו-23"]}]}}
    assert pockettts.original_text(row, "committee")[0] == "15 ו-23"
    row["alt_words"]["norms"][1]["from"] = 1
    with pytest.raises(ValueError):
        pockettts.original_text(row, "committee")


@pytest.mark.parametrize("start,duration", [(-1, 5), (0, 31), (0, 0), (float("nan"), 5), (0, float("inf"))])
def test_invalid_ranges(start, duration):
    assert pockettts.candidate({"path": "a.wav", "start": start, "duration": duration,
                               "transcript": "שלום 12"}, "plenum", Path("x"), 1, 1.5, 28) is None


def test_alignment_error_is_not_used():
    assert pockettts.candidate({"path": "a.wav", "start": 0, "duration": 5,
                               "transcript": "שלום 12", "alignment_error": "failed"},
                              "plenum", Path("x"), 1, 1.5, 28) is None


class Context:
    job = SimpleNamespace(id="test")

    def __init__(self):
        self.summary = None
        self.rows = []

    def check(self): pass
    def log(self, msg): pass
    def progress(self, *args): pass
    def write_json(self, name, obj): self.summary = obj
    def append_jsonl(self, name, obj): self.rows.append((name, obj))


def test_import_balances_backfills_deduplicates_and_reports_shortfall(tmp_path, monkeypatch):
    monkeypatch.setattr(pockettts.store, "update_references", lambda updates: None)
    monkeypatch.setattr(pockettts, "crowd_originals", lambda root: {})
    for source, n in [("plenum", 6), ("committee", 2), ("crowd_transcribe_v5", 0)]:
        d = tmp_path / source
        d.mkdir()
        rows = [{"path": f"{i}.wav", "start": 10, "duration": 3, "transcript": f"מספר {i}"} for i in range(n)]
        for split in ["train", "valid"]:
            (d / f"{split}_aligned.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    inserted = []
    monkeypatch.setattr(pockettts.store, "query", lambda **kw: [])
    monkeypatch.setattr(pockettts.store, "audio_path_for", lambda s, k: tmp_path / f"{k}.wav")
    monkeypatch.setattr(pockettts, "crop_to_wav", lambda src, dst, start, end: end - start)
    def add(*args, **kwargs):
        inserted.append((args, kwargs))
        return len(inserted)
    monkeypatch.setattr(pockettts.store, "add_sample", add)
    cfg = {**pockettts.DEFAULTS, "root": str(tmp_path), "limit": 10, "workers": 2}
    ctx = Context()
    summary = pockettts.import_aligned(ctx, cfg)
    assert summary["added"] == 8 and summary["shortfall"] == 2
    assert summary["by_source"] == {"pockettts_plenum": 6, "pockettts_committee": 2}
    assert not summary["target_reached"]
    assert sum(v for k, v in summary["scan"].items() if k.endswith(":duplicates")) == 8
    assert all(args[2].startswith("מספר ") and kw["start"] == 10 and kw["end"] == 13 for args, kw in inserted)


def test_reject_english_sources(tmp_path):
    with pytest.raises(ValueError):
        pockettts.import_aligned(Context(), {**pockettts.DEFAULTS, "sources": ["hifitts2"]})


def test_existing_audio_reference_is_repaired_without_recropping(tmp_path, monkeypatch):
    row = {"path": "/audio/session.wav", "start": 1, "duration": 3,
           "transcript": "מספר ארבעים ושתיים", "words": [{"word": w} for w in ["מספר", "ארבעים", "ושתיים"]],
           "alt_words": {"norms": [{"from": 1, "to": 2, "replace": ["42"]}]}}
    path = tmp_path / "plenum" / "train_aligned.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(row) + "\n")
    c = pockettts.candidate(row, "plenum", path, 1, 1.5, 28)
    old = {"id": 7, "source": c["source"], "source_key": c["key"], "meta": {"recording": "old"}}
    monkeypatch.setattr(pockettts.store, "query", lambda **kw: [old])
    repairs = []
    monkeypatch.setattr(pockettts.store, "update_references", lambda updates: repairs.extend(updates))
    monkeypatch.setattr(pockettts, "crop_to_wav", lambda *args: pytest.fail("must reuse existing audio"))
    summary = pockettts.import_aligned(Context(), {**pockettts.DEFAULTS, "root": str(tmp_path),
                                                 "sources": ["plenum"], "splits": ["train"], "limit": 1})
    assert summary["references_repaired"] == 1 and summary["added"] == 0
    assert repairs[0][:2] == (7, "מספר 42")
    assert repairs[0][2]["reference_version"] == 2


def test_requires_aligned_manifests(tmp_path):
    with pytest.raises(FileNotFoundError):
        pockettts.import_aligned(Context(), {**pockettts.DEFAULTS, "root": str(tmp_path)})


def test_real_audio_crop_uses_aligned_offset(tmp_path):
    import array
    import shutil
    import wave
    from bwhisper.audio import crop_to_wav

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    src, dst = tmp_path / "long.wav", tmp_path / "slice.wav"
    # Distinct one-second constant levels make an incorrect source offset visible.
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", [1000] * 32000 + [2000] * 32000 + [3000] * 32000).tobytes())
    c = pockettts.candidate({"path": str(src), "start": 2, "duration": 2,
                             "original_start": 0, "transcript": "מספר 2"},
                            "plenum", tmp_path / "train_aligned.jsonl", 1, 1.5, 28)
    assert crop_to_wav(c["path"], dst, c["start"], c["start"] + c["duration"]) == 2
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        values = array.array("h", w.readframes(w.getnframes()))
        assert min(values) == max(values) == 2000
