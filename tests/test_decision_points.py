import json

import pytest

from bwhisper import phase1


@pytest.fixture
def points(tmp_path, monkeypatch):
    monkeypatch.setattr(phase1, "RUNS_DIR", tmp_path)
    run = tmp_path / "phase1"
    run.mkdir()
    rows = [{"sample_id": sid, "has_digits": True, "cardinal_token": 7, "verdict": "cardinal"}
            for sid in range(1, 5)]
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (run / "verdicts.json").write_text(json.dumps({"4": "bad"}))


def test_explicit_discovery_subset_filters_before_limit(points):
    got = phase1.decision_points(["phase1"], ["cardinal"], 1, sample_ids=[2, 3])
    assert [p["sample_id"] for p in got] == [2]
    assert got[0]["run_id"] == "phase1"


def test_legacy_selection_is_unchanged(points):
    assert [p["sample_id"] for p in phase1.decision_points(["phase1"], ["cardinal"], 2)] == [1, 2]


@pytest.mark.parametrize("ids", [[], [1, 1], [True], [-1], [0], ["1"]])
def test_invalid_discovery_subset_is_rejected(points, ids):
    with pytest.raises(ValueError, match="sample_ids"):
        phase1.decision_points(["phase1"], ["cardinal"], sample_ids=ids)


@pytest.mark.parametrize("ids", [[4], [99]])
def test_subset_does_not_silently_drop_missing_or_human_bad_points(points, ids):
    with pytest.raises(ValueError, match="accepted"):
        phase1.decision_points(["phase1"], ["cardinal"], sample_ids=ids)
