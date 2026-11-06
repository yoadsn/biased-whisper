from types import SimpleNamespace

import pytest

from bwhisper import phase1
from bwhisper.config import PHASE1_DEFAULTS
from bwhisper.setup_scale import split_samples


def test_recording_disjoint_reproducible_split():
    samples = [{"id": 2 * i + j, "source": "explore", "meta": {"uuid": f"episode-{i}"}}
               for i in range(30) for j in range(2)]
    split = split_samples(samples, 0.2, 0)
    assert split == split_samples(samples, 0.2, 0)
    assert split["discovery"] and split["evaluation"]
    assert not set(split["discovery"]) & set(split["evaluation"])
    assert len(split["discovery"]) + len(split["evaluation"]) == len(samples)
    for i in range(30):
        assert (2 * i in split["evaluation"]) == (2 * i + 1 in split["evaluation"])
    with pytest.raises(ValueError, match="recording identity"):
        split_samples([{"id": 1, "source": "explore", "meta": {}}], 0.2, 0)


def test_pockettts_split_uses_recording_not_slice():
    samples = [{"id": i, "source": "pockettts_plenum", "meta": {"recording": "same.wav"}}
               for i in range(10)]
    split = split_samples(samples, 0.2, 0)
    assert len(split["groups"]) == 1
    assert len(split["discovery"]) in (0, 10)


def test_empty_explicit_phase1_cohort_does_not_fall_back(monkeypatch):
    def fail(**kwargs):
        pytest.fail("must not query a random cohort")
    monkeypatch.setattr(phase1.store, "query", fail)
    assert phase1.select_samples({**PHASE1_DEFAULTS, "sample_ids": []}) == []


def test_parallel_phase1_shards_and_merges_without_model_execution(monkeypatch):
    import torch

    samples = [{"id": i} for i in range(13)]
    monkeypatch.setattr(phase1, "select_samples", lambda cfg: samples)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    seen = []
    progress = []
    def fake_run(ctx, cfg, shard, device=None):
        seen.append((device, [s["id"] for s in shard]))
        ctx.progress(len(shard), len(shard))
        return {"samples": len(shard), "verdicts": {"cardinal": len(shard)},
                "with_digits": len(shard), "mean_p_digit_at_decision": 0.5, "model": "fake"}
    monkeypatch.setattr(phase1, "_run_samples", fake_run)
    ctx = SimpleNamespace(log=lambda msg: None, progress=lambda done, total, message=None: progress.append((done, total)))
    devices = [f"cuda:{i}" for i in range(8)]
    result = phase1.run_phase1(ctx, {**PHASE1_DEFAULTS, "devices": devices})
    assert sorted(i for _, ids in seen for i in ids) == list(range(13))
    assert {d for d, _ in seen} == set(devices)
    assert result["samples"] == 13 and result["verdicts"] == {"cardinal": 13}
    assert result["mean_p_digit_at_decision"] == 0.5
    assert progress[-1] == (13, 13)


@pytest.mark.parametrize("devices", [["cuda:0", "cuda:0"], ["cuda:99"], "cuda:0"])
def test_parallel_devices_validation(monkeypatch, devices):
    import torch

    monkeypatch.setattr(phase1, "select_samples", lambda cfg: [{"id": 1}])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    with pytest.raises(ValueError):
        phase1.run_phase1(SimpleNamespace(), {**PHASE1_DEFAULTS, "devices": devices})


def test_explore_fetch_fails_before_search_without_auth(monkeypatch):
    from bwhisper.corpus import jobs

    monkeypatch.setattr(jobs.explore, "check_auth", lambda: {"ok": False, "reason": "no cookie configured"})
    monkeypatch.setattr(jobs.explore, "fetch_episodes_batch", lambda **kw: pytest.fail("must not search"))
    with pytest.raises(RuntimeError, match="authentication required"):
        jobs.fetch_explore(None, jobs.EXPLORE_FETCH_DEFAULTS)


def test_setup_writes_configs_without_submitting_phases(tmp_path, monkeypatch):
    import torch
    from bwhisper import jobs, setup_scale

    (tmp_path / "model.safetensors").touch()
    samples = [{"id": i, "source": "explore", "meta": {"uuid": f"ep-{i}"}} for i in range(100)]
    monkeypatch.setattr(setup_scale.store, "query", lambda **kw: samples)
    monkeypatch.setattr(setup_scale, "save_settings", lambda patch: patch)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(name="fake", total_memory=123))
    class FakeTensor:
        def __matmul__(self, other): return self
        def __getitem__(self, index): return self
        def item(self): return 16
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: FakeTensor())
    monkeypatch.setattr(jobs.Manager, "submit", lambda *args, **kwargs: pytest.fail("must not submit phases"))
    monkeypatch.setattr(jobs.Manager, "run_sync", lambda *args, **kwargs: pytest.fail("must not run phases"))
    artifacts = {}
    ctx = SimpleNamespace(check=lambda: None, log=lambda msg: None,
                          write_json=lambda name, obj: artifacts.update({name: obj}))
    summary = setup_scale.setup_scale(ctx, {**setup_scale.DEFAULTS, "model": str(tmp_path),
                                          "devices": ["cuda:0"], "check_model": False})
    assert summary["phases_started"] is False
    assert summary["discovery"] + summary["evaluation"] == 100
    assert set(artifacts["phase1.json"]["sample_ids"]).isdisjoint(artifacts["phase3.template.json"]["eval_sample_ids"])
    assert artifacts["phase1.json"]["devices"] == ["cuda:0"]
    assert artifacts["phase3.template.json"]["eval_limit"] <= 50
    assert artifacts["explore-fetch.json"]["target"] == 10000
