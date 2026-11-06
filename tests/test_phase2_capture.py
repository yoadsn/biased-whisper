import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bwhisper import model as M, phase2
from bwhisper.config import PHASE2_DEFAULTS
from bwhisper.paramspace import ParamSpace
from bwhisper.phase2_capture import _reference, compare_sparse, sparse_linear


def space_and_gradient(device="cpu"):
    model = torch.nn.Sequential(torch.nn.Linear(13, 17), torch.nn.Linear(17, 7)).to(device)
    space = ParamSpace(model, "all")
    g = torch.randn(space.total, generator=torch.Generator().manual_seed(42)).to(device)
    return space, g


@pytest.mark.parametrize("normalize", ["rms", "none"])
@pytest.mark.parametrize("d0", [0.0, 1.3, 20.0, -20.0])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_device_sparse_capture_matches_reference(normalize, d0, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    space, g = space_and_gradient(device)
    cfg = {**PHASE2_DEFAULTS, "normalize": normalize}
    actual = sparse_linear(g, space, d0, cfg, 37)
    reference = _reference(g.cpu(), space, d0, cfg, 37)
    reports = compare_sparse(reference, actual, g.cpu(), space, cfg, d0)
    assert all(r["passed"] for r in reports.values()), reports
    for name in reference:
        if reference[name]["sparse"] is not None:
            np.testing.assert_array_equal(reference[name]["sparse"]["idx"], actual[name]["sparse"]["idx"])


def test_both_linear_directions_reuse_one_selection(monkeypatch):
    space, g = space_and_gradient()
    calls = []
    topk = torch.topk
    def count(*args, **kwargs):
        calls.append(1)
        return topk(*args, **kwargs)
    monkeypatch.setattr(torch, "topk", count)
    out = sparse_linear(g, space, 0.0, PHASE2_DEFAULTS, 30)
    assert len(calls) == 1
    a, b = out["to_digits"]["sparse"], out["to_cardinal"]["sparse"]
    assert a["idx"] is b["idx"]
    np.testing.assert_array_equal(a["val"], -b["val"])
    np.testing.assert_array_equal(a["rms"], b["rms"])
    np.testing.assert_array_equal(a["energy"], b["energy"])


def test_zero_gradient_payload_and_nonfinite_rejection():
    space, g = space_and_gradient()
    g.zero_()
    out = sparse_linear(g, space, 0, PHASE2_DEFAULTS, 10)
    for row in out.values():
        assert np.all(row["sparse"]["val"] == 0)
        assert np.all(row["sparse"]["energy"] == 0)
        assert np.all(row["sparse"]["rms"] == 1)
    g[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        sparse_linear(g, space, 0, PHASE2_DEFAULTS, 10)


def test_flat_grad_default_cpu_and_explicit_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(2, 3).to(device)
    space = ParamSpace(model, "all")
    model.weight.grad = torch.ones_like(model.weight)
    expected = torch.tensor([1.] * 6 + [0.] * 3)
    torch.testing.assert_close(space.flat_grad(), expected)
    assert space.flat_grad().device.type == "cpu"
    packed = space.flat_grad(device=device)
    assert str(packed.device) == device
    torch.testing.assert_close(packed.cpu(), expected)


def test_parity_rejects_changed_support():
    space, g = space_and_gradient()
    cfg = {**PHASE2_DEFAULTS, "normalize": "none"}
    reference = _reference(g, space, 0, cfg, 30)
    actual = sparse_linear(g, space, 0, cfg, 30)
    actual["to_digits"]["sparse"]["idx"] = np.arange(30, dtype=np.int64)
    assert not compare_sparse(reference, actual, g, space, cfg, 0)["to_digits"]["passed"]


def test_release_unused_model_cache_keeps_active_replica(monkeypatch):
    active, unused = SimpleNamespace(device="cpu"), SimpleNamespace(device="cpu")
    cache = {("active", "cpu"): active, ("unused", "cpu"): unused}
    monkeypatch.setattr(M, "_cache", cache)
    assert M.unload_unused(active) == 1
    assert list(cache.values()) == [active]


class Context:
    def __init__(self, directory):
        self.dir = directory
        self.records = []
    def log(self, message): pass
    def write_json(self, name, value):
        (self.dir / name).write_text(json.dumps(value))
    def append_jsonl(self, name, value):
        self.records.append(value)


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    monkeypatch.setattr(phase2, "RUNS_DIR", tmp_path)
    space = ParamSpace(torch.nn.Linear(2, 2), "all")
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir(); target.mkdir()
    cfg = {**PHASE2_DEFAULTS, "sample_k": 4, "resume_run": "old"}
    old = {**cfg, "resume_run": None}
    (source / "job.json").write_text(json.dumps({"type": "phase2", "status": "cancelled", "config": old}))
    (source / "space.json").write_text(json.dumps({"signature": space.signature}))
    pt = {"run_id": "p1", "sample_id": 1, "model": "toy"}
    rows = []
    for kind in phase2.NUDGES:
        for directory in (source, target):
            (directory / "vectors" / kind).mkdir(parents=True)
        row = {"key": "p1__1", "type": kind, "sample_id": 1, "run_id": "p1", "d0": 0}
        rows.append(row)
        np.savez(source / "vectors" / kind / "p1__1.npz", idx=np.arange(4, dtype=np.uint32),
                 val=np.ones(4, np.float32), rms=np.ones(len(space.names), np.float32),
                 energy=np.ones(len(space.names), np.float64))
    (source / "points.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Context(target), cfg, space, [pt], source


def test_resume_reuses_complete_captures_without_changing_source(resume_case):
    ctx, cfg, space, pts, source = resume_case
    before = (source / "vectors/to_digits/p1__1.npz").read_bytes()
    complete = phase2.resume_captures(ctx, cfg, space, "toy", pts)
    assert set(complete) == {"p1__1"}
    energies = {kind: np.zeros(len(space.names)) for kind in phase2.NUDGES}
    phase2.restore_capture(ctx, cfg, "p1__1", complete["p1__1"], energies)
    assert (ctx.dir / "vectors/to_digits/p1__1.npz").read_bytes() == before
    assert (source / "vectors/to_digits/p1__1.npz").read_bytes() == before
    assert len(ctx.records) == 2 and all(r["reused_from"] == "old" for r in ctx.records)
    assert all(np.isclose(v.sum(), 1) for v in energies.values())


def test_resume_ignores_incomplete_point(resume_case):
    ctx, cfg, space, pts, source = resume_case
    (source / "vectors/to_digits/p1__1.npz").unlink()
    assert not phase2.resume_captures(ctx, cfg, space, "toy", pts)


def test_resume_rejects_configuration_change(resume_case):
    ctx, cfg, space, pts, source = resume_case
    with pytest.raises(ValueError, match="normalize"):
        phase2.resume_captures(ctx, {**cfg, "normalize": "none"}, space, "toy", pts)
    with pytest.raises(ValueError, match="invalid resume_run"):
        phase2.resume_captures(ctx, {**cfg, "resume_run": "../old"}, space, "toy", pts)
