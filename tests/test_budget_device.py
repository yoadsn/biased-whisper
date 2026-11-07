import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bwhisper import budget, budget_device, phase2
from bwhisper.jobs import Cancelled
from bwhisper.paramspace import ParamSpace


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("normalize", ["none", "rms"])
def test_nested_rank_and_raw_energy_match_reference(device, normalize):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = torch.nn.Sequential(torch.nn.Linear(17, 19), torch.nn.Linear(19, 5)).to(device)
    space = ParamSpace(model, "all")
    g = torch.randn(space.total, generator=torch.Generator().manual_seed(12)).to(device)
    ks = [7, 41, 103]
    raw, total = budget_device.raw_energy_curve(g, ks)
    rank = budget_device.ranked_support(g, space, normalize, max(ks))
    ref = g.cpu()
    score, _ = phase2.selection_score(ref, space, normalize)
    cumulative = ref.square().sort(descending=True).values.double().cumsum(0)
    for i, k in enumerate(ks):
        idx, bounds = budget_device.sorted_support(space, rank, k)
        expected = budget._topk(score, k)
        np.testing.assert_array_equal(idx.cpu().numpy(), expected)
        assert raw[i] == pytest.approx(float(cumulative[k - 1]) / float(ref.square().sum()), rel=2e-6)
        assert bounds == space.sparse_bounds(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sparse_slice_edits_and_undo_are_exact(device, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 3)).to(device)
    space = ParamSpace(model, "all")
    before = [p.detach().clone() for p in space.params]
    idx = np.arange(0, space.total, 2, dtype=np.int64)
    vals = np.linspace(-0.1, 0.2, len(idx), dtype=np.float32)
    monkeypatch.setattr(space, "tensor_ids", lambda *args: pytest.fail("must not scan a tensor-id vector"))
    undo = space.add_sparse(idx, vals)
    cpu_edited = [p.detach().clone() for p in space.params]
    space.undo_sparse(undo)
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))
    ti, tv = torch.from_numpy(idx).to(device), torch.from_numpy(vals).to(device)
    undo = space.add_sparse_device(ti, tv)
    assert all(torch.equal(a, b) for a, b in zip(cpu_edited, space.params))
    space.undo_sparse(undo)
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


def test_edit_failure_rolls_back_earlier_tensors(monkeypatch):
    space = ParamSpace(torch.nn.Linear(3, 2), "all")
    before = [p.detach().clone() for p in space.params]
    original = torch.Tensor.index_copy_
    calls = []
    def fail_second(tensor, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated edit failure")
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "index_copy_", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        space.add_sparse_device(torch.arange(space.total), torch.ones(space.total))
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


@pytest.mark.parametrize("indices", [[2, 1], [1, 1], [-1], [100]])
def test_bad_sparse_indices_rejected(indices):
    space = ParamSpace(torch.nn.Linear(2, 2), "all")
    with pytest.raises(ValueError):
        space.add_sparse(np.array(indices), np.ones(len(indices)))


class Context:
    def __init__(self, directory, cfg, stop_phase=None):
        directory.mkdir()
        self.dir = directory
        self.stop_phase = stop_phase
        self.stopped = False
        self.last_progress = None
        self.job = SimpleNamespace(id=directory.name)
        (directory / "job.json").write_text(json.dumps({"type": "budget_sweep", "status": "cancelled", "config": cfg}))
    def check(self):
        if self.stopped:
            raise Cancelled()
    def log(self, message): pass
    def progress(self, done, total, message=None): self.last_progress = (done, total)
    def write_json(self, name, value): (self.dir / name).write_text(json.dumps(value))
    def append_jsonl(self, name, value):
        with (self.dir / name).open("a") as f:
            f.write(json.dumps(value) + "\n")
        if name == "measurements.jsonl" and value["phase"] == self.stop_phase:
            self.stopped = True


@pytest.fixture
def toy(tmp_path, monkeypatch):
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).eval()
    for p in model.parameters(): p.requires_grad_(False)
    loaded = SimpleNamespace(model=model, device="cpu", name="toy")
    pts = [{"sample_id": i, "run_id": "p1", "model": "toy"} for i in range(8)]
    def objective(L, enc, pt, kind): return torch.sin(L.model(enc)).sum()
    def audio(path): return np.random.default_rng(int(path) + 10).standard_normal(4).astype(np.float32)
    monkeypatch.setattr(budget.M, "load_model", lambda name: loaded)
    monkeypatch.setattr(budget.M, "unload_unused", lambda L: 0)
    monkeypatch.setattr(budget.M, "encode", lambda L, a, grad=False: torch.tensor(a).reshape(1, 4))
    monkeypatch.setattr(budget, "point_objective", objective)
    monkeypatch.setattr(phase2, "point_objective", objective)
    monkeypatch.setattr(budget, "decision_points", lambda *args: list(pts))
    monkeypatch.setattr(budget_device, "decision_points", lambda *args: list(pts))
    monkeypatch.setattr(budget.store, "get", lambda sid: {"audio_path": str(sid)})
    monkeypatch.setattr(budget, "load_audio", audio)
    monkeypatch.setattr(budget_device, "load_audio", audio)
    monkeypatch.setattr(budget_device, "RUNS_DIR", tmp_path)
    cfg = {**budget.BUDGET_DEFAULTS, "param_filter": "all", "train_points": 3,
           "test_points": 2, "fractions": [0.03, 0.1, 0.3, 0.75]}
    return cfg, loaded


def assert_curves_close(a, b):
    assert a["train"] == b["train"] and a["test"] == b["test"]
    for x, y in zip(a["curve"], b["curve"]):
        assert x["k"] == y["k"]
        for key in x:
            if isinstance(x[key], dict):
                assert x[key]["n"] == y[key]["n"]
                assert x[key]["mean"] == pytest.approx(y[key]["mean"], rel=2e-4, abs=2e-5), (key, x, y)
            else:
                assert x[key] == pytest.approx(y[key], rel=2e-5, abs=1e-7)


def test_complete_sweep_matches_reference_and_restores_model(toy, tmp_path):
    cfg, L = toy
    before = [p.clone() for p in L.model.parameters()]
    reference = Context(tmp_path / "reference", cfg)
    budget.run_budget_reference(reference, cfg)
    actual = Context(tmp_path / "actual", cfg)
    budget_device.run_device_budget(actual, cfg)
    assert_curves_close(json.loads((reference.dir / "summary.json").read_text()),
                        json.loads((actual.dir / "summary.json").read_text()))
    assert actual.last_progress[0] == actual.last_progress[1]
    assert all(torch.equal(a, b) for a, b in zip(before, L.model.parameters()))


@pytest.mark.parametrize("stop_phase", ["train", "test"])
def test_checkpoint_resume_matches_fresh_and_preserves_control_rng(toy, tmp_path, stop_phase):
    cfg, L = toy
    before = [p.clone() for p in L.model.parameters()]
    fresh = Context(tmp_path / "fresh", cfg)
    budget_device.run_device_budget(fresh, cfg)
    partial = Context(tmp_path / "partial", cfg, stop_phase)
    with pytest.raises(Cancelled):
        budget_device.run_device_budget(partial, cfg)
    assert all(torch.equal(a, b) for a, b in zip(before, L.model.parameters()))
    cfg2 = {**cfg, "resume_run": "partial"}
    resumed = Context(tmp_path / "resumed", cfg2)
    result = budget_device.run_device_budget(resumed, cfg2)
    assert result["reused_checkpoint_records"] > 0
    assert_curves_close(json.loads((fresh.dir / "summary.json").read_text()),
                        json.loads((resumed.dir / "summary.json").read_text()))
    rows = [json.loads(line) for line in (resumed.dir / "measurements.jsonl").read_text().splitlines()]
    assert len({budget_device.Journal.key(r) for r in rows}) == len(rows)
    assert resumed.last_progress[0] == resumed.last_progress[1]
    assert all(torch.equal(a, b) for a, b in zip(before, L.model.parameters()))


def test_resume_rejects_changed_scientific_grid(toy, tmp_path):
    cfg, _ = toy
    partial = Context(tmp_path / "partial", cfg, "train")
    with pytest.raises(Cancelled):
        budget_device.run_device_budget(partial, cfg)
    changed = {**cfg, "resume_run": "partial", "fractions": [0.2, 0.3]}
    with pytest.raises(ValueError, match="mismatch"):
        budget_device.run_device_budget(Context(tmp_path / "changed", changed), changed)


def test_edit_is_undone_when_evaluation_raises(toy, monkeypatch):
    _, L = toy
    space = ParamSpace(L.model, "all")
    before = [p.clone() for p in space.params]
    idx = torch.arange(0, space.total, 2)
    support = (idx, space.sparse_bounds(idx))
    def fail(*args): raise RuntimeError("evaluation failed")
    monkeypatch.setattr(budget_device, "eval_d", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        budget_device.apply_eval(L, space, None, {}, "auto", support, torch.ones(len(idx)))
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


def test_endpoint_benchmark_checks_full_restoration(toy, tmp_path):
    cfg, L = toy
    before = [p.detach().clone() for p in L.model.parameters()]
    result = budget_device.benchmark(Context(tmp_path / "benchmark", cfg), cfg)
    assert result["passed"]
    assert result["peak_allocated_bytes"] is None
    assert [row["k"] for row in result["effects"]] == [min(result["ks"]), max(result["ks"])]
    assert all(torch.equal(a, b) for a, b in zip(before, L.model.parameters()))
