import json
from types import SimpleNamespace

import numpy as np
import pytest

from bwhisper import controller as C


class Context:
    def __init__(self, path):
        path.mkdir()
        self.dir = path
        self.job = SimpleNamespace(id=path.name, config={}, save=lambda: None)

    def write_json(self, name, value):
        (self.dir / name).write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CAMPAIGNS", tmp_path / "campaigns")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    d = C.CAMPAIGNS / "test"
    d.mkdir(parents=True)
    (d / "campaign.json").write_text(json.dumps({"manual_review_cap": 10, "calibration_ids": [1, 2],
                                                "confirmation_ids": []}))
    run = tmp_path / "phase3"
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"type": "phase3", "status": "done"}))
    rows = [{"cond": cond, "strength": strength, "sample_id": sid, "text": text}
            for sid in [1, 2] for cond, strength, text in [("baseline", 0, "מספר 24."),
                                                         ("heads_to_cardinal", 10, "מספר עשרים וארבע.")]]
    (run / "outputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(C.store, "get", lambda sid: {"source": "test", "text": "מספר 24."})
    return tmp_path, {"campaign": "test", "phase3_run": "phase3", "cond": "heads_to_cardinal",
                      "strength": 10, "sample_ids": [1, 2]}


def test_review_reservation_is_charged_before_labeling_and_cannot_repeat(campaign):
    path, cfg = campaign
    result = C.review_batch(Context(path / "packet"), cfg)
    assert result["spent"] == 2
    with pytest.raises(ValueError, match="already reserved"):
        C.review_batch(Context(path / "duplicate"), cfg)
    labels = [{"sample_id": sid, "value_ok": True, "style_ok": True, "text_ok": True,
               "uncertain": sid == 2, "note": "24 preserved; uncertainty counts as failure."} for sid in [1, 2]]
    result = C.review_submit(Context(path / "labels"), {"campaign": "test", "packet_run": "packet", "labels": labels})
    assert result["successes"] == 1
    with pytest.raises(ValueError, match="already labeled"):
        C.review_submit(Context(path / "overwrite"), {"campaign": "test", "packet_run": "packet", "labels": labels})


def test_budget_cannot_be_raised_above_100(campaign):
    path, cfg = campaign
    d = C.CAMPAIGNS / "test"
    manifest = json.loads((d / "campaign.json").read_text())
    manifest["manual_review_cap"] = 1000
    (d / "campaign.json").write_text(json.dumps(manifest))
    (d / "reviews.json").write_text(json.dumps({"spent": 99, "packets": {}}))
    with pytest.raises(ValueError, match="cap reached"):
        C.review_batch(Context(path / "over"), cfg)


def test_multi_sentence_packet_limit(campaign, monkeypatch):
    path, cfg = campaign
    monkeypatch.setattr(C.store, "get", lambda sid: {"source": "test", "text": "א. " * 6})
    with pytest.raises(ValueError, match="12 sentences"):
        C.review_batch(Context(path / "too-many"), cfg)
    assert C.sentence_cost("3.14 שווה.", "א. ב.") == 2


def test_binomial_interval():
    assert C.exact_interval(0, 10)[0] == 0
    assert C.exact_interval(10, 10)[1] == 1
    assert C.exact_interval(10, 10)[0] < 0.70  # ten successes do not establish high reliability
    with pytest.raises(ValueError):
        C.exact_interval(2, 1)


@pytest.mark.parametrize("name", ["../bad", "..", "", "/absolute"])
def test_artifact_path_validation(name, tmp_path):
    with pytest.raises(ValueError):
        C.child(tmp_path, name)


def test_candidate_filter_and_composition(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    src = tmp_path / "source"
    src.mkdir()
    (src / "job.json").write_text(json.dumps({"status": "done", "config": {"model": "test", "param_filter": "all"}}))
    (src / "space.json").write_text(json.dumps({"sizes": [2, 2], "names": ["decoder.weight", "encoder.weight"]}))
    np.savez(src / "candidates_heads.npz", idx=np.arange(4), mean_all=np.array([1, 2, 3, 4], np.float32))
    ctx = Context(tmp_path / "made")
    C.build_candidates(ctx, {"source_run": "source", "variants": [{"name": "decoder_to_cardinal", "parts": [
        {"edit": "heads", "include_regex": "decoder", "scale": 2},
        {"edit": "heads", "include_regex": "decoder", "scale": -0.5}]}]})
    with np.load(ctx.dir / "candidates_decoder_to_cardinal.npz") as z:
        assert z["idx"].tolist() == [0, 1]
        assert z["mean_all"].tolist() == [1.5, 3]
    assert ctx.job.config["model"] == "test"


def test_nested_composition_preserves_direct_source_and_own_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    source = tmp_path / "parent"
    source.mkdir()
    (source / "job.json").write_text(json.dumps({"status": "done", "config": {
        "model": "test", "source_run": "grandparent", "variants": [{"name": "old"}]}}))
    (source / "space.json").write_text(json.dumps({"sizes": [2], "names": ["decoder.weight"]}))
    np.savez(source / "candidates_heads.npz", idx=np.arange(2), mean_all=np.ones(2, np.float32))
    ctx = Context(tmp_path / "child")
    variants = [{"name": "new", "parts": [{"edit": "heads"}]}]
    ctx.job.config.update(source_run="parent", variants=variants)
    result = C.build_candidates(ctx, ctx.job.config)
    assert result["source_run"] == ctx.job.config["source_run"] == "parent"
    assert ctx.job.config["variants"] == variants


def test_saved_attribution_refit_selects_decoder_only(tmp_path, monkeypatch):
    from bwhisper import model, paramspace, phase2n

    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    source = tmp_path / "original"
    source.mkdir()
    pcfg = {"model": "test", "param_filter": "all", "include_regex": "", "exclude_regex": "",
            "gain": .1, "fdr": .05}
    (source / "job.json").write_text(json.dumps({"type": "phase2n", "status": "done", "config": pcfg}))
    layout = [{"name": "dec.mlp", "tower": "decoder", "kind": "mlp", "offset": 0, "n": 2},
              {"name": "dec.cross", "tower": "decoder", "kind": "cross", "offset": 2, "n": 1},
              {"name": "enc.mlp", "tower": "encoder", "kind": "mlp", "offset": 3, "n": 1}]
    (source / "components.json").write_text(json.dumps(layout))
    (source / "space.json").write_text(json.dumps({"signature": "test"}))
    values = np.array([[1, -2, 3, 100], [1.1, -2.1, 3.1, 101], [.9, -1.9, 2.9, 99]], np.float32)
    np.savez(source / "attributions.npz", full=values, decision=values * 2,
             keys=np.array(["p1__1", "p1__2", "p1__3"]))
    loaded = SimpleNamespace(name="test", model=SimpleNamespace(config=SimpleNamespace(
        d_model=4, decoder_attention_heads=2, encoder_attention_heads=2)))
    monkeypatch.setattr(model, "load_model", lambda *a: loaded)
    monkeypatch.setattr(model, "unload_unused", lambda *a: None)
    monkeypatch.setattr(paramspace, "ParamSpace", lambda *a: SimpleNamespace(signature="test"))
    seen = []
    def translate(L, space, layout, selected, gain, heads_dim):
        assert all(site["tower"] == "decoder" for site, _, _ in selected)
        seen.append(selected)
        return np.arange(len(selected)), np.ones(len(selected), np.float32)
    monkeypatch.setattr(phase2n, "_weight_edits", translate)
    ctx = Context(tmp_path / "refit")
    ctx.check = lambda: None
    ctx.log = lambda message: None
    result = C.refit_candidates(ctx, {"source_run": "original", "capture_sample_ids": [1, 2, 3]})
    assert len(result["variants"]) == len(seen) == 8
    with np.load(ctx.dir / "candidates_dec_decision_heads2_to_digits.npz") as z:
        assert np.all(z["mean_all"] == -1)
    assert ctx.job.config["model"] == "test"
    assert ctx.job.config["sample_ids"] == [1, 2, 3]
    assert ctx.job.config["max_points"] == 3


def test_residual_candidates_reproduce_phase2n_scaling(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path)
    source = tmp_path / "original"
    source.mkdir()
    (source / "job.json").write_text(json.dumps({"type": "phase2n", "status": "done",
                                               "config": {"model": "test", "steer_target": 1.}}))
    (source / "space.json").write_text(json.dumps({"sizes": [4, 4], "names": [
        "model.decoder.layers.0.fc2.bias", "model.decoder.layers.1.fc2.bias"]}))
    gradients = np.arange(24, dtype=np.float32).reshape(3, 2, 4) + 1
    np.savez(source / "attributions.npz", resid_grad=gradients)
    ctx = Context(tmp_path / "residual")
    result = C.residual_candidates(ctx, {"source_run": "original"})
    assert result["variants"] == ["steer_L0", "steer_L1"]
    unit = gradients / np.linalg.norm(gradients, axis=2, keepdims=True)
    v = unit.mean(0)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    expected = -v / np.einsum("nld,ld->nl", gradients, v).mean(0)[:, None]
    for layer in range(2):
        with np.load(ctx.dir / f"candidates_steer_L{layer}_to_cardinal.npz") as z:
            np.testing.assert_allclose(z["mean_all"], expected[layer], atol=1e-7)
        with np.load(ctx.dir / f"candidates_steer_L{layer}_to_digits.npz") as z:
            np.testing.assert_allclose(z["mean_all"], -expected[layer], atol=1e-7)


def test_confirmation_cannot_be_opened_without_freezing(campaign):
    path, cfg = campaign
    d = C.CAMPAIGNS / "test"
    (d / "campaign.json").write_text(json.dumps({"manual_review_cap": 100,
                                                "calibration_ids": [], "confirmation_ids": [1, 2]}))
    with pytest.raises(ValueError, match="freeze"):
        C.review_batch(Context(path / "confirmation"), cfg)


def test_labels_cannot_claim_repetition_is_success(campaign):
    path, cfg = campaign
    C.review_batch(Context(path / "packet"), cfg)
    p = path / "packet" / "packet.json"
    packet = json.loads(p.read_text())
    packet["rows"][0]["output"] = "אחד אחד אחד אחד אחד"
    p.write_text(json.dumps(packet))
    labels = [{"sample_id": sid, "value_ok": True, "style_ok": True, "text_ok": True,
               "uncertain": False, "note": "A mistakenly optimistic reviewer label."} for sid in [1, 2]]
    result = C.review_submit(Context(path / "labels"), {"campaign": "test", "packet_run": "packet", "labels": labels})
    assert result["successes"] == 1


def test_freeze_binds_confirmation_to_candidate_hash(campaign):
    path, cfg = campaign
    d = C.CAMPAIGNS / "test"
    (d / "campaign.json").write_text(json.dumps({"manual_review_cap": 100,
                                                "calibration_ids": [1, 2], "confirmation_ids": [3]}))
    run = path / "phase3"
    (run / "job.json").write_text(json.dumps({"type": "phase3", "status": "done", "config": {
        "phase2_run": "p2", "stat": "mean_all", "candidate_limit": None}}))
    p2 = path / "p2"
    p2.mkdir()
    candidate = p2 / "candidates_heads_to_cardinal.npz"
    np.savez(candidate, idx=np.array([0]), mean_all=np.array([1.]))
    with (run / "outputs.jsonl").open("a") as f:
        for cond, strength, text in [("baseline", 0, "מספר 24."), ("heads_to_cardinal", 10, "מספר עשרים וארבע.")]:
            f.write("\n" + json.dumps({"cond": cond, "strength": strength, "sample_id": 3, "text": text}))
    C.review_batch(Context(path / "packet"), cfg)
    labels = [{"sample_id": sid, "value_ok": True, "style_ok": True, "text_ok": True,
               "uncertain": False, "note": "Value and text preserved."} for sid in [1, 2]]
    C.review_submit(Context(path / "labels"), {"campaign": "test", "packet_run": "packet", "labels": labels})
    selection = C.freeze_controller(Context(path / "freeze"), {"campaign": "test", "packet_run": "packet"})
    assert selection["confirmation_ids"] == [3]
    with pytest.raises(ValueError, match="already frozen"):
        C.freeze_controller(Context(path / "freeze-again"), {"campaign": "test", "packet_run": "packet"})
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="modified"):
        C.review_batch(Context(path / "confirmation"), {**cfg, "sample_ids": [3]})
