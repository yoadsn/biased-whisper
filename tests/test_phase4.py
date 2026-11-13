import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from bwhisper import phase4 as P
from bwhisper.jobs import Cancelled, Ctx, Job
from bwhisper.lora import Adapter


def tiny_model():
    return WhisperForConditionalGeneration(WhisperConfig(
        vocab_size=32, num_mel_bins=8, d_model=16, encoder_layers=2, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=16, max_target_positions=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=31, decoder_start_token_id=1,
        suppress_tokens=[], begin_suppress_tokens=[],
    )).eval()


@pytest.fixture
def cfg():
    return {**copy.deepcopy(P.DEFAULTS), "parallel": False}


@pytest.mark.parametrize("key,value", [("epochs", 0), ("epochs", 4), ("epochs", True),
    ("rank", 0), ("gradient_accumulation", -1), ("eval_limit", 0),
    ("learning_rate", float("nan")), ("dropout", 1), ("min_rest_match", -1),
    ("eval_every_steps", -1), ("max_grad_norm", 0)])
def test_invalid_config(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ValueError):
        P.validate(cfg)


def test_natural_cardinal_and_no_number_are_valid_teacher_targets(cfg):
    row = {"verdict": "no_digits", "baseline_text": "עשרים אנשים הגיעו.",
           "baseline_tokens": [3, 4], "baseline_ended": True}
    target, reason = P.training_target(row, cfg)
    assert reason == "accepted" and target["kind"] == "natural_cardinal"
    row["baseline_text"] = "שלום עולם."
    assert P.training_target(row, cfg)[0]["kind"] == "preservation"
    cfg["include_no_number"] = False
    assert P.training_target(row, cfg)[0] is None


def test_suppressed_target_and_filters(cfg):
    row = {"verdict": "cardinal", "baseline_text": "בשנת 2024.",
           "suppressed_text": "בשנת אלפיים עשרים וארבע.", "y_cardinal": [3, 4],
           "suppressed_ended": True, "diff": {"rest_match": .95}}
    assert P.training_target(row, cfg)[0]["kind"] == "suppressed_cardinal"
    for change, expected in [({"verdict": "degenerate"}, "degenerate"),
                             ({"verdict": "partial"}, "partial"),
                             ({"suppressed_ended": False}, "truncated"),
                             ({"diff": {"rest_match": .2}}, "rest_match"),
                             ({"suppressed_text": "בשנת 2024"}, "remaining_digits")]:
        assert P.training_target({**row, **change}, cfg) == (None, expected)
    # Optional parser never excludes unknowns unless explicitly requested.
    row["suppressed_text"] = "בשנת עשרים עשרים וארבע."
    assert P.training_target(row, cfg)[0]
    cfg["value_filter"] = True
    assert P.training_target(row, cfg) == (None, "value_mismatch_or_unknown")


def test_prevalence_includes_quality_warnings():
    metrics = P.prevalence([{"text": "יש 20 אנשים.", "ended": True},
                            {"text": "יש עשרים אנשים.", "ended": True},
                            {"text": "", "ended": False}])
    assert metrics["digit_clip_rate"] == 1/3
    assert metrics["digit_mention_share"] == .5
    assert metrics["truncated"] == metrics["empty"] == 1


def test_actual_cached_whisper_decode_works_with_lora():
    from bwhisper import model as M

    model = tiny_model()
    L = SimpleNamespace(model=model, device="cpu", sot_seq=[1, 3], eos=31, vocab=32,
                        digit_mask=torch.zeros(32, dtype=torch.bool), suppress_ids=[], suppress_begin=[])
    features = torch.randn(1, 8, 32)
    with torch.no_grad():
        enc = model.model.encoder(features).last_hidden_state
    expected = M.greedy(L, enc, L.sot_seq, max_new=4).tokens
    adapter = Adapter(model, rank=2)
    try:
        observed = M.greedy(L, enc, L.sot_seq, max_new=4).tokens
        assert observed == expected
        with adapter.teacher():
            assert M.greedy(L, enc, L.sot_seq, max_new=4).tokens == expected
    finally:
        adapter.restore()


def test_lora_frozen_teacher_save_reload_restore(tmp_path):
    model = tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = Adapter(model, rank=2, alpha=4., dropout=0., encoder_top_layers=1)
    assert any(n.startswith("model.encoder.layers.1") for n in adapter.sites)
    assert not any(n.startswith("model.encoder.layers.0") for n in adapter.sites)
    params = adapter.parameters()
    assert all(p.requires_grad for p in params)
    assert sum(p.requires_grad for p in model.parameters()) == len(params)
    layer = adapter.layers()[0]
    x = torch.randn(1, 3, layer.A.shape[1])
    expected = layer.base(x).detach()
    torch.testing.assert_close(layer(x), expected)  # zero initialization is identity
    with torch.no_grad():
        layer.B.fill_(.25)
    assert not torch.equal(layer(x), expected)
    model.train()
    with pytest.raises(RuntimeError):
        with adapter.teacher():
            assert not model.training
            torch.testing.assert_close(layer(x), expected)
            raise RuntimeError("teacher interrupted")
    assert model.training and layer.enabled
    adapter.save(tmp_path / "checkpoint", {"model": "tiny"})
    saved = layer.B.clone()
    with torch.no_grad():
        layer.B.zero_()
    adapter.load(tmp_path / "checkpoint")
    torch.testing.assert_close(layer.B, saved)
    adapter.restore()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_loss_excludes_forced_prefix_and_includes_eos():
    class Model:
        config = SimpleNamespace(max_target_positions=20)
        def __call__(self, **kwargs):
            assert kwargs["decoder_input_ids"].tolist() == [[1, 3, 4, 5]]
            # [SOT, language, target 4, target 5, EOS=2]
            logits = torch.zeros(1, 4, 8, requires_grad=True)
            return SimpleNamespace(logits=logits)
    L = SimpleNamespace(sot_seq=[1, 3], eos=2, device="cpu", vocab=8, model=Model())
    loss = P.teacher_loss(L, torch.zeros(1), [4, 5])
    assert float(loss.detach()) == pytest.approx(math_log(8))
    loss.backward()


def math_log(value):
    import math
    return math.log(value)


@pytest.fixture
def cohort(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(P, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    directory = P.RUNS_DIR / "setup_scale-test"
    directory.mkdir(parents=True)
    (directory / "cohorts.json").write_text(json.dumps({"discovery": [1, 2], "evaluation": [3, 4, 5]}))
    samples = {i: {"id": i, "source": "test", "meta": {"recording": str(i)},
                   "audio_path": str(i), "text": "unused"} for i in range(1, 6)}
    samples[4]["meta"]["recording"] = "3"  # monitor keeps only one clip per recording
    monkeypatch.setattr(P.store, "query", lambda **kw: list(samples.values()))
    monkeypatch.setattr(P.store, "get", samples.get)
    return samples


def test_split_preserved_and_confirmation_excluded(cohort, cfg):
    campaign = P.DATA / "controller_campaigns" / "old"
    campaign.mkdir(parents=True)
    (campaign / "campaign.json").write_text(json.dumps({"confirmation_ids": [5]}))
    _, manifest = P.select_cohorts(cfg)
    assert set(manifest["train_ids"]) == {1, 2}
    assert len(manifest["eval_ids"]) == 1
    assert manifest["eval_ids"][0] in (3, 4)
    assert manifest["protected_recordings"] == 1
    cohort[3]["meta"]["recording"] = "1"
    with pytest.raises(ValueError, match="leakage"):
        P.select_cohorts(cfg)


def test_monitor_api_is_bounded_and_tolerates_partial_append(tmp_path, monkeypatch):
    from bwhisper import api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)
    run = tmp_path / "phase4-test"
    run.mkdir()
    (run / "job.json").write_text(json.dumps({"type": "phase4"}))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"step": i}) for i in range(600)) + '\n{"step":')
    (run / "monitor.json").write_text(json.dumps({"stage": "training"}))
    client = TestClient(api.app)
    response = client.get("/api/phase4/phase4-test/monitor")
    assert response.status_code == 200
    data = response.json()
    assert data["monitor"]["stage"] == "training"
    assert len(data["metrics"]) == 499 and data["metrics"][-1]["step"] == 599
    assert data["evaluations"] == data["checkpoints"] == []
    (run / "job.json").write_text(json.dumps({"type": "phase1"}))
    assert client.get("/api/phase4/phase4-test/monitor").status_code == 400


@pytest.mark.parametrize("cancel", [False, True])
def test_tiny_whisper_online_training_and_cancellation(cohort, cfg, tmp_path, monkeypatch, cancel):
    model = tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    L = SimpleNamespace(model=model, device="cpu", name="tiny-random", sot_seq=[1, 3], eos=31, vocab=32)
    monkeypatch.setattr(P.M, "load_model", lambda *a, **kw: L)
    monkeypatch.setattr(P.M, "unload_unused", lambda *a: None)
    monkeypatch.setattr(P, "StyleTokens", lambda *a: None)
    monkeypatch.setattr(P, "load_audio", lambda p: np.zeros(100, dtype=np.float32))
    monkeypatch.setattr(P.M, "features", lambda *a: torch.ones(1, 8, 32))
    teachers = []
    def analyse(*args):
        from bwhisper.lora import LoRALinear
        assert all(not m.enabled for m in model.modules() if isinstance(m, LoRALinear))
        assert not model.training
        teachers.append(1)
        return {"verdict": "no_digits", "baseline_text": "עשרים אנשים", "baseline_tokens": [6, 7], "baseline_ended": True}
    monkeypatch.setattr(P, "analyse_sample", analyse)
    monkeypatch.setattr(P.M, "transcribe", lambda *a: ("עשרים אנשים", SimpleNamespace(ended=True)))
    cfg.update(epochs=2, rank=2, gradient_accumulation=3, max_new_tokens=10,
               eval_limit=1, bf16=False, dropout=0., learning_rate=.01, warmup_steps=0)
    directory = tmp_path / "training"
    directory.mkdir()
    job = Job("training", "phase4", cfg, directory)
    ctx = Ctx(job)
    if cancel:
        check = ctx.check
        def stop_after_step():
            monitor = directory / "monitor.json"
            if monitor.exists() and json.loads(monitor.read_text())["step"] >= 1:
                job.cancel_flag.set()
            check()
        ctx.check = stop_after_step
        with pytest.raises(Cancelled):
            P.run_phase4(ctx, cfg)
        assert (directory / "latest/adapter.safetensors").exists()
    else:
        result = P.run_phase4(ctx, cfg)
        assert result["updates"] == 2  # remainder accumulation flushed at each epoch
        assert len(teachers) == 2  # one fixed teacher target per sample, reused in epoch 2
        assert (directory / "epoch-02/adapter.safetensors").exists()
        from safetensors.torch import load_file
        assert any(v.abs().sum() > 0 for k, v in load_file(str(directory / "latest/adapter.safetensors")).items() if k.endswith(".B"))
        monitor = json.loads((directory / "monitor.json").read_text())
        assert monitor["stage"] == "complete" and monitor["evaluation"]["suppression_at_evaluation"] is False
        # Saved adapter is independently reloadable by the evaluation job.
        (directory / "manifest.json").write_text(json.dumps({"eval_ids": [3]}))
        job.config = cfg
        job.status = "done"
        job.save()
        monkeypatch.setattr(P, "RUNS_DIR", tmp_path)
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        evaluation = P.evaluate_checkpoint(Ctx(Job("evaluation", "phase4_evaluate", {}, eval_dir)),
                                          {"training_run": "training", "checkpoint": "epoch-02", "device": "cpu"})
        assert evaluation["adapter"]["n"] == 1
        assert evaluation["suppression_at_evaluation"] is False
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in model.parameters())
