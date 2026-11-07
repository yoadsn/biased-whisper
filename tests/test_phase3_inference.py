import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from bwhisper import model as M, phase3, phase3_device as P, phase3_inference as I
from bwhisper.config import PHASE3_DEFAULTS
from bwhisper.jobs import Cancelled
from bwhisper.paramspace import ParamSpace
from bwhisper.phase2 import point_objective


@pytest.fixture
def loaded():
    torch.manual_seed(15)
    config = WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=16,
                           encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                           decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                           max_source_positions=8, max_target_positions=64,
                           pad_token_id=12, bos_token_id=1, eos_token_id=12,
                           decoder_start_token_id=1, suppress_tokens=[4], begin_suppress_tokens=[5])
    model = WhisperForConditionalGeneration(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[6:8] = True
    def feature_extractor(audio, **kwargs):
        return SimpleNamespace(input_features=torch.tensor(audio).reshape(1, 4, 16))
    return SimpleNamespace(model=model, device="cpu", vocab=32, sot_seq=[1, 2, 3],
                           eos=12, suppress_ids=[4], suppress_begin=[5], digit_mask=mask,
                           feature_extractor=feature_extractor, name="random-whisper",
                           decode=lambda ids: " ".join(map(str, ids)))


@pytest.mark.parametrize("max_new", [0, 1, 7])
def test_batched_greedy_matches_individual_cached_decoder(loaded, max_new):
    enc = torch.randn(3, 8, 16)
    ref = [M.greedy(loaded, e[None], list(loaded.sot_seq), max_new=max_new).tokens for e in enc]
    assert I.greedy_tokens(loaded, enc, max_new) == ref


def test_mixed_eos_and_suppression():
    # Each row has a prescribed trajectory: EOS immediately, after a token, or
    # never before max_new. Highest illegal logits must always be suppressed.
    def decoder(input_ids, encoder_hidden_states, past_key_values=None, **kwargs):
        step = 0 if past_key_values is None else past_key_values
        hidden = torch.zeros(len(input_ids), 1, 10)
        for row, value in enumerate(encoder_hidden_states[:, 0, 0].tolist()):
            hidden[row, 0, 7] = 10 if step >= value else -10  # EOS
            hidden[row, 0, 4] = 4  # legal token
            hidden[row, 0, 5] = 6 if step == 0 else 0  # begin suppression
            hidden[row, 0, [1, 2, 3, 6, 8, 9]] = 100  # forbidden
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=step + 1)
    L = SimpleNamespace(device="cpu", vocab=10, sot_seq=[1, 2, 3], suppress_ids=[6],
                        suppress_begin=[5], eos=7,
                        model=SimpleNamespace(model=SimpleNamespace(decoder=decoder), proj_out=lambda x: x))
    enc = torch.tensor([0, 1, 20]).reshape(3, 1, 1)
    assert I.greedy_tokens(L, enc, 3) == [[], [4], [4, 4, 4]]
    calls = []
    def cancel():
        calls.append(1)
        raise Cancelled()
    with pytest.raises(Cancelled):
        I.greedy_tokens(L, enc, 3, cancel)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["auto", "seq_contrast", "mass", "pair", "mass_vs_card"])
def test_probe_padding_and_span_reduction_match_reference(loaded, kind):
    enc = torch.randn(3, 8, 16)
    pts = [dict(y_digit=[4, 6, 8][:n], y_cardinal=[5, 7, 9, 10][:n + 1],
                div_digit=[0, n], div_cardinal=[1, n + 1], prefix_ids=[1, 2, 3, 4],
                digit_token={"id": 6}, cardinal_token={"id": 7}) for n in [1, 2, 3]]
    with torch.inference_mode():
        expected = [float(point_objective(loaded, enc[i:i + 1], pt, kind)) for i, pt in enumerate(pts)]
    np.testing.assert_allclose(I.probe_values(loaded, enc, pts, kind), expected, atol=2e-5, rtol=2e-5)


class Context:
    def __init__(self, path):
        path.mkdir()
        self.dir = path
        self.last_progress = None
    def check(self): pass
    def log(self, text): pass
    def progress(self, done, total, message=None): self.last_progress = (done, total)
    def write_json(self, name, data): (self.dir / name).write_text(json.dumps(data))
    def append_jsonl(self, name, data):
        with (self.dir / name).open("a") as f:
            f.write(json.dumps(data) + "\n")


@pytest.fixture
def experiment(loaded, tmp_path, monkeypatch):
    pts = [dict(sample_id=i, y_digit=[4, 6, 8][:i + 1], y_cardinal=[5, 7, 9, 10][:i + 2],
                div_digit=[0, i + 1], div_cardinal=[1, i + 2], prefix_ids=[1, 2, 3, 4],
                digit_token={"id": 6}, cardinal_token={"id": 7}) for i in range(3)]
    rows = [{"id": i, "audio_path": str(i)} for i in range(6)]
    def audio(path):
        return np.random.default_rng(int(path)).standard_normal(64).astype(np.float32)
    monkeypatch.setattr(I, "load_audio", audio)
    monkeypatch.setattr(P, "load_audio", audio)
    monkeypatch.setattr(phase3, "load_audio", audio)
    monkeypatch.setattr(M, "load_model", lambda *args: loaded)
    monkeypatch.setattr(M, "unload_unused", lambda *args: 0)
    monkeypatch.setattr(P, "decision_points", lambda *args: list(pts))
    monkeypatch.setattr(phase3, "decision_points", lambda *args: list(pts))
    monkeypatch.setattr(P.store, "query", lambda **kwargs: list(rows))
    monkeypatch.setattr(P.store, "get", lambda sid: rows[sid])
    monkeypatch.setattr(P, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(phase3, "RUNS_DIR", tmp_path)
    space = ParamSpace(loaded.model, "all_no_embed")
    directory = tmp_path / "p2"
    directory.mkdir()
    cfg2 = dict(model="random-whisper", param_filter="all_no_embed", include_regex="", exclude_regex="",
                phase1_runs=["p1"], verdicts=["cardinal"], max_points=3, objective="auto")
    (directory / "job.json").write_text(json.dumps({"config": cfg2}))
    (directory / "space.json").write_text(json.dumps({"signature": space.signature}))
    # Deliberately edit encoder and decoder weights with a large enough signal
    # that reusing stale baseline encoder outputs fails the metric comparison.
    idx = np.arange(space.total, dtype=np.int64)
    vals = np.random.default_rng(8).normal(0, 0.03, len(idx)).astype(np.float32)
    for name in ["to_digits", "to_cardinal"]:
        np.savez(directory / f"candidates_{name}.npz", idx=idx, mean_all=vals)
    cfg = {**PHASE3_DEFAULTS, "phase2_run": "p2", "batch_size": 2, "max_new_tokens": 7,
           "strengths": [2.0], "eval_limit": 3}
    return cfg, space


def test_full_conditions_match_reference_with_encoder_edits(experiment, loaded, tmp_path):
    cfg, space = experiment
    before = space.snapshot()
    ref = Context(tmp_path / "ref")
    actual = Context(tmp_path / "batched")
    a = phase3.run_phase3_reference(ref, cfg)
    b = P.run_batched(actual, cfg)
    for expected, observed in zip(a["table"], b["table"]):
        assert expected.keys() == observed.keys()
        for key in expected:
            if isinstance(expected[key], float):
                assert observed[key] == pytest.approx(expected[key], rel=2e-4, abs=2e-4), (key, expected, observed)
            else:
                assert observed[key] == expected[key]
    assert (ref.dir / "outputs.jsonl").read_text() == (actual.dir / "outputs.jsonl").read_text()
    assert actual.last_progress[0] == actual.last_progress[1]
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


@pytest.mark.parametrize("failure", [RuntimeError, Cancelled])
def test_mid_edit_failures_restore_encoder_weights(experiment, loaded, tmp_path, monkeypatch, failure):
    cfg, space = experiment
    before = space.snapshot()
    original = I.greedy_tokens
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:  # baseline takes two batches, then an edit is live
            raise failure("interrupted during edited decode")
        return original(*args, **kwargs)
    monkeypatch.setattr(P, "greedy_tokens", fail)
    with pytest.raises(failure):
        P.run_batched(Context(tmp_path / "failed"), cfg)
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


def test_benchmark_stays_on_discovery_and_checks_edited_parity(experiment, tmp_path):
    cfg, space = experiment
    before = space.snapshot()
    cfg.update(benchmark_samples=3, benchmark_batch_sizes=[1, 2])
    ctx = Context(tmp_path / "benchmark")
    result = P.benchmark(ctx, cfg)
    assert result["passed"]
    summary = json.loads((ctx.dir / "benchmark.json").read_text())
    assert summary["samples"] == [0, 1, 2]
    assert summary["cohort"] == "phase2_discovery_only"
    assert all(summary["checks"].values())
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


def test_features_are_reused_but_encoder_is_recomputed(loaded, monkeypatch, tmp_path):
    monkeypatch.setattr(I, "load_audio", lambda path: np.zeros(64, dtype=np.float32))
    ctx = Context(tmp_path / "features")
    bank = I.FeatureBank(loaded, [{"id": 0, "audio_path": "0"}], ctx)
    first = bank.encode(loaded, [0])
    space = ParamSpace(loaded.model, "encoder")
    idx = np.arange(space.total, dtype=np.int64)
    undo = space.add_sparse(idx, np.full(space.total, 0.1, dtype=np.float32))
    try:
        changed = bank.encode(loaded, [0])
    finally:
        space.undo_sparse(undo)
    assert not torch.equal(first, changed)
    torch.testing.assert_close(first, bank.encode(loaded, [0]), rtol=0, atol=0)
    assert list(bank.values) == [0]
    assert bank.values[0].device.type == "cpu"
