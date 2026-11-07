import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from bwhisper import interactive as T, model as M, phase3_inference as I
from bwhisper.audio import write_pcm16_wav
from bwhisper.jobs import Cancelled
from test_phase3_inference import Context, experiment, loaded  # noqa: F401 (shared fixtures)


@pytest.fixture
def playground(experiment, loaded, tmp_path, monkeypatch):
    cfg, space = experiment
    phase3 = tmp_path / "phase3-source"
    phase3.mkdir()
    cfg["model"] = loaded.name
    (phase3 / "job.json").write_text(json.dumps({"type": "phase3", "status": "done", "config": cfg}))
    monkeypatch.setattr(T, "RUNS_DIR", tmp_path)
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(T, "TEST_AUDIO_DIR", uploads)
    aid = "a" * 32
    directory = uploads / aid
    write_pcm16_wav(directory / "audio.wav", np.zeros(1600, dtype=np.int16))
    (directory / "audio.json").write_text(json.dumps({"id": aid, "duration": 0.1, "filename": "test.wav"}))
    monkeypatch.setattr(I, "load_audio", lambda path: np.zeros(64, dtype=np.float32))
    loads = []
    def load(name, device=None):
        loads.append((name, device))
        return loaded
    monkeypatch.setattr(M, "load_model", load)
    return {**T.TEST_DEFAULTS, "phase3_run": phase3.name, "audio_id": aid}, space, loads


def test_configs_are_independent_and_one_model_serves_every_row(playground, loaded, tmp_path, monkeypatch):
    cfg, space, loads = playground
    cfg["edit_configs"] = [{"edit": "to_cardinal", "strength": value} for value in [2, -1, 0, 2]]
    before = space.snapshot()
    seen = []
    original = T.greedy_tokens
    def capture(*args, **kwargs):
        seen.append(space.snapshot())
        return original(*args, **kwargs)
    monkeypatch.setattr(T, "greedy_tokens", capture)
    result = T.transcribe(Context(tmp_path / "test"), cfg)
    assert len(loads) == 1
    assert result["model_kept_loaded"]
    assert len(result["results"]) == 5
    assert result["results"][0]["edit"] is None
    assert result["results"][0]["text"] == result["results"][3]["text"]
    assert result["results"][1]["text"] == result["results"][4]["text"]
    assert all(torch.equal(a, b) for a, b in zip(before, seen[0]))
    assert all(torch.equal(a, b) for a, b in zip(before, seen[3]))
    assert all(torch.equal(a, b) for a, b in zip(seen[1], seen[4]))
    assert any(not torch.equal(a, b) for a, b in zip(before, seen[1]))
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))
    # A subsequent comparison also starts from the untouched cached model.
    again = T.transcribe(Context(tmp_path / "again"), {**cfg, "edit_configs": []})
    assert again["results"][0]["text"] == result["results"][0]["text"]
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


@pytest.mark.parametrize("failure", [RuntimeError, Cancelled])
def test_test_failure_or_cancel_restores_weights(playground, tmp_path, monkeypatch, failure):
    cfg, space, _ = playground
    cfg["edit_configs"] = [{"edit": "to_digits", "strength": 2}]
    before, calls = space.snapshot(), []
    original = T.greedy_tokens
    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise failure("interrupted while edit is live")
        return original(*args, **kwargs)
    monkeypatch.setattr(T, "greedy_tokens", fail)
    with pytest.raises(failure):
        T.transcribe(Context(tmp_path / "failed"), cfg)
    assert all(torch.equal(a, b) for a, b in zip(before, space.params))


@pytest.mark.parametrize("config", [None, [{}], [{"edit": "../escape", "strength": 1}],
                                    [{"edit": "to_digits", "strength": "1"}],
                                    [{"edit": "to_digits", "strength": float("nan")}],
                                    [{"edit": "to_digits", "strength": True}]])
def test_invalid_configs_rejected_before_loading_model(playground, tmp_path, config):
    cfg, _, loads = playground
    with pytest.raises(ValueError):
        T.transcribe(Context(tmp_path / "invalid"), {**cfg, "edit_configs": config})
    assert not loads


def test_model_provenance_never_falls_back_to_current_active_model(playground, tmp_path):
    cfg, _, _ = playground
    path = tmp_path / cfg["phase3_run"] / "job.json"
    job = json.loads(path.read_text())
    job["config"]["model"] = None
    path.write_text(json.dumps(job))
    # Phase 2 config still pins the model.
    assert T.source_spec(cfg["phase3_run"])["model"] == "random-whisper"
    p2path = tmp_path / "p2" / "job.json"
    job2 = json.loads(p2path.read_text())
    job2["config"]["model"] = None
    p2path.write_text(json.dumps(job2))
    with pytest.raises(ValueError, match="model provenance"):
        T.source_spec(cfg["phase3_run"])


@pytest.mark.parametrize("value", ["../elsewhere", "..", "", None])
def test_source_paths_cannot_escape_run_directory(value):
    with pytest.raises(ValueError):
        T.source_spec(value)


@pytest.mark.parametrize("duration,passes", [(0, False), (0.1, True), (30, True), (30.01, False), (31, False)])
def test_audio_duration_is_decoded_and_enforced(tmp_path, monkeypatch, duration, passes):
    monkeypatch.setattr(T, "TEST_AUDIO_DIR", tmp_path / "uploads")
    path = tmp_path / "source.wav"
    write_pcm16_wav(path, np.zeros(round(duration * 16000), dtype=np.int16))
    if passes:
        result = T.normalize_upload(path, "source.wav")
        assert result["duration"] == pytest.approx(duration)
        assert T.audio_dir(result["id"]).is_dir()
    else:
        with pytest.raises(ValueError):
            T.normalize_upload(path, "source.wav")


def test_invalid_audio_has_safe_error(tmp_path):
    path = tmp_path / "secret-filename.audio"
    path.write_text("not an audio file")
    with pytest.raises(ValueError, match="cannot decode") as error:
        T.normalize_upload(path, "source")
    assert "secret-filename" not in str(error.value)


@pytest.mark.parametrize("extension,codec", [("mp3", "libmp3lame"), ("webm", "libopus"), ("m4a", "aac")])
def test_common_compressed_upload_formats(tmp_path, monkeypatch, extension, codec):
    monkeypatch.setattr(T, "TEST_AUDIO_DIR", tmp_path / "uploads")
    source, encoded = tmp_path / "source.wav", tmp_path / f"source.{extension}"
    write_pcm16_wav(source, np.zeros(16000, dtype=np.int16))
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-c:a", codec, str(encoded)],
                   capture_output=True, check=True)
    result = T.normalize_upload(encoded, encoded.name)
    assert 0.9 <= result["duration"] <= 1.1


def test_api_upload_metadata_submission_and_limits(playground, tmp_path, monkeypatch):
    from bwhisper import api

    cfg, _, _ = playground
    queued = []
    def submit(kind, config):
        queued.append((kind, config))
        return SimpleNamespace(to_dict=lambda: {"id": "test-queued", "status": "queued"})
    monkeypatch.setattr(api, "manager", lambda: SimpleNamespace(submit=submit))
    path = tmp_path / "clip.wav"
    write_pcm16_wav(path, np.zeros(1600, dtype=np.int16))
    with TestClient(api.app) as client:
        source = client.get("/api/test/runs/" + cfg["phase3_run"])
        assert source.status_code == 200
        assert "p2_config" not in source.json()
        response = client.post("/api/test/audio?filename=clip.wav", content=path.read_bytes(), headers={"Content-Type": "audio/wav"})
        assert response.status_code == 200, response.text
        audio = response.json()
        assert client.get(audio["url"]).status_code == 200
        assert client.get(audio["url"] + "/info").json()["duration"] == 0.1
        response = client.post("/api/test/transcribe", json={"phase3_run": cfg["phase3_run"],
                               "audio_id": audio["id"], "edit_configs": [{"edit": "to_cardinal", "strength": 2}]})
        assert response.status_code == 200
        assert queued[0][0] == "test_transcribe"
        assert queued[0][1]["edit_configs"][0]["strength"] == 2
        monkeypatch.setattr(T, "MAX_UPLOAD_BYTES", 100)
        assert client.post("/api/test/audio", content=b"x" * 101).status_code == 413
        assert client.post("/api/test/audio", content=b"").status_code == 400
        assert client.get("/api/test/audio/not-an-id").status_code == 404
        assert client.post("/api/test/transcribe", json={"phase3_run": "../escape"}).status_code == 400
