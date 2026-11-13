import pytest
from pathlib import Path
import sys
import json
import subprocess
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
from contextlib import contextmanager

import torch

from fastapi.testclient import TestClient

from deploy.app import Runtime, app, decode_audio, load_presets, selection
from deploy.select_presets import select


@pytest.mark.parametrize("position,expected", [
    (-100, ("digits", 1.0)), (-25, ("digits", 0.25)),
    (0, (None, 0.0)), (40, ("cardinals", 0.4)), (100, ("cardinals", 1.0)),
])
def test_selection(position, expected):
    assert selection(position) == expected


@pytest.mark.parametrize("position", [-101, 101, 0.5, True, None])
def test_selection_rejects_invalid(position):
    with pytest.raises(ValueError):
        selection(position)


def test_runtime_switches_exclusively_and_restores_base(monkeypatch):
    class Layer:
        def __init__(self):
            self.A = torch.nn.Parameter(torch.zeros(1))
            self.B = torch.nn.Parameter(torch.zeros(1))
            self.scale = 2.0
            self.enabled = True

    layer = Layer()

    class Adapter:
        def layers(self):
            return [layer]

        @contextmanager
        def teacher(self):
            layer.enabled = False
            try:
                yield
            finally:
                layer.enabled = True

    runtime = Runtime.__new__(Runtime)
    runtime.lock = threading.Lock()
    runtime.adapter = Adapter()
    runtime.loaded = object()
    runtime.base_scales = [2.0]
    runtime.tensors = {
        "digits": [(torch.tensor([1.0]), torch.tensor([3.0]))],
        "cardinals": [(torch.tensor([4.0]), torch.tensor([5.0]))],
    }
    seen = []

    def fake_transcribe(*args, **kwargs):
        seen.append((layer.enabled, layer.A.item(), layer.B.item(), layer.scale))
        return "שלום", type("Result", (), {"ended": True})()

    monkeypatch.setattr("deploy.app.M.transcribe", fake_transcribe)
    runtime.transcribe(None, -25)
    runtime.transcribe(None, 0)
    runtime.transcribe(None, 100)
    assert seen == [(True, 1.0, 3.0, 0.5), (False, 1.0, 3.0, 0.5), (True, 4.0, 5.0, 2.0)]
    assert layer.enabled


def test_preset_selection_is_reproducible_and_keeps_audio_out_of_source(tmp_path):
    source, output = tmp_path / "audio", tmp_path / "assets" / "presets"
    source.mkdir()
    for i in range(24):
        with wave.open(str(source / f"clip-{i:02d}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 16000)
    manifest = select(source, output, seed=42)
    sources = json.loads((output / "sources.json").read_text())
    assert len(manifest) == 20
    assert len({x["source"] for x in sources["sources"]}) == 20
    assert load_presets(tmp_path / "assets") == manifest
    assert select(source, output, seed=42) == manifest
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert "clip-" not in (output / "manifest.json").read_text()


def test_preset_api_uses_allowlist_and_same_inference_path(tmp_path, monkeypatch):
    folder = tmp_path / "presets"
    folder.mkdir()
    (folder / "sample-01.wav").write_bytes(b"test wav")
    monkeypatch.setattr(app.state, "presets", [{"id": "sample-01", "label": "Sample 01", "duration": 1}], raising=False)
    monkeypatch.setattr(app.state, "preset_dir", folder, raising=False)

    class FakeRuntime:
        def transcribe(self, audio, position):
            return {"bias": position, "text": "שלום"}

    monkeypatch.setattr(app.state, "runtime", FakeRuntime(), raising=False)
    monkeypatch.setattr("deploy.app.decode_audio", lambda path: path.name)
    client = TestClient(app)
    assert client.get("/presets").json()[0]["id"] == "sample-01"
    assert client.get("/presets/sample-01/audio").content == b"test wav"
    assert client.get("/presets/../audio").status_code == 404
    assert client.get("/presets/sample-99/audio").status_code == 404
    assert client.post("/transcribe", data={"preset": "sample-01", "bias": "-50"}).json() == {"bias": -50, "text": "שלום"}
    assert client.post("/transcribe", data={"preset": "sample-99"}).status_code == 404
    assert client.post("/transcribe", data={"preset": "sample-01"}, files={"audio": ("a.wav", b"x")}).status_code == 422
    assert client.post("/transcribe", data={"bias": "0"}).status_code == 422


def test_recording_without_duration_metadata_still_decodes_and_checks_length(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def command(argv, **kwargs):
        if argv[0] == "ffprobe":
            return SimpleNamespace(stdout="N/A\n")
        return SimpleNamespace(stdout=b"\0\0\0\0" * 16000)

    monkeypatch.setattr("deploy.app.subprocess.run", command)
    audio = decode_audio(tmp_path / "mic.webm")
    assert len(audio) == 16000
    assert audio[0] == 0


def test_real_unseekable_webm_recording_is_accepted(tmp_path):
    path = tmp_path / "microphone.webm"
    with path.open("wb") as out:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:a", "libopus", "-f", "webm", "-"], stdout=out, check=True)
    assert 15000 <= len(decode_audio(path)) <= 17000
