import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import threading
from contextlib import contextmanager

import torch

from deploy.app import Runtime, selection


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
