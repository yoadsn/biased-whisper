"""Standalone Cloud Run inference service for the two Phase 4 LoRA checkpoints."""
from contextlib import asynccontextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from bwhisper import model as M
from bwhisper.lora import Adapter


ASSETS = Path(os.environ.get("BW_ASSETS", Path(__file__).parent / "assets"))
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 30.0  # Whisper's feature extractor otherwise silently truncates.


def selection(position: int) -> tuple[str | None, float]:
    """-100 = digits, 0 = untouched base, +100 = cardinals."""
    if type(position) is not int or not -100 <= position <= 100:
        raise ValueError("Bias must be an integer between -100 and 100")
    return ("digits", -position / 100) if position < 0 else (("cardinals", position / 100) if position > 0 else (None, 0.0))


class Runtime:
    def __init__(self, assets: Path = ASSETS):
        self.lock = threading.Lock()
        self.loaded = M.load_model(str(assets / "model"), device="cuda")
        self.adapters = {}
        self.tensors = {}
        architecture = None
        for name in ("digits", "cardinals"):
            directory = assets / name
            metadata = json.loads((directory / "adapter.json").read_text())
            if metadata.get("format") != "bwhisper-lora-v1" or metadata.get("target_style", "cardinals") != name:
                raise ValueError(f"Incorrect {name} adapter metadata")
            keys = ("rank", "alpha", "modules", "encoder_top_layers", "sites")
            signature = tuple(json.dumps(metadata[k], sort_keys=True) for k in keys)
            if architecture is not None and signature != architecture:
                raise ValueError("Digit and cardinal adapter architectures differ")
            architecture = signature
            adapter = Adapter(self.loaded.model, metadata["rank"], metadata["alpha"], 0.0,
                              metadata["modules"], metadata["encoder_top_layers"])
            # Training dropout is disabled at inference. load() validates the saved
            # training config, so temporarily use its exact architecture for loading.
            adapter.config["dropout"] = metadata["dropout"]
            try:
                adapter.load(directory)
                self.tensors[name] = [(layer.A.detach().clone(), layer.B.detach().clone()) for layer in adapter.layers()]
                self.adapters[name] = metadata["step"]
            finally:
                adapter.restore()
        metadata = json.loads((assets / "digits" / "adapter.json").read_text())
        self.adapter = Adapter(self.loaded.model, metadata["rank"], metadata["alpha"], 0.0,
                               metadata["modules"], metadata["encoder_top_layers"])
        self.loaded.model.eval()
        self.base_scales = [layer.scale for layer in self.adapter.layers()]
        if not self.tensors or any(len(t) != len(self.base_scales) for t in self.tensors.values()):
            raise ValueError("Adapter sites differ")

    def transcribe(self, audio, position: int):
        name, strength = selection(position)
        with self.lock, self.adapter.teacher() if name is None else nullcontext():
            if name:
                with __import__("torch").no_grad():
                    for layer, (a, b), scale in zip(self.adapter.layers(), self.tensors[name], self.base_scales):
                        layer.A.copy_(a)
                        layer.B.copy_(b)
                        layer.scale = scale * strength
            text, result = M.transcribe(self.loaded, audio, max_new=192)
            return {"text": text, "bias": position, "adapter": name or "base", "strength": strength,
                    "ended": result.ended}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = Runtime()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/ready")
def ready():
    return {"ready": True, "adapters": app.state.runtime.adapters}


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), bias: int = Form(0)):
    try:
        selection(bias)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "upload"
        count = 0
        with path.open("wb") as out:
            while chunk := audio.file.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_BYTES:
                    raise HTTPException(413, "File exceeds 25 MB")
                out.write(chunk)
        if count == 0:
            raise HTTPException(400, "Empty file")
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                                   capture_output=True, text=True, timeout=15, check=True)
            duration = float(probe.stdout.strip())
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid duration")
            if duration > MAX_SECONDS:
                raise HTTPException(413, "Audio must be at most 30 seconds")
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-t", "30.01",
                                      "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                                     capture_output=True, timeout=60, check=True)
            import numpy as np
            samples = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
            if not 0 < len(samples) <= 16000 * MAX_SECONDS:
                raise HTTPException(413, "Audio must be 0–30 seconds")
        except (subprocess.SubprocessError, ValueError) as exc:
            raise HTTPException(400, "Unable to decode audio") from exc
        return app.state.runtime.transcribe(samples, bias)
