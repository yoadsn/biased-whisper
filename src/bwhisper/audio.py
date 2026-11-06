from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def load_audio(path: str | Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode (a time range of) any ffmpeg-readable file to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def crop_to_wav(src: str | Path, dst: str | Path, start: float | None, end: float | None) -> float:
    """Crop [start, end] of src to a 16 kHz mono wav at dst. Returns duration."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return wav_duration(dst)


def wav_duration(path: str | Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
