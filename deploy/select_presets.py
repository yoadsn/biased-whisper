"""Reproducibly sample 20 short local corpus WAVs into ignored image assets.

Run from the repository root: uv run python -m deploy.select_presets
The source file names are saved only in the ignored assets directory; neither
the source names nor the audio files are exposed in the public manifest.
"""
import json
import random
import shutil
import wave
from pathlib import Path


SEED = 20261113
COUNT = 20
SOURCE = Path("data/corpus/audio")
DESTINATION = Path("deploy/assets/presets")


def select(source=SOURCE, destination=DESTINATION, seed=SEED, count=COUNT):
    candidates = sorted(source.glob("*.wav"))
    random.Random(seed).shuffle(candidates)
    selected = []
    for path in candidates:
        try:
            with wave.open(str(path)) as recording:
                duration = recording.getnframes() / recording.getframerate()
        except (OSError, EOFError, ValueError, ZeroDivisionError, wave.Error):
            continue
        if not 0.5 <= duration <= 30 or path.stat().st_size > 25 * 1024 * 1024:
            continue
        selected.append((path, duration))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} valid short WAVs; found {len(selected)}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    sources = []
    for i, (path, duration) in enumerate(selected, 1):
        sample_id = f"sample-{i:02d}"
        shutil.copyfile(path, destination / f"{sample_id}.wav")
        manifest.append({"id": sample_id, "label": f"Sample {i:02d}", "duration": round(duration, 2)})
        sources.append({"id": sample_id, "source": path.name})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "sources.json").write_text(json.dumps({"seed": seed, "sources": sources}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(f"Prepared {len(select())} preset samples from {SOURCE}")
