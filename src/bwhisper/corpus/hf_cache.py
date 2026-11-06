"""Corpus source: ivrit.ai-style datasets already present in the local HF cache.

Each dataset entry has ``audio.m4a`` + ``transcript.aligned.json`` (timed
segments). We build utterance windows (merged adjacent segments, <= max_dur)
around segments containing digits and crop them to 16 kHz wav.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from ..audio import crop_to_wav
from ..hebrew_numbers import DIGIT_RE
from . import store

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"


def list_datasets() -> list[dict]:
    out = []
    for d in sorted(HF_HUB.glob("datasets--*")):
        snaps = list(d.glob("snapshots/*"))
        if not snaps:
            continue
        entries = [p for p in snaps[0].iterdir() if p.is_dir() and (p / "transcript.aligned.json").exists()]
        if entries:
            out.append({"name": d.name.removeprefix("datasets--").replace("--", "/"), "dir": str(snaps[0]), "entries": len(entries)})
    return out


def build_windows(segments: list[dict], min_dur: float, max_dur: float) -> list[tuple[int, int]]:
    """Index ranges [i, j] of segments, each containing >=1 digit segment, non-overlapping."""
    used = [False] * len(segments)
    wins = []
    for i, s in enumerate(segments):
        if used[i] or not DIGIT_RE.search(s.get("text", "")):
            continue
        a = b = i
        while True:
            dur = segments[b]["end"] - segments[a]["start"]
            if dur >= min_dur:
                break
            grew = False
            if a > 0 and not used[a - 1] and segments[b]["end"] - segments[a - 1]["start"] <= max_dur:
                a -= 1; grew = True
            if (segments[b]["end"] - segments[a]["start"]) < min_dur and b + 1 < len(segments) and not used[b + 1] \
               and segments[b + 1]["end"] - segments[a]["start"] <= max_dur:
                b += 1; grew = True
            if not grew:
                break
        if not (1.5 <= segments[b]["end"] - segments[a]["start"] <= max_dur):
            continue
        for k in range(a, b + 1):
            used[k] = True
        wins.append((a, b))
    return wins


def import_from_cache(
    dataset: str | None = None,
    limit: int = 200,
    seed: int = 0,
    min_dur: float = 6.0,
    max_dur: float = 25.0,
    log: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda a, b: None,
) -> dict:
    rng = random.Random(seed)
    dsets = [d for d in list_datasets() if dataset in (None, "", d["name"])]
    entries: list[Path] = []
    for d in dsets:
        entries += [p for p in Path(d["dir"]).iterdir() if p.is_dir() and (p / "transcript.aligned.json").exists()]
    rng.shuffle(entries)
    log(f"{len(entries)} cached entries in {[d['name'] for d in dsets]}")
    added = skipped = 0
    for n, e in enumerate(entries):
        if added >= limit or should_stop():
            break
        progress(added, limit)
        audio = e / "audio.m4a"
        if not audio.exists():
            continue
        segs = json.loads((e / "transcript.aligned.json").read_text()).get("segments", [])
        segs = [s for s in segs if s.get("end") is not None and s.get("text")]
        wins = build_windows(segs, min_dur, max_dur)
        rng.shuffle(wins)
        dset = e.parent.parent.parent.name.removeprefix("datasets--").replace("--", "/")
        for a, b in wins:
            if added >= limit:
                break
            start, end = segs[a]["start"], segs[b]["end"]
            key = f"{dset}/{e.name}@{start:.2f}-{end:.2f}"
            if store.exists("hf_cache", key):
                skipped += 1
                continue
            text = "".join(s["text"] for s in segs[a : b + 1]).strip()
            dst = store.audio_path_for("hf_cache", key)
            try:
                dur = crop_to_wav(audio, dst, start, end)
            except Exception as ex:  # noqa: BLE001
                log(f"skip {key}: {ex}")
                continue
            store.add_sample("hf_cache", key, text, dst, dur, start, end,
                             meta={"dataset": dset, "entry": e.name}, batch=f"hf_cache:seed{seed}")
            added += 1
        log(f"[{n + 1}/{len(entries)}] {dset}/{e.name}: total added {added}")
    progress(added, limit)
    return {"added": added, "skipped_existing": skipped}
