"""Import original-text digit examples using PocketTTS aligned audio slices."""
from __future__ import annotations

import hashlib
import gzip
import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..audio import crop_to_wav
from ..jobs import Ctx
from . import store

SOURCES = ("plenum", "committee", "crowd_transcribe_v5")
DEFAULTS = {
    "root": "/mlspeech/hebrew/pockettts-data",
    "sources": list(SOURCES),
    "splits": ["train", "valid"],
    "limit": 10000,
    "seed": 0,
    "min_dur": 1.5,
    "max_dur": 28.0,
    "workers": 8,
}


def crowd_originals(root: Path) -> dict[str, str]:
    """Use the same preferred source record as the crowd manifest builder."""
    paths = sorted((root / "crowd_transcribe_v5" / "meta").glob("*-index.json.gz"))
    if len(paths) != 1:
        raise ValueError("need exactly one crowd original-text source index")
    with gzip.open(paths[0], "rt", encoding="utf-8") as f:
        records = json.load(f)
    best = {}
    for record in records:
        key = record["uuid"]
        priority = (not record.get("retranscribed", False), record["shard"], record["row"])
        if key not in best or priority < best[key][0]:
            best[key] = (priority, record["text"])
    return {key: value[1] for key, value in best.items()}


def original_text(row: dict, dataset: str, originals: dict[str, str] | None = None) -> tuple[str, str]:
    if dataset == "crowd_transcribe_v5":
        # Older crowd manifests do not preserve inverse-normalization spans.
        if originals is None or row.get("clip") not in originals:
            raise ValueError("crowd clip missing from original-text source index")
        return originals[row["clip"]], "crowd_source_index"
    spans = row.get("alt_words", {}).get("norms", [])
    if not spans:
        return row.get("transcript", ""), "unchanged_manifest"
    # Despite the label 'norms', replace contains the PRE-normalization form.
    # Builder contract: from..to inclusive; alignment preserves word indices.
    words = [w["word"] for w in row["words"]]
    previous = -1
    for span in spans:
        first, last = span["from"], span["to"]
        if not isinstance(first, int) or not isinstance(last, int) or not previous < first <= last < len(words):
            raise ValueError("invalid original-text replacement spans")
        if not isinstance(span["replace"], list) or not all(isinstance(w, str) for w in span["replace"]):
            raise ValueError("invalid original-text replacement words")
        previous = last
    for span in reversed(spans):
        words[span["from"]:span["to"] + 1] = span["replace"]
    return " ".join(words), "restored_pre_normalization"


def candidate(row: dict, dataset: str, manifest: Path, line: int,
              min_dur: float, max_dur: float, originals: dict[str, str] | None = None) -> dict | None:
    if not isinstance(row, dict) or row.get("alignment_error"):
        return None
    text, text_origin = original_text(row, dataset, originals)
    if not isinstance(text, str) or not any(c.isdigit() for c in text) or not re.search(r"[א-ת]", text):
        return None
    try:
        start, duration = float(row["start"]), float(row["duration"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or not min_dur <= duration <= max_dur:
        return None
    if not row.get("path"):
        return None
    path = Path(row["path"])
    if not path.is_absolute():
        path = manifest.parent / path
    # Crowd files are per-clip; their parent directory identifies the recording.
    recording = str(path.parent if dataset == "crowd_transcribe_v5" else path)
    key = hashlib.sha256(f"{path}:{start:.6f}:{duration:.6f}".encode()).hexdigest()[:32]
    return {"source": f"pockettts_{dataset}", "key": key, "text": text,
            "path": str(path), "start": start, "duration": duration,
            "meta": {"dataset": dataset, "manifest": str(manifest), "line": line,
                     "text_origin": text_origin, "reference_version": 2,
                     "recording": recording, "manifest_split": manifest.name.split("_")[0],
                     "alignment_backend": row.get("alignment_backend"),
                     "alignment_error": row.get("alignment_error"),
                     "original_start": row.get("original_start"),
                     "original_duration": row.get("original_duration"),
                     "speaker": row.get("speaker"), "clip": row.get("clip")}}


def import_aligned(ctx: Ctx, cfg: dict) -> dict:
    sources = list(cfg["sources"])
    splits = list(cfg["splits"])
    limit, seed = int(cfg["limit"]), int(cfg["seed"])
    workers = int(cfg["workers"])
    if (not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES)
            or not splits or set(splits) - {"train", "valid"}):
        raise ValueError("only plenum, committee, crowd_transcribe_v5 and train/valid are supported")
    if limit < 1 or not 1 <= workers <= 32 or not 0 < float(cfg["min_dur"]) <= float(cfg["max_dur"]) <= 30:
        raise ValueError("invalid limit, workers, or duration bounds")
    root = Path(cfg["root"])
    manifests = [(s, root / s / f"{split}_aligned.jsonl") for s in sources for split in splits]
    for _, path in manifests:
        if not path.is_file():
            raise FileNotFoundError(path)  # never silently fall back to unaligned/norm manifests
    existing = {(s["source"], s["source_key"]): s for s in store.query(limit=10**9)}
    originals = crowd_originals(root) if "crowd_transcribe_v5" in sources else None
    references_repaired = 0
    rng = random.Random(seed)
    pools = {s: [] for s in sources}
    counts, eligible, seen = Counter(), Counter(), set()
    # Bounded reservoirs retain replacements for missing/undecodable files.
    capacity = limit * 2
    for dataset, manifest in manifests:
        ctx.log(f"scanning {manifest}")
        repairs = []
        with manifest.open(encoding="utf-8") as f:
            for line, raw in enumerate(f, 1):
                if line % 10000 == 0:
                    ctx.check()
                    ctx.progress(0, limit, f"scanning {dataset}: {line:,} rows; {eligible[dataset]:,} digit candidates")
                counts[f"{dataset}:rows"] += 1
                try:
                    row = json.loads(raw)
                    c = candidate(row, dataset, manifest, line, float(cfg["min_dur"]), float(cfg["max_dur"]), originals)
                except (ValueError, TypeError):
                    counts[f"{dataset}:invalid"] += 1
                    continue
                if c is None:
                    continue
                identity = (c["source"], c["key"])
                if identity in existing and existing[identity]["meta"].get("reference_version", 1) < 2:
                    old = existing[identity]
                    repairs.append((old["id"], c["text"], {**old["meta"], **c["meta"]}))
                    old["meta"]["reference_version"] = 2
                    references_repaired += 1
                    if len(repairs) >= 128:
                        store.update_references(repairs)
                        repairs.clear()
                if identity in existing or identity in seen:
                    counts[f"{dataset}:duplicates"] += 1
                    continue
                seen.add(identity)
                eligible[dataset] += 1
                pool = pools[dataset]
                if len(pool) < capacity:
                    pool.append(c)
                else:
                    j = rng.randrange(eligible[dataset])
                    if j < capacity:
                        pool[j] = c
        store.update_references(repairs)
        ctx.log(f"{dataset}: {eligible[dataset]:,} eligible original-text digit slices so far")
    for pool in pools.values():
        rng.shuffle(pool)

    def crop(c):
        dst = store.audio_path_for(c["source"], c["key"])
        try:
            dur = crop_to_wav(c["path"], dst, c["start"], c["start"] + c["duration"])
            if abs(dur - c["duration"]) > 0.1:
                raise ValueError(f"truncated audio: expected {c['duration']}, got {dur}")
            return c, dst, dur, None
        except Exception as exc:
            dst.unlink(missing_ok=True)
            return c, dst, 0, str(exc)

    added, errors, by_source = 0, 0, Counter()
    with ThreadPoolExecutor(workers) as pool:
        while added < limit and any(pools.values()):
            ctx.check()
            batch = []
            # Balanced across available sources, redistributing when a source runs out.
            while len(batch) < min(workers * 2, limit - added) and any(pools.values()):
                for dataset in sources:
                    if pools[dataset] and len(batch) < min(workers * 2, limit - added):
                        batch.append(pools[dataset].pop())
            for c, dst, dur, error in pool.map(crop, batch):
                if error:
                    errors += 1
                    ctx.append_jsonl("errors.jsonl", {"key": c["key"], "error": error})
                    continue
                sid = store.add_sample(c["source"], c["key"], c["text"], dst, dur,
                                       start=c["start"], end=c["start"] + c["duration"],
                                       meta=c["meta"], batch=ctx.job.id)
                if sid is not None:
                    added += 1
                    by_source[c["source"]] += 1
                    ctx.append_jsonl("samples.jsonl", {"sample_id": sid, **c})
            ctx.progress(added, limit, f"imported {added:,}/{limit:,}; crop errors={errors}")
    summary = {"added": added, "target": limit, "target_reached": added == limit,
               "shortfall": max(0, limit - added), "by_source": dict(by_source),
               "eligible": {s: eligible[s] for s in sources}, "scan": dict(counts), "crop_errors": errors,
               "references_repaired": references_repaired, "reference_version": 2, "seed": seed}
    ctx.write_json("inventory.json", summary)
    ctx.log(f"import complete: {summary}")
    return summary
