"""Local audio playground: serialized jobs reuse one baseline model, not replicas."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import wave

import numpy as np
import torch

from . import model as M
from .audio import write_pcm16_wav
from .config import DATA, RUNS_DIR, SAMPLE_RATE
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase2_capture import measured
from .phase3_inference import FeatureBank, greedy_tokens

TEST_AUDIO_DIR = DATA / "test_audio"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 30


def run_dir(run_id):
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run id")
    directory = RUNS_DIR / run_id
    if not directory.is_dir():
        raise ValueError("run not found")
    return directory


def source_spec(phase3_run):
    directory = run_dir(phase3_run)
    job = json.loads((directory / "job.json").read_text())
    if job["type"] != "phase3" or job["status"] != "done":
        raise ValueError("select a completed Phase 3 run")
    cfg = job["config"]
    p2dir = run_dir(cfg["phase2_run"])
    p2job = json.loads((p2dir / "job.json").read_text())
    p2cfg = p2job["config"]
    model = cfg.get("model") or (job.get("summary") or {}).get("model")
    # Older Phase 3 jobs often stored model=null. Resolve from their immutable
    # Phase 2 output, never today's potentially different active-model setting.
    if not model:
        summary_path = p2dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        model = p2cfg.get("model") or summary.get("model") or (p2job.get("summary") or {}).get("model")
    if not model:
        raise ValueError("this run did not record its resolved model; choose a run with model provenance")
    stat = cfg["stat"]
    if stat not in ("mean_all", "mean_present", "median_present"):
        raise ValueError("unsupported candidate statistic")
    edits = []
    for name in cfg["nudge_types"]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("invalid edit type in source run")
        path = p2dir / f"candidates_{name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as values:
            count = len(values["idx"])
            limit = cfg.get("candidate_limit")
            if limit:
                count = len(values["idx"][:int(limit)])
            if count and stat in values:
                edits.append({"id": name, "params": count})
    if not edits:
        raise ValueError("no candidate edits available for this run")
    return {"phase3_run": phase3_run, "phase2_run": cfg["phase2_run"], "model": model,
            "stat": stat, "candidate_limit": cfg.get("candidate_limit"), "edits": edits,
            "strengths": cfg["strengths"],
            "max_new_tokens": int(cfg["max_new_tokens"]), "p2_config": p2cfg}


def audio_dir(audio_id):
    if not isinstance(audio_id, str) or not re.fullmatch(r"[0-9a-f]{32}", audio_id):
        raise ValueError("invalid audio id")
    directory = TEST_AUDIO_DIR / audio_id
    if not (directory / "audio.wav").is_file():
        raise ValueError("audio not found; upload or record a sample first")
    return directory


def normalize_upload(path, filename):
    """Bound decoding by time/output size, reject >30 s (never silently crop)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
           "-format_whitelist", "wav,mp3,flac,ogg,matroska,webm,mov,aac,aiff,asf",
           "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", "30.001",
           "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    try:
        decoded = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError("audio decoding timed out") from None
    if decoded.returncode:
        raise ValueError("cannot decode this audio file; try WAV, MP3, FLAC, Ogg, WebM or M4A")
    pcm = np.frombuffer(decoded.stdout, dtype=np.int16)
    if not len(pcm):
        raise ValueError("audio is empty")
    if len(pcm) > SAMPLE_RATE * MAX_AUDIO_SECONDS:
        raise ValueError("audio must be at most 30 seconds; trim it and upload again")
    token = secrets.token_hex(16)
    directory = TEST_AUDIO_DIR / token
    directory.mkdir(parents=True)
    duration = write_pcm16_wav(directory / "audio.wav", pcm)
    info = {"id": token, "duration": duration, "filename": str(filename or "audio")[:255],
            "url": f"/api/test/audio/{token}"}
    (directory / "audio.json").write_text(json.dumps(info))
    return info


def validated_configs(spec, configs):
    if not isinstance(configs, list):
        raise ValueError("edit_configs must be an array")
    available = {edit["id"] for edit in spec["edits"]}
    result = []
    for i, row in enumerate(configs):
        if not isinstance(row, dict) or row.get("edit") not in available:
            raise ValueError(f"edit config {i + 1}: choose an available edit")
        amount = row.get("strength")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError(f"edit config {i + 1}: strength must be a finite number")
        result.append({"edit": row["edit"], "strength": float(amount),
                       "label": str(row.get("label") or f"{row['edit']} × {amount}")[:120]})
    return result


TEST_DEFAULTS = {"phase3_run": None, "audio_id": None, "edit_configs": [],
                 "release_model_after_run": False}


@register("test_transcribe", kind="gpu", defaults=TEST_DEFAULTS)
def transcribe(ctx: Ctx, cfg):
    ctx.check()
    spec = source_spec(cfg["phase3_run"])
    configs = validated_configs(spec, cfg["edit_configs"])
    directory = audio_dir(cfg["audio_id"])
    with wave.open(str(directory / "audio.wav")) as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnframes() > SAMPLE_RATE * MAX_AUDIO_SECONDS:
            raise ValueError("invalid normalized audio")
    # Explicit device preference: CUDA when available, otherwise configured auto
    # fallback; unlike scientific experiments this playground defaults to GPU.
    device = "cuda:0" if torch.cuda.is_available() else M.pick_device("auto")
    L = M.load_model(spec["model"], device=device)
    M.unload_unused(L)
    p2cfg = spec["p2_config"]
    space = ParamSpace(L.model, p2cfg["param_filter"], p2cfg["include_regex"], p2cfg["exclude_regex"])
    p2dir = run_dir(spec["phase2_run"])
    if space.signature != json.loads((p2dir / "space.json").read_text())["signature"]:
        raise ValueError("model/parameter signature does not match the selected run")
    bank = FeatureBank(L, [{"id": 0, "audio_path": str(directory / "audio.wav")}], ctx)
    supports = {}
    for edit in dict.fromkeys(row["edit"] for row in configs):
        with np.load(p2dir / f"candidates_{edit}.npz", allow_pickle=False) as z:
            limit = spec["candidate_limit"]
            sl = slice(0, int(limit)) if limit else slice(None)
            idx, val = z["idx"][sl], z[spec["stat"]][sl]
            order = np.argsort(idx)
            idx = torch.as_tensor(idx[order].astype(np.int64), device=L.device)
            val = torch.as_tensor(val[order].astype(np.float32), device=L.device)
        if not bool(torch.isfinite(val).all()):
            raise ValueError("nonfinite source edit values")
        supports[edit] = idx, val, space.sparse_bounds(idx)
    rows = []
    conditions = [{"edit": None, "strength": 0.0, "label": "Base model (untouched)"}] + configs
    ctx.log(f"Test model={L.name} device={L.device}; one model, {len(conditions)} independent configs")
    for i, config in enumerate(conditions):
        ctx.check()
        ctx.progress(i, len(conditions), config["label"])
        undo, timing = None, {}
        try:
            with measured(timing, "edit_s", L.device):
                if config["edit"] is not None:
                    idx, value, bounds = supports[config["edit"]]
                    delta = value * config["strength"]
                    if not bool(torch.isfinite(delta).all()):
                        raise ValueError("edit amount overflows model precision; use a smaller strength")
                    undo = space.add_sparse_device(idx, delta, bounds)
            # Never reuse a baseline encoder output for an edited condition.
            with measured(timing, "encoder_s", L.device):
                enc = bank.encode(L, [0])
            with measured(timing, "decoder_s", L.device):
                tokens = greedy_tokens(L, enc, spec["max_new_tokens"], ctx.check)[0]
            del enc
        finally:
            if undo is not None:
                try:
                    space.undo_sparse(undo)
                except Exception:
                    # Do not allow a failed restoration to poison the next test.
                    M.unload_all()
                    raise
        row = {"index": i, **config, "text": L.decode(tokens).strip(), "timings": timing,
               "truncated": len(tokens) == spec["max_new_tokens"]}
        rows.append(row)
        ctx.append_jsonl("results.jsonl", row)
        ctx.progress(i + 1, len(conditions), config["label"])
    summary = {"phase3_run": cfg["phase3_run"], "audio_id": cfg["audio_id"], "model": L.name,
               "device": L.device, "stat": spec["stat"], "candidate_limit": spec["candidate_limit"],
               "max_new_tokens": spec["max_new_tokens"],
               "model_kept_loaded": not bool(cfg.get("release_model_after_run")), "results": rows}
    ctx.write_json("summary.json", summary)
    return summary


@register("model_release", kind="gpu", defaults={"release_model_after_run": True})
def release(ctx: Ctx, cfg):
    # Actual teardown occurs in the common worker after this frame unwinds.
    return {"release_requested": True}
