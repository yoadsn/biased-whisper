"""Hardware/model preflight and frozen cohorts. Never starts an experimental phase."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from .config import PHASE1_DEFAULTS, PHASE3_DEFAULTS, save_settings
from .corpus import store
from .jobs import Ctx, register

MODEL = "/mlspeech/data/yoadsnapir/models/hf/ivrit-ai/whisper-large-v3-turbo"
DEFAULTS = {"model": MODEL, "devices": [f"cuda:{i}" for i in range(8)],
            "holdout_fraction": 0.2, "seed": 0, "check_model": True}


def split_samples(samples: list[dict], fraction: float, seed: int) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    result = {"discovery": [], "evaluation": [], "groups": {}, "by_source": {}}
    for s in samples:
        meta = s["meta"]
        recording = meta.get("recording") or meta.get("uuid")
        if not recording:
            raise ValueError(f"sample {s['id']} has no recording identity")
        group = f"{s['source']}:{recording}"
        score = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
        split = "evaluation" if score < fraction else "discovery"
        result[split].append(s["id"])
        result["groups"][group] = split
        counts = result["by_source"].setdefault(s["source"], {"discovery": 0, "evaluation": 0})
        counts[split] += 1
    return result


@register("setup_scale", kind="gpu", defaults=DEFAULTS)
def setup_scale(ctx: Ctx, cfg: dict) -> dict:
    import torch
    from . import model as M

    model = str(Path(cfg["model"]).resolve())
    if not (Path(model) / "model.safetensors").is_file():
        raise FileNotFoundError(f"local model checkpoint missing: {model}")
    devices = cfg["devices"]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be unique and nonempty")
    available = {f"cuda:{i}" for i in range(torch.cuda.device_count())}
    if set(devices) - available:
        raise ValueError(f"requested devices unavailable: {set(devices) - available}")
    hardware = []
    for d in devices:
        ctx.check()
        p = torch.cuda.get_device_properties(d)
        x = torch.ones((16, 16), device=d)
        if (x @ x)[0, 0].item() != 16:
            raise RuntimeError(f"CUDA arithmetic failed on {d}")
        hardware.append({"device": d, "name": p.name, "memory_bytes": p.total_memory})
        del x
    model_info = None
    if cfg["check_model"]:
        ctx.log(f"loading {model} on {devices[0]} (no transcription or phase execution)")
        try:
            L = M.load_model(model, device=devices[0])
            model_info = {"parameters": sum(p.numel() for p in L.model.parameters()),
                          "encoder_layers": L.model.config.encoder_layers,
                          "decoder_layers": L.model.config.decoder_layers,
                          "num_mel_bins": L.model.config.num_mel_bins,
                          "digit_tokens": len(L.digit_ids)}
        finally:
            M.unload_all()
    sources = {"pockettts_plenum", "pockettts_committee", "pockettts_crowd_transcribe_v5", "explore"}
    samples = [s for s in store.query(digits_only=True, limit=10**9) if s["source"] in sources]
    cohorts = split_samples(samples, float(cfg["holdout_fraction"]), int(cfg["seed"]))
    if not cohorts["discovery"] or not cohorts["evaluation"]:
        raise ValueError("need nonempty recording-disjoint discovery and evaluation cohorts")
    ctx.write_json("cohorts.json", cohorts)
    ctx.write_json("phase1.json", {**PHASE1_DEFAULTS, "model": model, "devices": devices,
                                  "sample_ids": cohorts["discovery"], "limit": len(cohorts["discovery"]),
                                  "max_new_tokens": 192})
    ctx.write_json("phase3.template.json", {**PHASE3_DEFAULTS, "model": model,
                                           "eval_sample_ids": cohorts["evaluation"],
                                           "eval_limit": min(50, len(cohorts["evaluation"])),
                                           "max_new_tokens": 192})
    ctx.write_json("explore-fetch.json", {"target": 10000, "per_seed": 500, "per_episode": 6,
                                         "workers": 8, "mode": "range", "seeds": [234608111]})
    save_settings({"model_active": model, "model_infer": model, "device": devices[0]})
    summary = {"hardware": hardware, "model": model, "model_info": model_info,
               "corpus": dict(Counter(s["source"] for s in samples)), "cohorts": cohorts["by_source"],
               "discovery": len(cohorts["discovery"]), "evaluation": len(cohorts["evaluation"]),
               "phases_started": False}
    ctx.write_json("preflight.json", summary)
    ctx.log("setup complete; phase configs written, NOT submitted")
    return summary
