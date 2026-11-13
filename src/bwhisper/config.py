"""Paths, persistent settings and default hyperparameters.

Everything the harness writes lives under ``data/`` (gitignored).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("BW_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("BW_DATA", ROOT / "data"))
CORPUS_DIR = DATA / "corpus"
AUDIO_DIR = CORPUS_DIR / "audio"
RUNS_DIR = DATA / "runs"
SECRETS_DIR = DATA / "secrets"
KNOWLEDGE_DIR = ROOT / "knowledge"
WEB_DIR = ROOT / "web"
SETTINGS_PATH = DATA / "settings.json"

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30.0

DEFAULT_SETTINGS: dict[str, Any] = {
    # models
    "model_smoke": "yoad/whisper-tiny",
    "model_infer": "yoad/whisper-small",
    "model_active": "yoad/whisper-tiny",
    "device": "auto",  # auto | cpu | mps | cuda
    "language": "he",
    "timestamps": False,
    # explore.ivrit.ai
    "explore_base_url": "https://explore.ivrit.ai",
}

_lock = threading.Lock()


def ensure_dirs() -> None:
    for d in (DATA, CORPUS_DIR, AUDIO_DIR, RUNS_DIR, SECRETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    ensure_dirs()
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        s.update(json.loads(SETTINGS_PATH.read_text()))
    return s


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        s = load_settings()
        for k, v in patch.items():
            if k in DEFAULT_SETTINGS:
                s[k] = v
        SETTINGS_PATH.write_text(json.dumps(s, indent=2, ensure_ascii=False))
        return s


# Hyperparameters per phase. The UI renders its forms from these defaults and
# every job stores the full resolved config next to its results. Ordinary
# weight-voting Phase 2 is retained for old artifacts; current discovery uses
# the component-level `phase2n` job.
PHASE1_DEFAULTS: dict[str, Any] = {
    "model": None,  # None -> settings.model_active
    "devices": [],  # explicit CUDA devices -> one independent model replica per device
    "sample_ids": None,  # explicit corpus ids, or None to take `limit` samples
    "limit": 50,
    "require_ref_digits": True,
    "max_new_tokens": 96,
    "extra_digit_chars": "",  # e.g. "%" to also treat as digit-related
    "span_mode": True,  # v2: rewind to the style span start (prefix/hyphen), ban the digit-style family
    "satellites_run": None,  # phase1_5 run supplying satellite tokens; null -> latest, or built-in rules
}

PHASE2_DEFAULTS: dict[str, Any] = {
    "model": None,
    "capture_backend": "auto",  # auto: device-native linear capture on CUDA; reference: original CPU path
    "resume_run": None,  # reuse complete compatible captures from a stopped Phase 2 run
    "release_unused_models": True,
    "phase1_runs": [],  # run ids supplying decision points
    "verdicts": ["cardinal", "human_good"],  # which phase-1 verdicts are usable
    "max_points": 100,
    "objective": "auto",  # auto (seq_contrast for v2 points, else mass) | seq_contrast | mass | pair | mass_vs_card
    "mistake_prob": 0.01,  # target prob = 1 - mistake_prob
    "nudge_mode": "linear",  # linear | iterative
    "skip_satisfied": True,  # only compute a nudge from samples that need it
    "iter_steps": 15,
    "iter_lr": 0.5,
    "iter_tol": 0.05,  # log-odds tolerance for "target reached"
    # knob space: every weight and bias except the token/position embeddings (the token embedding is
    # tied to the output readout and is handled as a separate baseline). bitfit = biases + LayerNorm only.
    "param_filter": "all_no_embed",  # all_no_embed | all | decoder_no_embed | decoder | encoder | bitfit
    "include_regex": "",
    "exclude_regex": "",
    "normalize": "rms",  # none | rms  (per-tensor scaling used for selection & voting)
    "sample_pct": 5.0,  # "upstream params sample pct"
    "sample_k": 1000000,  # absolute per-sample budget; overrides sample_pct when set (null -> use pct)
    "common_pct": 30.0,  # "Common upstream params percent"
    "common_k": None,  # absolute cap on the common set (applied after common_pct)
    "agree_top_pct": 50.0,  # "top agreement params percent"
    "group_size": 3,
    "rounds": 2000,
    "fdr": 0.05,
    "max_candidates": 50000,
    "seed": 0,
    "max_store_gb": 8.0,
}

PHASE3_DEFAULTS: dict[str, Any] = {
    "model": None,
    "inference_backend": "auto",  # CUDA: batched tokens-only; reference: original diagnostic decoder
    "batch_size": 8,
    "release_model_after_run": True,
    "eval_sample_ids": None,  # explicit recording-held-out cohort from setup_scale
    "phase2_run": None,
    "nudge_types": ["to_digits", "to_cardinal"],
    "strengths": [2.0, 5.0, 10.0, 20.0],
    "stat": "mean_all",  # mean_all | mean_present | median_present
    "candidate_limit": None,  # use top-N candidates by votes
    "eval_limit": 50,
    "require_ref_digits": True,
    "exclude_phase2_samples": True,
    "eval_source": None,  # restrict eval utterances to one corpus source (e.g. hf_cache, explore)
    "controls": ["shuffled_values", "random_indices"],
    "max_new_tokens": 96,
    "seed": 0,
}
