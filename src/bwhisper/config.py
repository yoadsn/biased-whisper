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
# every job stores the full resolved config next to its results.
PHASE1_DEFAULTS: dict[str, Any] = {
    "model": None,  # None -> settings.model_active
    "sample_ids": None,  # explicit corpus ids, or None to take `limit` samples
    "limit": 50,
    "require_ref_digits": True,
    "max_new_tokens": 96,
    "extra_digit_chars": "",  # e.g. "%" to also treat as digit-related
    "min_rewind_digit_tokens": 1,
}

PHASE2_DEFAULTS: dict[str, Any] = {
    "model": None,
    "phase1_runs": [],  # run ids supplying decision points
    "verdicts": ["cardinal", "human_good"],  # which phase-1 verdicts are usable
    "max_points": 100,
    "objective": "mass",  # mass | pair | mass_vs_card
    "mistake_prob": 0.01,  # target prob = 1 - mistake_prob
    "nudge_mode": "linear",  # linear | iterative
    "skip_satisfied": True,  # only compute a nudge from samples that need it
    "iter_steps": 15,
    "iter_lr": 0.5,
    "param_filter": "bitfit",  # bitfit | decoder | decoder_no_embed | encoder | all_no_embed | all
    "include_regex": "",
    "exclude_regex": "",
    "normalize": "rms",  # none | rms  (per-tensor scaling used for selection & voting)
    "sample_pct": 5.0,  # "upstream params sample pct"
    "common_pct": 30.0,  # "Common upstream params percent"
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
    "phase2_run": None,
    "nudge_types": ["to_digits", "to_cardinal"],
    "strengths": [2.0, 5.0, 10.0, 20.0],
    "stat": "mean_all",  # mean_all | mean_present | median_present
    "candidate_limit": None,  # use top-N candidates by votes
    "eval_limit": 50,
    "require_ref_digits": True,
    "exclude_phase2_samples": True,
    "controls": ["shuffled_values", "random_indices"],
    "max_new_tokens": 96,
    "seed": 0,
}
