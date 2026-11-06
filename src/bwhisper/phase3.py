"""Phase 3 - control experiments.

Apply the Phase-2 candidate nudge (a statistic of per-sample nudges, scaled by
`strength`) to the weights, decode held-out corpus utterances, and measure how
the digit/cardinal split of number mentions moves. Every nudge condition is
paired with *controls* of identical size/magnitude (values shuffled across the
same params, or the same values at random params) to show specificity.

strength = 1.0 means: the average per-sample min-norm nudge restricted to the
candidate set. (Restricting to a subset delivers less than the full first-order
change, so useful strengths are usually > 1.)
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .config import PHASE3_DEFAULTS, RUNS_DIR
from .corpus import store
from .hebrew_numbers import classify_word, count_mentions, is_degenerate, words
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase1 import decision_points
from .phase2 import point_objective


def _strip_numbers(text: str) -> list[str]:
    return [w for w in words(text) if classify_word(w) is None]


def _metrics(texts: dict[int, str], base: dict[int, str] | None) -> dict:
    n = len(texts)
    dm = wm = with_digit = degen = 0
    drifts = []
    for sid, t in texts.items():
        c = count_mentions(t)
        dm += c["digit_mentions"]
        wm += c["word_mentions"]
        with_digit += c["digit_mentions"] > 0
        degen += is_degenerate(t)
        if base is not None:
            drifts.append(1 - SequenceMatcher(a=_strip_numbers(base[sid]), b=_strip_numbers(t), autojunk=False).ratio())
    return {
        "n": n,
        "digit_output_rate": with_digit / max(n, 1),
        "digit_mentions": dm,
        "word_mentions": wm,
        "digit_share": dm / (dm + wm) if dm + wm else None,
        "degenerate_rate": degen / max(n, 1),
        "text_drift": float(np.mean(drifts)) if drifts else 0.0,
    }


@register("phase3", kind="gpu", defaults=PHASE3_DEFAULTS)
def run_phase3(ctx: Ctx, cfg: dict) -> dict:
    p2dir = RUNS_DIR / cfg["phase2_run"]
    p2cfg = json.loads((p2dir / "job.json").read_text())["config"]
    L = M.load_model(cfg["model"] or p2cfg.get("model"))
    space = ParamSpace(L.model, p2cfg["param_filter"], p2cfg["include_regex"], p2cfg["exclude_regex"])
    sig = json.loads((p2dir / "space.json").read_text())["signature"]
    if sig != space.signature:
        raise RuntimeError("parameter space signature mismatch: phase-2 candidates were computed on a different model/filter")
    rng = np.random.default_rng(int(cfg["seed"]))

    pts = decision_points(p2cfg["phase1_runs"], p2cfg["verdicts"], int(p2cfg["max_points"]))
    p2_ids = {p["sample_id"] for p in pts}
    cands = {}
    for t in cfg["nudge_types"]:
        f = p2dir / f"candidates_{t}.npz"
        if not f.exists():
            ctx.log(f"no candidates for {t}; skipping")
            continue
        z = np.load(f)
        if len(z["idx"]) == 0:
            ctx.log(f"candidate set for {t} is empty; skipping")
            continue
        lim = cfg.get("candidate_limit")
        sl = slice(0, int(lim)) if lim else slice(None)
        idx, val = z["idx"][sl], z[cfg["stat"]][sl]
        o = np.argsort(idx)
        cands[t] = (idx[o].astype(np.int64), val[o].astype(np.float32))
        ctx.log(f"{t}: {len(idx):,} candidates, |delta|_2={np.linalg.norm(val):.4g}")
    if not cands:
        raise RuntimeError("no candidate sets to test")

    pool = store.query(source=cfg.get("eval_source") or None, digits_only=bool(cfg["require_ref_digits"]), limit=10**6)
    if cfg["exclude_phase2_samples"]:
        pool = [s for s in pool if s["id"] not in p2_ids]
    rng.shuffle(pool)
    samples = pool[: int(cfg["eval_limit"])]
    probe = pts[:30]
    ctx.log(f"eval samples={len(samples)}  in-sample probe points={len(probe)}")

    audio = {s["id"]: load_audio(s["audio_path"]) for s in samples}
    probe_audio = {p["sample_id"]: load_audio(store.get(p["sample_id"])["audio_path"]) for p in probe}

    conds = [("baseline", None, 0.0, "none")]
    for t in cands:
        for s in cfg["strengths"]:
            conds.append((t, t, float(s), "nudge"))
            for c in cfg["controls"]:
                conds.append((f"{t}|{c}", t, float(s), c))
    ctx.log(f"{len(conds)} conditions x {len(samples)} samples")

    enc_cache: dict[int, torch.Tensor] = {}
    probe_enc: dict[int, torch.Tensor] = {}
    base_texts: dict[int, str] | None = None
    table, done, total = [], 0, len(conds) * len(samples)

    for name, t, strength, kind in conds:
        ctx.check()
        undo = None
        touches_encoder = False
        info = {}
        if t is not None:
            idx, val = cands[t]
            if kind == "shuffled_values":
                val = rng.permutation(val)
            elif kind == "random_indices":
                idx = np.sort(rng.choice(space.total, len(idx), replace=False)).astype(np.int64)
            enc_mask = np.array([n.startswith("model.encoder.") for n in space.names])
            touches_encoder = bool(enc_mask[space.tensor_ids(idx)].any())
            undo = space.add_sparse(idx, val * strength)
            info = {"n_params": int(len(idx)), "delta_l2": float(np.linalg.norm(val * strength))}
        try:
            texts: dict[int, str] = {}
            for s in samples:
                ctx.check()
                sid = s["id"]
                if touches_encoder or sid not in enc_cache:
                    e = M.encode(L, audio[sid])
                    if not touches_encoder:
                        enc_cache[sid] = e
                else:
                    e = enc_cache[sid]
                r = M.greedy(L, e, list(L.sot_seq), max_new=int(cfg["max_new_tokens"]))
                texts[sid] = L.decode(r.tokens).strip()
                ctx.append_jsonl("outputs.jsonl", {"cond": name, "strength": strength, "sample_id": sid, "text": texts[sid]})
                done += 1
                ctx.progress(done, total, f"{name} x{strength}")
            # in-sample probe: digit log-odds at the phase-2 decision points
            ds, ps = [], []
            with torch.inference_mode():
                for p in probe:
                    sid = p["sample_id"]
                    e = M.encode(L, probe_audio[sid]) if touches_encoder or sid not in probe_enc else probe_enc[sid]
                    if not touches_encoder:
                        probe_enc[sid] = e
                    d = float(point_objective(L, e, p, p2cfg["objective"]))
                    ds.append(d)
                    ps.append(1 / (1 + np.exp(-d)))
        finally:
            if undo is not None:
                space.undo_sparse(undo)
        if name == "baseline":
            base_texts = texts
        m = _metrics(texts, base_texts)
        row = {"cond": name, "type": t, "strength": strength, "kind": kind, **info, **m,
               "probe_mean_logodds": float(np.mean(ds)) if ds else None, "probe_mean_p_digit": float(np.mean(ps)) if ps else None}
        table.append(row)
        ctx.write_json("table.json", table)
        ctx.log(f"{name:28s} x{strength:<4} digit_share={m['digit_share']} digit_rate={m['digit_output_rate']:.2f} "
                f"degenerate={m['degenerate_rate']:.2f} drift={m['text_drift']:.3f} probe_p={row['probe_mean_p_digit']}")
    return {"conditions": len(conds), "eval_samples": len(samples), "table": table}
