"""Budget sweep: how many parameters does the digit/cardinal signal actually need?

Answers, per model and knob space, which per-sample budget k (Phase 2 `sample_k`) is worth
storing. For a grid of k it measures, on real decision points:

  energy_raw   fraction of |grad|^2 in the k largest-|grad| params (upper bound; a min-norm
               nudge restricted to support S must be |g|/|g_S| times larger than the full one)
  energy_sel   same, for the params the Phase-2 selection rule (per-tensor RMS) would keep
  self_real    apply the min-norm nudge restricted to that sample's own top-k toward
               `direction`, then measure the real log-odds change / requested change
  xfer_cos     held-out: cosine between a test point's gradient and the train consensus
               direction restricted to its top-k params
  xfer_real    held-out: apply the consensus restricted to top-k (scaled on the train points to
               a predicted change of `xfer_target` log-odds) and measure the real change
  xfer_shuf    same params, consensus values shuffled (control)

The knee of xfer_real vs k is the budget to use; energy curves alone say nothing about
whether the selected params generalise across utterances.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .corpus import store
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase1 import decision_points
from .phase2 import _eval_grad, point_objective, selection_score

BUDGET_DEFAULTS = {
    "model": None,
    "phase1_runs": [],
    "verdicts": ["cardinal", "human_good"],
    "param_filter": "all_no_embed",
    "include_regex": "",
    "exclude_regex": "",
    "objective": "auto",
    "normalize": "rms",
    "direction": "to_cardinal",  # to_cardinal | to_digits
    "mistake_prob": 0.01,
    "train_points": 24,
    "test_points": 12,
    "fractions": [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1],
    "xfer_target": -3.0,  # predicted mean log-odds change on train points for the transfer edit
    "seed": 0,
}


@torch.inference_mode()
def eval_d(L, audio, pt, kind) -> float:
    enc = M.encode(L, audio)
    return float(point_objective(L, enc, pt, kind))


def _topk(score: torch.Tensor, k: int) -> np.ndarray:
    idx = torch.topk(score, k, sorted=False).indices.numpy()
    idx.sort()
    return idx.astype(np.int64)


def _apply_eval(L, space, audio, pt, kind, idx, vals) -> float:
    undo = space.add_sparse(idx, vals.astype(np.float32))
    try:
        return eval_d(L, audio, pt, kind)
    finally:
        space.undo_sparse(undo)


def _ms(x) -> dict:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    return {"mean": float(a.mean()) if len(a) else None, "std": float(a.std()) if len(a) else None, "n": int(len(a))}


@register("budget_sweep", kind="gpu", defaults=BUDGET_DEFAULTS)
def run_budget(ctx: Ctx, cfg: dict) -> dict:
    L = M.load_model(cfg["model"])
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    pts = decision_points(cfg["phase1_runs"], cfg["verdicts"])
    rng = np.random.default_rng(int(cfg["seed"]))
    rng.shuffle(pts)
    ntr, nte = int(cfg["train_points"]), int(cfg["test_points"])
    if len(pts) < ntr + nte:
        ntr = max(3, int(len(pts) * 2 / 3))
        nte = len(pts) - ntr
    train, test = pts[:ntr], pts[ntr : ntr + nte]
    ks = sorted({max(1, int(round(f * space.total))) for f in cfg["fractions"]})
    eps = float(cfg["mistake_prob"])
    dstar = math.log((1 - eps) / eps) * (1 if cfg["direction"] == "to_digits" else -1)
    kind = cfg["objective"]
    enc_mask = np.array([n.startswith("model.encoder.") for n in space.names])
    ctx.log(f"model={L.name} filter={cfg['param_filter']} knobs={space.total:,} train={len(train)} test={len(test)} ks={ks}")
    total_steps = len(train) + len(test)

    audio = {}

    def get_audio(pt):
        sid = pt["sample_id"]
        if sid not in audio:
            audio[sid] = load_audio(store.get(sid)["audio_path"])
        return audio[sid]

    rows = {k: {"energy_raw": [], "energy_sel": [], "self_real": [], "xfer_cos": [], "xfer_real": [], "xfer_shuf": []} for k in ks}
    consensus = torch.zeros(space.total, dtype=torch.float32)
    train_grads_dot: list[tuple[dict, float]] = []
    space.enable_grad()
    try:
        # ---- pass 1: per-sample curves on train points + consensus accumulation
        for i, pt in enumerate(train):
            ctx.check()
            ctx.progress(i, total_steps, f"train {i + 1}/{len(train)}")
            a = get_audio(pt)
            d0, g = _eval_grad(L, space, a, pt, kind)
            delta = dstar - d0
            g2 = g.pow(2)
            tot = float(g2.sum()) + 1e-30
            raw_sorted = torch.sort(g2, descending=True).values
            cum = torch.cumsum(raw_sorted.double(), 0)
            sel_score, _ = selection_score(g, space, cfg["normalize"])
            for k in ks:
                rows[k]["energy_raw"].append(float(cum[k - 1]) / tot)
                idx = _topk(sel_score, k)
                gs = g[idx]
                es = float(gs.pow(2).sum())
                rows[k]["energy_sel"].append(es / tot)
                if es > 0:
                    vals = (gs * (delta / es)).numpy()
                    d1 = _apply_eval(L, space, a, pt, kind, idx, vals)
                    rows[k]["self_real"].append((d1 - d0) / delta)
            consensus += g / (g.norm() + 1e-30)
            ctx.log(f"train {pt['sample_id']}: d0={d0:+.2f} |g|={math.sqrt(tot):.3g} "
                    f"E_raw@{ks[len(ks)//2]}={rows[ks[len(ks)//2]]['energy_raw'][-1]:.3f}")
            del g, g2, raw_sorted, cum, sel_score
        consensus /= max(len(train), 1)
        c_score, _ = selection_score(consensus, space, cfg["normalize"])
        supports = {k: _topk(c_score, k) for k in ks}

        # first-order calibration of the transfer edit on train points (needs train grads again:
        # cheaper to recompute g.c_k per train point than to keep all gradients in memory)
        dots = {k: [] for k in ks}
        for i, pt in enumerate(train):
            ctx.check()
            _, g = _eval_grad(L, space, get_audio(pt), pt, kind)
            for k in ks:
                ck = consensus[supports[k]]
                dots[k].append(float(g[supports[k]] @ ck) / (float(ck.norm()) + 1e-30))
            del g
        scale = {}
        for k in ks:
            m = float(np.mean(dots[k]))
            scale[k] = float(cfg["xfer_target"]) / m if abs(m) > 1e-12 else 0.0

        # ---- pass 2: held-out transfer
        for i, pt in enumerate(test):
            ctx.check()
            ctx.progress(len(train) + i, total_steps, f"test {i + 1}/{len(test)}")
            a = get_audio(pt)
            d0, g = _eval_grad(L, space, a, pt, kind)
            gn = float(g.norm()) + 1e-30
            for k in ks:
                idx = supports[k]
                ck = consensus[idx]
                cn = float(ck.norm()) + 1e-30
                rows[k]["xfer_cos"].append(float(g[idx] @ ck) / (gn * cn))
                vals = (ck / cn * scale[k]).numpy()
                rows[k]["xfer_real"].append(_apply_eval(L, space, a, pt, kind, idx, vals) - d0)
                rows[k]["xfer_shuf"].append(_apply_eval(L, space, a, pt, kind, idx, rng.permutation(vals)) - d0)
            del g
    finally:
        space.disable_grad()

    curve = []
    for k in ks:
        r = rows[k]
        enc_share = float(enc_mask[space.tensor_ids(supports[k])].mean())
        curve.append({
            "k": k, "fraction": k / space.total, "bytes_per_sample": k * 8,
            "consensus_encoder_share": enc_share,
            "xfer_edit_l2": abs(scale[k]), "xfer_edit_rms_per_param": abs(scale[k]) / math.sqrt(k),
            **{name: _ms(v) for name, v in r.items()},
        })
        ctx.log(f"k={k:>11,} ({k / space.total:.0e}) E_raw={curve[-1]['energy_raw']['mean']:.3f} "
                f"E_sel={curve[-1]['energy_sel']['mean']:.3f} self={curve[-1]['self_real']['mean']} "
                f"xcos={curve[-1]['xfer_cos']['mean']:.3f} xreal={curve[-1]['xfer_real']['mean']:+.2f} "
                f"shuf={curve[-1]['xfer_shuf']['mean']:+.2f} enc={enc_share:.2f}")
    summary = {"model": L.name, "param_filter": cfg["param_filter"], "knobs": space.total, "direction": cfg["direction"],
               "train": len(train), "test": len(test), "curve": curve}
    ctx.write_json("summary.json", summary)
    ctx.progress(total_steps, total_steps)
    return {"model": L.name, "knobs": space.total, "ks": ks}
