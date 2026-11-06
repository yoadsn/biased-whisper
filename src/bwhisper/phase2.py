"""Phase 2 - discover upstream decision points.

For each verified Phase-1 decision point we compute, for two nudge types, the
parameter-space nudge that moves the digit-vs-cardinal log-odds at that
decision step to the target probability (1 - mistake_prob):

  to_digits   : log-odds -> +d*      to_cardinal : log-odds -> -d*      d* = ln((1-eps)/eps)

linear mode    : n = delta * g / |g|^2           (min-norm first-order solution; g = grad of objective)
iterative mode : damped Gauss-Newton steps on the true model until target reached; n = accumulated change

Top `sample_pct` entries (after optional per-tensor RMS normalisation) form
the sample's suspect set, saved sparsely. aggregate.py then finds the common
set, votes, and applies the FDR threshold.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import model as M
from .aggregate import Aggregator, concentration, jaccard_topk, pick_threshold
from .audio import load_audio
from .config import PHASE2_DEFAULTS, RUNS_DIR
from .corpus import store
from .jobs import Ctx, register
from .paramspace import ParamSpace, group_of, tower_of
from .phase1 import decision_points

NUDGES = ("to_digits", "to_cardinal")


def objective(L: M.Loaded, logits: torch.Tensor, pt: dict, kind: str) -> torch.Tensor:
    """Scalar log-odds in favour of the digit form at the decision step."""
    z = logits.float()
    if L.suppress_ids:
        z = z.index_fill(0, torch.tensor(L.suppress_ids, device=z.device), float("-inf"))
    D = L.digit_mask
    t, c = pt["digit_token"]["id"], pt["cardinal_token"]["id"]
    if kind == "pair":
        return z[t] - z[c]
    lse_d = torch.logsumexp(z[D], 0)
    if kind == "mass_vs_card":
        return lse_d - z[c]
    return lse_d - torch.logsumexp(z[~D], 0)


def resolve_kind(pt: dict, kind: str) -> str:
    if kind == "auto":
        return "seq_contrast" if pt.get("y_digit") else "mass"
    if kind == "seq_contrast" and not pt.get("y_digit"):
        raise ValueError("objective seq_contrast needs span-mode (v2) phase-1 points with y_digit/y_cardinal")
    return kind


def point_objective(L: M.Loaded, enc: torch.Tensor, pt: dict, kind: str) -> torch.Tensor:
    """Digit-style vs word-style log-odds for one decision point.

    seq_contrast: log P(Y_D) - log P(Y_C) summed over the divergent region (teacher forced), i.e. the
    whole style span incl. prefix, number, '%'/'אחוז' and agreement. Other kinds look at one step."""
    kind = resolve_kind(pt, kind)
    if kind == "seq_contrast":
        n0 = len(L.sot_seq)
        a, b = pt["div_digit"]
        c, e = pt["div_cardinal"]
        lp_d = M.seq_logprobs(L, enc, list(L.sot_seq) + pt["y_digit"][:b])
        lp_c = M.seq_logprobs(L, enc, list(L.sot_seq) + pt["y_cardinal"][:e])
        return lp_d[n0 + a - 1 : n0 + b - 1].sum() - lp_c[n0 + c - 1 : n0 + e - 1].sum()
    return objective(L, M.prefix_logits(L, enc, pt["prefix_ids"]), pt, kind)


def _eval_grad(L, space: ParamSpace, audio, pt, kind: str):
    space.zero_grad()
    enc = M.encode(L, audio, grad=space.has_encoder)
    with torch.enable_grad():
        d = point_objective(L, enc, pt, kind)
        d.backward()
    g = space.flat_grad()
    space.zero_grad()
    return float(d.detach()), g


def compute_nudges(L, space: ParamSpace, audio, pt, cfg) -> dict[str, dict]:
    eps = float(cfg["mistake_prob"])
    dstar = math.log((1 - eps) / eps)
    kind = cfg["objective"]
    d0, g = _eval_grad(L, space, audio, pt, kind)
    g2 = float(g.pow(2).sum()) + 1e-20
    targets = {"to_digits": dstar, "to_cardinal": -dstar}
    out = {}
    for name, tgt in targets.items():
        delta = tgt - d0
        info = {"d0": d0, "target": tgt, "delta": delta, "grad_norm": math.sqrt(g2), "reached": None}
        if cfg.get("skip_satisfied", True) and delta * tgt <= 0:
            # sample is already past the target in this nudge's direction: a nudge here would
            # point the *opposite* way and corrupt cross-sample sign agreement
            out[name] = {"n": None, "info": {**info, "skipped": "already_satisfied"}}
            continue
        if cfg["nudge_mode"] == "linear":
            n = g * (delta / g2)
        else:
            snap = space.snapshot()
            total = torch.zeros_like(g)
            d, gg = d0, g
            try:
                tol = float(cfg.get("iter_tol", 0.05))
                for _ in range(int(cfg["iter_steps"])):
                    if (tgt - d) * math.copysign(1, tgt) <= tol:
                        break
                    step = gg * ((tgt - d) / (float(gg.pow(2).sum()) + 1e-20)) * float(cfg["iter_lr"])
                    space.apply_flat(step)
                    total += step
                    d, gg = _eval_grad(L, space, audio, pt, kind)
                info["reached"] = (tgt - d) * math.copysign(1, tgt) <= tol
                info["d_final"] = d
            finally:
                space.restore_snapshot(snap)
            n = total
        out[name] = {"n": n, "info": info}
    return out


def sample_budget(space: ParamSpace, cfg: dict) -> int:
    if cfg.get("sample_k"):
        return int(min(space.total, int(cfg["sample_k"])))
    return max(1, int(math.ceil(space.total * float(cfg["sample_pct"]) / 100.0)))


def selection_score(n: torch.Tensor, space: ParamSpace, normalize: str):
    rms = space.rms_per_tensor(n)
    if normalize == "rms":
        return n.abs() / torch.from_numpy(np.repeat(rms, space.sizes)), rms
    return n.abs(), rms


def select_top(n: torch.Tensor, space: ParamSpace, k: int, normalize: str):
    score, rms = selection_score(n, space, normalize)
    top = torch.topk(score, k, sorted=False).indices.numpy()
    top.sort()
    return top, n[top].numpy().astype(np.float32), rms


@register("phase2", kind="gpu", defaults=PHASE2_DEFAULTS)
def run_phase2(ctx: Ctx, cfg: dict) -> dict:
    L = M.load_model(cfg["model"])
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    pts = decision_points(cfg["phase1_runs"], cfg["verdicts"], int(cfg["max_points"]))
    if not pts:
        raise RuntimeError("no usable decision points: pick phase-1 run(s) and verdicts with results")
    types = list(NUDGES)
    k = sample_budget(space, cfg)
    est_gb = len(pts) * len(types) * k * 8 / 1e9
    ctx.log(f"model={L.name} device={L.device} filter={cfg['param_filter']} tensors={len(space.names)} knobs={space.total:,}")
    ctx.log(f"{len(pts)} decision points; per-sample set={k:,} params; est. storage {est_gb:.2f} GB (limit {cfg['max_store_gb']})")
    if est_gb > float(cfg["max_store_gb"]):
        raise RuntimeError(f"estimated storage {est_gb:.1f} GB exceeds max_store_gb; lower sample_pct/max_points or narrow param_filter")
    ctx.write_json("space.json", {"signature": space.signature, "total": space.total, "filter": cfg["param_filter"],
                                  "names": space.names, "sizes": space.sizes.tolist(),
                                  "shapes": [list(x) for x in space.shapes]})
    vdir = ctx.dir / "vectors"
    for t in types:
        (vdir / t).mkdir(parents=True, exist_ok=True)
    total_steps = len(pts) + 4
    energy_acc = {t: np.zeros(len(space.names)) for t in types}
    n_ok = 0

    space.enable_grad()
    try:
        for i, pt in enumerate(pts):
            ctx.check()
            ctx.progress(i, total_steps, f"capture {i + 1}/{len(pts)}")
            s = store.get(pt["sample_id"])
            if s is None:
                continue
            try:
                audio = load_audio(s["audio_path"])
                nud = compute_nudges(L, space, audio, pt, cfg)
            except Exception as e:  # noqa: BLE001
                ctx.log(f"point {pt['sample_id']} failed: {e}")
                continue
            key = f"{pt['run_id']}__{pt['sample_id']}"
            for t, d in nud.items():
                if d["n"] is None:
                    ctx.append_jsonl("points.jsonl", {"key": key, "type": t, "sample_id": pt["sample_id"], "run_id": pt["run_id"], **d["info"]})
                    continue
                idx, val, rms = select_top(d["n"], space, k, cfg["normalize"])
                en = space.energy_per_tensor(d["n"])
                energy_acc[t] += en / max(en.sum(), 1e-30)
                np.savez(vdir / t / f"{key}.npz", idx=idx.astype(np.uint32 if space.total < 2**32 else np.int64), val=val, rms=rms, energy=en)
                ctx.append_jsonl("points.jsonl", {"key": key, "type": t, "sample_id": pt["sample_id"], "run_id": pt["run_id"], **d["info"]})
            n_ok += 1
            i0 = nud["to_digits"]["info"]
            need = [t for t in types if nud[t]["n"] is not None]
            ctx.log(f"{key}: d0={i0['d0']:+.2f} |g|={i0['grad_norm']:.3g} nudges={need}")
    finally:
        space.disable_grad()

    # ---------------- aggregation
    summary: dict = {"decision_points": n_ok, "knobs": space.total, "types": {}, "model": L.name, "signature": space.signature}
    if n_ok < int(cfg["group_size"]):
        raise RuntimeError(f"only {n_ok} decision points captured; need >= group_size")
    for ti, t in enumerate(types):
        ctx.check()
        files = sorted((vdir / t).glob("*.npz"))
        summary.setdefault("cohort_sizes", {})[t] = len(files)
        if len(files) < int(cfg["group_size"]):
            ctx.log(f"[{t}] only {len(files)} samples need this nudge (< group_size); skipping aggregation")
            summary["types"][t] = {"skipped": f"only {len(files)} samples needed this nudge", "n_candidates": 0, "threshold_votes": None}
            continue
        ctx.progress(len(pts) + ti * 2, total_steps, f"aggregate {t}")
        agg = Aggregator(files, space.total, space.tensor_ids, cfg["normalize"])
        common = agg.build_common_set(float(cfg["common_pct"]), max_k=cfg.get("common_k"))
        ctx.log(f"[{t}] union={common['union']:,} common={common['common']:,}")
        rounds = int(cfg["rounds"])
        real = agg.vote(rounds, int(cfg["group_size"]), float(cfg["agree_top_pct"]), int(cfg["seed"]),
                        on_round=lambda r: ctx.progress(len(pts) + ti * 2, total_steps, f"vote {t} round {r}/{rounds}"),
                        should_stop=ctx.should_stop)
        ctx.check()
        null = agg.vote(rounds, int(cfg["group_size"]), float(cfg["agree_top_pct"]), int(cfg["seed"]), null=True,
                        on_round=lambda r: ctx.progress(len(pts) + ti * 2 + 1, total_steps, f"null {t} round {r}/{rounds}"),
                        should_stop=ctx.should_stop)
        thr, fdr_at = pick_threshold(real["votes"], null["votes"], float(cfg["fdr"]))
        order = np.lexsort((-np.arange(len(agg.C)), real["votes"]))[::-1]  # votes desc
        if thr is None:
            cand_pos = np.zeros(0, dtype=np.int64)
        else:
            cand_pos = order[real["votes"][order] >= thr][: int(cfg["max_candidates"])]
        st = agg.stats(cand_pos)
        # order candidates by votes desc then |mean|
        o2 = np.lexsort((-np.abs(st["mean_all"]), -real["votes"][cand_pos]))
        cand_pos = cand_pos[o2]
        st = {k: v[o2] for k, v in st.items()}
        cand_flat = agg.C[cand_pos]
        np.savez(RUNS_DIR / ctx.job.id / f"candidates_{t}.npz", idx=cand_flat, votes=real["votes"][cand_pos],
                 null_votes=null["votes"][cand_pos], **st)
        np.savez(RUNS_DIR / ctx.job.id / f"votes_{t}.npz", C=agg.C, votes=real["votes"], null=null["votes"])
        by_group: dict[str, int] = {}
        by_tower: dict[str, int] = {}
        if len(cand_flat):
            tids = space.tensor_ids(cand_flat)
            for tid in tids:
                nm = space.names[int(tid)]
                by_group[group_of(nm)] = by_group.get(group_of(nm), 0) + 1
                by_tower[tower_of(nm)] = by_tower.get(tower_of(nm), 0) + 1
        en = energy_acc[t] / max(len(files), 1)
        en_group: dict[str, float] = {}
        for nm, e in zip(space.names, en):
            en_group[group_of(nm)] = en_group.get(group_of(nm), 0) + float(e)
        en_tower: dict[str, float] = {}
        for nm, e in zip(space.names, en):
            en_tower[tower_of(nm)] = en_tower.get(tower_of(nm), 0) + float(e)
        vmax = int(max(real["votes"].max(), null["votes"].max(), 1))
        bins = np.linspace(0, vmax, 41)
        summary["types"][t] = {
            **common,
            "rounds": real["rounds"],
            "vote_k": real["k"],
            "threshold_votes": thr,
            "fdr_at_threshold": fdr_at,
            "n_candidates": int(len(cand_flat)),
            "concentration": concentration(real["votes"]),
            "concentration_null": concentration(null["votes"]),
            "split_half_jaccard_top1pct": jaccard_topk(real["halves"][0], real["halves"][1]),
            "hist_real": np.histogram(real["votes"], bins)[0].tolist(),
            "hist_null": np.histogram(null["votes"], bins)[0].tolist(),
            "hist_edges": bins.tolist(),
            "candidates_by_group": dict(sorted(by_group.items(), key=lambda kv: -kv[1])[:60]),
            "candidates_by_tower": by_tower,
            "gradient_energy_by_group": dict(sorted(en_group.items(), key=lambda kv: -kv[1])[:40]),
            "gradient_energy_by_tower": en_tower,
            "top": [
                {**space.locate(int(f)), "votes": int(real["votes"][p]), "mean_all": float(st["mean_all"][j]),
                 "mean_present": float(st["mean_present"][j]), "median_present": float(st["median_present"][j]),
                 "std_present": float(st["std_present"][j]), "count": int(st["count"][j]),
                 "sign_consistency": float(st["sign_consistency"][j])}
                for j, (f, p) in enumerate(zip(cand_flat[:200], cand_pos[:200]))
            ],
        }
        ctx.log(f"[{t}] threshold={thr} fdr={fdr_at} candidates={len(cand_flat):,} "
                f"split-half-J={summary['types'][t]['split_half_jaccard_top1pct']:.2f} "
                f"gini real={summary['types'][t]['concentration']['gini']:.3f} null={summary['types'][t]['concentration_null']['gini']:.3f}")
    ctx.progress(total_steps, total_steps)
    ctx.write_json("summary.json", summary)
    return {k: v for k, v in summary.items() if k != "types"} | {
        "candidates": {t: v["n_candidates"] for t, v in summary["types"].items()},
        "threshold": {t: v["threshold_votes"] for t, v in summary["types"].items()},
    }
