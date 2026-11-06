"""Phase 2N - component-level attribution (neurons, heads, residual directions).

Discovery happens in activation space, where the units are functional (an MLP neuron, an
attention head, a residual-stream direction), and the result is translated back into weight
edits so the deliverable stays a weight adapter (see knowledge/plan/approach-v2.md).

Per decision point we run the objective of Phase 2 (`auto` = sequence contrast over the style
span for span-mode points) with hooks that keep every hooked activation's gradient, then

  attribution(unit) = sum over positions of activation * d(objective)/d(activation)

(gradient x activation: first-order estimate of how much the unit pushes toward the digit
style; positive = toward digits). Units:

  dec.L<i>.mlp     post-GELU neurons of decoder layer i (input of fc2)
  dec.L<i>.self    decoder self-attention heads (input of out_proj, summed per head)
  dec.L<i>.cross   decoder cross-attention heads
  enc.L<i>.mlp     encoder neurons (summed over audio frames)
  enc.L<i>.self    encoder self-attention heads

Two maps per point: `full` (all positions) and `decision` (decoder units at the decision step
only; encoder units are not position-specific and are copied). The residual-stream gradient at
the decision step is kept per decoder layer for a steering direction.

Statistics: one-sample t-test per unit across points + Benjamini-Hochberg FDR (primary), and the
owner's random-group voting against a permutation null (secondary, same code as Phase 2).

Weight translation (written in Phase 2's candidates_<type>.npz format, so Phase 3 runs unchanged):
  neuron j of layer L   ->  scale fc2.weight[:, j] by (1 -/+ gain)         (output gain)
  head h of layer L     ->  scale out_proj.weight[:, h*dh:(h+1)*dh]        (output gain)
  steering (best layer) ->  add alpha * v to fc2.bias of that layer  ==  activation addition
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
import torch

from . import model as M
from .aggregate import Aggregator, concentration, pick_threshold
from .audio import load_audio
from .corpus import store
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase1 import decision_points
from .phase2 import point_objective, resolve_kind

PHASE2N_DEFAULTS = {
    "model": None,
    "phase1_runs": [],
    "verdicts": ["cardinal", "human_good"],
    "max_points": 200,
    "objective": "auto",
    "towers": ["decoder", "encoder"],
    "map": "full",  # full | decision : which attribution map drives selection
    "fdr": 0.05,
    "top_neurons": 64,  # max significant MLP neurons turned into gain edits
    "top_heads": 8,
    "gain": 0.1,  # gain change per unit at Phase-3 strength 1
    "steer_target": 1.0,  # steering edit sized for this predicted mean |log-odds| change at strength 1
    "vote_rounds": 1000,
    "group_size": 3,
    "agree_top_pct": 5.0,
    "seed": 0,
    # Phase-3 compatibility: the weight space the edits live in
    "param_filter": "all_no_embed",
    "include_regex": "",
    "exclude_regex": "",
}


def _layers(L: M.Loaded, tower: str):
    return (L.model.model.decoder if tower == "decoder" else L.model.model.encoder).layers


def component_layout(L: M.Loaded, towers: list[str]) -> list[dict]:
    cfg = L.model.config
    out, off = [], 0
    for tower in ("decoder", "encoder"):
        if tower not in towers:
            continue
        t = "dec" if tower == "decoder" else "enc"
        heads = cfg.decoder_attention_heads if tower == "decoder" else cfg.encoder_attention_heads
        for i, layer in enumerate(_layers(L, tower)):
            sites = [("mlp", layer.fc1.out_features), ("self", heads)]
            if tower == "decoder":
                sites.append(("cross", heads))
            for kind, n in sites:
                out.append({"name": f"{t}.L{i:02d}.{kind}", "tower": tower, "layer": i, "kind": kind, "n": int(n), "offset": off})
                off += n
    return out


class Recorder:
    """Hooks that keep hooked activations (with grads) for one objective evaluation."""

    def __init__(self, L: M.Loaded, layout: list[dict]):
        self.L, self.layout = L, layout
        self.acts: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.resid: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        dec = L.model.model.decoder
        self.handles.append(dec.embed_tokens.register_forward_hook(lambda m, i, o: o.detach().requires_grad_(True)))
        for site in layout:
            layer = _layers(L, site["tower"])[site["layer"]]
            mod = {"mlp": layer.fc2, "self": layer.self_attn.out_proj,
                   "cross": getattr(layer, "encoder_attn", None) and layer.encoder_attn.out_proj}[site["kind"]]
            self.handles.append(mod.register_forward_pre_hook(self._keep(site["name"])))
        for i, layer in enumerate(dec.layers):
            self.handles.append(layer.register_forward_hook(self._keep_resid(i)))

    def _keep(self, name):
        def hook(mod, inp):
            x = inp[0]
            if x.requires_grad:
                x.retain_grad()
                self.acts[name].append(x)
        return hook

    def _keep_resid(self, i):
        def hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.requires_grad:
                x.retain_grad()
                self.resid[i].append(x)
        return hook

    def clear(self):
        self.acts.clear()
        self.resid.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def attribute_point(L, rec: Recorder, layout, audio, pt, kind, heads_dim: dict[str, int]):
    """Returns (full[N], decision[N], resid_grad[n_dec_layers, d_model], d0)."""
    rec.clear()
    n_units = layout[-1]["offset"] + layout[-1]["n"]
    with torch.enable_grad():
        f = M.features(L, audio).requires_grad_(True)
        enc = L.model.model.encoder(f).last_hidden_state
        d = point_objective(L, enc, pt, kind)
        d.backward()
    dec_pos = len(L.sot_seq) + int(pt["decision_index"]) - 1 if "decision_index" in pt else len(pt["prefix_ids"]) - 1
    full = np.zeros(n_units, dtype=np.float32)
    decision = np.zeros(n_units, dtype=np.float32)
    for site in layout:
        xs = rec.acts.get(site["name"], [])
        if not xs:
            continue
        o, n = site["offset"], site["n"]
        tot = torch.zeros(n)
        at_dec = torch.zeros(n)
        for j, x in enumerate(xs):
            if x.grad is None:
                continue
            ag = (x * x.grad)[0].float()  # [T, width]
            if site["kind"] != "mlp":
                ag = ag.reshape(ag.shape[0], n, heads_dim[site["tower"]]).sum(-1)
            tot += ag.sum(0).detach().cpu()
            if site["tower"] == "decoder" and j == 0 and dec_pos < ag.shape[0]:
                at_dec += ag[dec_pos].detach().cpu()
        full[o : o + n] = tot.numpy()
        decision[o : o + n] = (at_dec if site["tower"] == "decoder" else tot).numpy()
    n_dec = len(L.model.model.decoder.layers)
    resid = np.zeros((n_dec, L.model.config.d_model), dtype=np.float32)
    for i in range(n_dec):
        xs = rec.resid.get(i, [])
        if xs and xs[0].grad is not None and dec_pos < xs[0].shape[1]:
            resid[i] = xs[0].grad[0, dec_pos].float().cpu().numpy()
    return full, decision, resid, float(d.detach())


def bh_fdr(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries."""
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def unit_stats(mat: np.ndarray) -> dict[str, np.ndarray]:
    from scipy import stats

    n = mat.shape[0]
    mean = mat.mean(0)
    std = mat.std(0, ddof=1) + 1e-30
    t = mean / (std / math.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    sign = (np.sign(mat) == np.sign(mean)).mean(0)
    return {"mean": mean, "std": std, "t": t, "p": p, "sign_consistency": sign}


def _weight_edits(L, space: ParamSpace, layout, sel_units: list[tuple[dict, int, float]], gain: float, heads_dim):
    """Return (flat idx, delta) for 'to_cardinal' at strength 1 (to_digits is the negation)."""
    name_to_tensor = {n: i for i, n in enumerate(space.names)}
    idx_parts, val_parts = [], []
    for site, u, m in sel_units:
        pre = "model.decoder" if site["tower"] == "decoder" else "model.encoder"
        if site["kind"] == "mlp":
            wname = f"{pre}.layers.{site['layer']}.fc2.weight"
            cols = [u]
        else:
            attn = "self_attn" if site["kind"] == "self" else "encoder_attn"
            wname = f"{pre}.layers.{site['layer']}.{attn}.out_proj.weight"
            dh = heads_dim[site["tower"]]
            cols = list(range(u * dh, (u + 1) * dh))
        if wname not in name_to_tensor:
            continue
        t = name_to_tensor[wname]
        W = space.params[t].detach().float().cpu()  # [out, in]
        rows = np.arange(W.shape[0])
        for c in cols:
            idx_parts.append(space.starts[t] + rows * W.shape[1] + c)
            # neuron pushes toward digits (m > 0): reduce its gain for to_cardinal; boost if m < 0
            val_parts.append((-math.copysign(gain, m) * W[:, c]).numpy())
    if not idx_parts:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    idx = np.concatenate(idx_parts).astype(np.int64)
    val = np.concatenate(val_parts).astype(np.float32)
    o = np.argsort(idx)
    return idx[o], val[o]


def _save_candidates(path, idx, val):
    z = np.zeros(len(idx), np.int32)
    np.savez(path, idx=idx, votes=z, null_votes=z, mean_all=val, mean_present=val, median_present=val,
             std_present=np.zeros_like(val), count=z, sign_consistency=np.ones_like(val))


@register("phase2n", kind="gpu", defaults=PHASE2N_DEFAULTS)
def run_phase2n(ctx: Ctx, cfg: dict) -> dict:
    L = M.load_model(cfg["model"])
    layout = component_layout(L, cfg["towers"])
    n_units = layout[-1]["offset"] + layout[-1]["n"]
    mc = L.model.config
    heads_dim = {"decoder": mc.d_model // mc.decoder_attention_heads, "encoder": mc.d_model // mc.encoder_attention_heads}
    pts = decision_points(cfg["phase1_runs"], cfg["verdicts"], int(cfg["max_points"]))
    if len(pts) < 3:
        raise RuntimeError("need >= 3 decision points")
    kinds = {resolve_kind(p, cfg["objective"]) for p in pts}
    ctx.log(f"model={L.name} units={n_units:,} sites={len(layout)} points={len(pts)} objective={cfg['objective']}->{sorted(kinds)}")
    ctx.write_json("components.json", layout)

    rec = Recorder(L, layout)
    fulls, decs, resids, d0s, keys = [], [], [], [], []
    try:
        for i, pt in enumerate(pts):
            ctx.check()
            ctx.progress(i, len(pts) + 2, f"attribute {i + 1}/{len(pts)}")
            s = store.get(pt["sample_id"])
            try:
                full, dec, resid, d0 = attribute_point(L, rec, layout, load_audio(s["audio_path"]), pt, cfg["objective"], heads_dim)
            except Exception as e:  # noqa: BLE001
                ctx.log(f"point {pt['sample_id']} failed: {e}")
                continue
            fulls.append(full); decs.append(dec); resids.append(resid); d0s.append(d0)
            keys.append(f"{pt['run_id']}__{pt['sample_id']}")
            ctx.log(f"{keys[-1]}: d0={d0:+.2f} sum|attr|={np.abs(full).sum():.3g}")
    finally:
        rec.remove()
    F, D, R = np.stack(fulls), np.stack(decs), np.stack(resids)
    np.savez(ctx.dir / "attributions.npz", full=F, decision=D, resid_grad=R, d0=np.array(d0s), keys=np.array(keys))
    n = len(F)
    ctx.progress(n, n + 2, "statistics")

    X = F if cfg["map"] == "full" else D
    st = unit_stats(X)
    sig = bh_fdr(st["p"], float(cfg["fdr"]))
    ctx.log(f"BH-FDR {cfg['fdr']}: {int(sig.sum()):,} of {n_units:,} units significant")

    # owner's voting procedure on the same per-sample vectors (dense: every unit present)
    vdir = ctx.dir / "vectors"
    vdir.mkdir(exist_ok=True)
    site_of = np.zeros(n_units, dtype=np.int64)
    for si, site in enumerate(layout):
        site_of[site["offset"] : site["offset"] + site["n"]] = si
    files = []
    for k, x in zip(keys, X):
        rms = np.array([np.sqrt(np.mean(x[s["offset"] : s["offset"] + s["n"]] ** 2)) + 1e-30 for s in layout], np.float32)
        p = vdir / f"{k}.npz"
        np.savez(p, idx=np.arange(n_units, dtype=np.int64), val=x.astype(np.float32), rms=rms)
        files.append(p)
    agg = Aggregator(files, n_units, lambda idx: site_of[idx], normalize="rms")
    agg.build_common_set(100.0)
    real = agg.vote(int(cfg["vote_rounds"]), int(cfg["group_size"]), float(cfg["agree_top_pct"]), int(cfg["seed"]))
    null = agg.vote(int(cfg["vote_rounds"]), int(cfg["group_size"]), float(cfg["agree_top_pct"]), int(cfg["seed"]), null=True)
    vthr, vfdr = pick_threshold(real["votes"], null["votes"], float(cfg["fdr"]))

    # per-site profile: where does the decision live?
    profile = []
    for s in layout:
        sl = slice(s["offset"], s["offset"] + s["n"])
        profile.append({**{k: s[k] for k in ("name", "tower", "layer", "kind", "n")},
                        "abs_mean_attr": float(np.abs(st["mean"][sl]).sum()),
                        "net_mean_attr": float(st["mean"][sl].sum()),
                        "n_significant": int(sig[sl].sum()),
                        "max_abs_t": float(np.abs(st["t"][sl]).max())})
    unit_name = lambda u: (layout[site_of[u]], int(u - layout[site_of[u]]["offset"]))  # noqa: E731
    order = np.argsort(-np.abs(st["mean"]) * sig - np.abs(st["mean"]) * 1e-9)  # significant units by effect size first
    top_units = []
    for u in order[:300]:
        site, j = unit_name(u)
        top_units.append({"unit": int(u), "site": site["name"], "kind": site["kind"], "tower": site["tower"], "layer": site["layer"],
                          "index": j, "mean": float(st["mean"][u]), "std": float(st["std"][u]), "t": float(st["t"][u]),
                          "p": float(st["p"][u]), "significant": bool(sig[u]), "sign_consistency": float(st["sign_consistency"][u]),
                          "votes": int(real["votes"][u]), "null_votes": int(null["votes"][u])})

    # ---- weight translation -> Phase-3 candidate files
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    ctx.write_json("space.json", {"signature": space.signature, "total": space.total, "filter": cfg["param_filter"],
                                  "names": space.names, "sizes": space.sizes.tolist(), "shapes": [list(x) for x in space.shapes]})
    # significance gates, effect size ranks: a tiny-but-consistent unit is not a useful knob
    sel_neurons, sel_heads = [], []
    for u in np.argsort(-np.abs(st["mean"])):
        if not sig[u]:
            continue
        site, j = unit_name(u)
        if site["kind"] == "mlp" and len(sel_neurons) < int(cfg["top_neurons"]):
            sel_neurons.append((site, j, float(st["mean"][u])))
        elif site["kind"] != "mlp" and len(sel_heads) < int(cfg["top_heads"]):
            sel_heads.append((site, j, float(st["mean"][u])))
    made = {}
    for tag, sel in (("neurons", sel_neurons), ("heads", sel_heads)):
        idx, val = _weight_edits(L, space, layout, sel, float(cfg["gain"]), heads_dim)
        if len(idx):
            _save_candidates(ctx.dir / f"candidates_{tag}_to_cardinal.npz", idx, val)
            _save_candidates(ctx.dir / f"candidates_{tag}_to_digits.npz", idx, -val)
            made[tag] = {"units": len(sel), "params": int(len(idx))}

    # steering direction per decoder layer from residual gradients at the decision step
    Rn = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-30)
    v = Rn.mean(0)  # [layers, d]
    proj = np.einsum("nld,ld->nl", R, v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-30))  # g . v_hat per point/layer
    pm, ps = proj.mean(0), proj.std(0, ddof=1) + 1e-30
    layer_t = pm / (ps / math.sqrt(n))
    best = int(np.argmax(layer_t))
    steer = {"per_layer_t": layer_t.tolist(), "per_layer_mean_proj": pm.tolist(), "best_layer": best}
    bname = f"model.decoder.layers.{best}.fc2.bias"
    if bname in space.names:
        t = space.names.index(bname)
        vh = v[best] / (np.linalg.norm(v[best]) + 1e-30)
        alpha = float(cfg["steer_target"]) / float(max(pm[best], 1e-12))  # first-order: delta d = alpha * g.v_hat
        idx = (space.starts[t] + np.arange(len(vh))).astype(np.int64)
        _save_candidates(ctx.dir / "candidates_steer_to_cardinal.npz", idx, (-alpha * vh).astype(np.float32))
        _save_candidates(ctx.dir / "candidates_steer_to_digits.npz", idx, (alpha * vh).astype(np.float32))
        steer.update({"alpha": alpha, "param": bname})
        made["steer"] = {"layer": best, "params": int(len(idx)), "alpha": alpha}

    summary = {
        "model": L.name, "points": n, "units": n_units, "map": cfg["map"],
        "objective_kinds": sorted(kinds), "d0_mean": float(np.mean(d0s)),
        "significant_units": int(sig.sum()),
        "significant_by_kind": {k: int(sum(sig[s["offset"]:s["offset"] + s["n"]].sum() for s in layout if s["kind"] == k))
                                for k in ("mlp", "self", "cross")},
        "vote_threshold": vthr, "vote_fdr": vfdr,
        "vote_candidates": int((real["votes"] >= vthr).sum()) if vthr else 0,
        "vote_concentration": concentration(real["votes"]), "vote_concentration_null": concentration(null["votes"]),
        "profile": profile, "top_units": top_units, "steer": steer, "edits": made,
        "phase3_nudge_types": [f"{t}_{d}" for t in made for d in ("to_digits", "to_cardinal")],
    }
    ctx.write_json("summary.json", summary)
    ctx.progress(n + 2, n + 2)
    ctx.log(f"significant by kind {summary['significant_by_kind']}; edits {made}; steering best layer L{best} t={layer_t[best]:.1f}")
    return {"points": n, "significant_units": int(sig.sum()), "edits": made}
