"""Device-native sparse capture. No dense parameter vector leaves its device."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .config import PHASE2_DEFAULTS
from .jobs import Ctx, register
from .paramspace import ParamSpace


@contextmanager
def measured(timings: dict | None, name: str, device):
    if timings is None:
        yield
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[name] = time.perf_counter() - start


@torch.no_grad()
def sparse_linear(g: torch.Tensor, space: ParamSpace, d0: float, cfg: dict, k: int) -> dict:
    """Rank once for both scalar-multiple linear nudges; export sparse arrays only.

    RMS/energy are reduced per tensor without a dense repeated scale vector.
    Floating-point reductions can differ from CPU; benchmark support boundaries
    and exported values before resuming legacy captures.
    """
    if g.ndim != 1 or g.numel() != space.total or not 1 <= k <= space.total:
        raise ValueError("invalid gradient or selection budget")
    if cfg["normalize"] not in ("none", "rms"):
        raise ValueError("normalize must be none or rms")
    energy = torch.stack([g[int(s):int(s + n)].square().sum()
                          for s, n in zip(space.starts, space.sizes)])
    sizes = torch.as_tensor(space.sizes, device=g.device, dtype=g.dtype)
    rms_raw = (energy / sizes).sqrt()
    g2_raw = float(energy.sum())
    if not math.isfinite(g2_raw) or not math.isfinite(d0):
        raise ValueError("nonfinite objective or gradient")
    g2 = g2_raw + 1e-20
    dstar = math.log((1 - float(cfg["mistake_prob"])) / float(cfg["mistake_prob"]))
    targets = {"to_digits": dstar, "to_cardinal": -dstar}
    result = {}
    active = []
    for name, target in targets.items():
        delta = target - d0
        info = {"d0": d0, "target": target, "delta": delta, "grad_norm": math.sqrt(g2), "reached": None}
        if cfg.get("skip_satisfied", True) and delta * target <= 0:
            result[name] = {"sparse": None, "info": {**info, "skipped": "already_satisfied"}}
        else:
            active.append((name, delta / g2, info))
    if not active:
        return result
    score = torch.empty_like(g)
    if cfg["normalize"] == "rms":
        denominator = torch.where(rms_raw == 0, torch.ones_like(rms_raw), rms_raw)
        for i, (s, n) in enumerate(zip(space.starts, space.sizes)):
            seg = slice(int(s), int(s + n))
            torch.div(g[seg].abs(), denominator[i], out=score[seg])
    else:
        torch.abs(g, out=score)
    top = torch.topk(score, k, sorted=False).indices.sort().values
    del score
    idx = top.cpu().numpy()
    selected = g.index_select(0, top)
    for name, scale, info in active:
        # A zero scale has an all-tied zero nudge. Any k indices have the same
        # score; use the shared support and retain the exact zero payload.
        rms = rms_raw * abs(scale)
        rms = torch.where(rms == 0, torch.ones_like(rms), rms)
        result[name] = {"info": info, "sparse": {
            "idx": idx, "val": (selected * scale).cpu().numpy(),
            "rms": rms.cpu().numpy(), "energy": (energy.double() * (scale * scale)).cpu().numpy(),
        }}
    return result


def capture_sparse_linear(L, space, audio, pt, cfg, timings=None):
    from .phase2 import _eval_grad

    d0, g = _eval_grad(L, space, audio, pt, cfg["objective"], gradient_device=L.device, timings=timings)
    with measured(timings, "selection_export_s", L.device):
        return sparse_linear(g, space, d0, cfg, int(cfg["_sample_budget"]))


def _reference(g, space, d0, cfg, k):
    from .phase2 import select_top

    g2 = float(g.square().sum()) + 1e-20
    dstar = math.log((1 - float(cfg["mistake_prob"])) / float(cfg["mistake_prob"]))
    result = {}
    for name, target in (("to_digits", dstar), ("to_cardinal", -dstar)):
        delta = target - d0
        if cfg.get("skip_satisfied", True) and delta * target <= 0:
            result[name] = {"sparse": None}
            continue
        n = g * (delta / g2)
        idx, val, rms = select_top(n, space, k, cfg["normalize"])
        result[name] = {"sparse": {"idx": idx, "val": val, "rms": rms, "energy": space.energy_per_tensor(n)}}
    return result


def compare_sparse(reference: dict, actual: dict, g: torch.Tensor, space, cfg, d0: float) -> dict:
    """Allow only tiny floating-point changes at the selection boundary, not rank drift."""
    reports = {}
    for name, ref in reference.items():
        expected, got = ref["sparse"], actual[name]["sparse"]
        if expected is None or got is None:
            reports[name] = {"passed": expected is None and got is None, "skipped": True}
            continue
        common, ri, ai = np.intersect1d(expected["idx"], got["idx"], return_indices=True)
        overlap = len(common) / max(1, len(expected["idx"]))
        values_ok = np.allclose(expected["val"][ri], got["val"][ai], rtol=2e-5, atol=1e-12)
        rms_ok = np.allclose(expected["rms"], got["rms"], rtol=2e-5, atol=1e-12)
        energy_ok = np.allclose(expected["energy"], got["energy"], rtol=2e-5, atol=1e-20)
        boundary_ok = True
        changed = np.setxor1d(expected["idx"], got["idx"])
        if len(changed):
            target = math.log((1 - float(cfg["mistake_prob"])) / float(cfg["mistake_prob"]))
            target *= 1 if name == "to_digits" else -1
            scale = (target - d0) / (float(g.square().sum()) + 1e-20)
            indices = np.concatenate([expected["idx"], changed])
            scores = (g[indices] * scale).abs().numpy()
            if cfg["normalize"] == "rms":
                scores /= expected["rms"][space.tensor_ids(indices)]
            cutoff = scores[:len(expected["idx"])].min()
            boundary_ok = bool(np.all(np.abs(scores[len(expected["idx"]):] - cutoff) <= 2e-5 * abs(cutoff) + 1e-12))
        reports[name] = {"passed": bool(overlap >= 0.9999 and values_ok and rms_ok and energy_ok and boundary_ok),
                         "support_overlap": overlap, "changed_indices": len(changed),
                         "values_close": bool(values_ok), "rms_close": bool(rms_ok),
                         "energy_close": bool(energy_ok), "boundary_close": boundary_ok}
    return reports


@register("phase2_benchmark", kind="gpu", defaults={**PHASE2_DEFAULTS, "benchmark_points": 2})
def benchmark(ctx: Ctx, cfg: dict) -> dict:
    from .corpus import store
    from .phase1 import decision_points
    from .phase2 import _eval_grad, sample_budget

    if cfg["nudge_mode"] != "linear":
        raise ValueError("device selection benchmark is for linear nudges")
    L = M.load_model(cfg["model"])
    released = M.unload_unused(L)
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    pts = decision_points(cfg["phase1_runs"], cfg["verdicts"], int(cfg["benchmark_points"]))
    if not pts:
        raise ValueError("no benchmark decision points")
    k = sample_budget(space, cfg)
    rows = []
    space.enable_grad()
    try:
        for i, pt in enumerate(pts):
            ctx.check()
            ctx.progress(i, len(pts), "benchmark gradient and sparse selection")
            timings = {}
            audio = load_audio(store.get(pt["sample_id"])["audio_path"])
            d0, g = _eval_grad(L, space, audio, pt, cfg["objective"], gradient_device=L.device, timings=timings)
            with measured(timings, "device_selection_s", L.device):
                actual = sparse_linear(g, space, d0, cfg, k)
            with measured(timings, "dense_cpu_export_s", L.device):
                cpu = g.cpu()
            del g
            with measured(timings, "reference_selection_s", "cpu"):
                reference = _reference(cpu, space, d0, cfg, k)
            parity = compare_sparse(reference, actual, cpu, space, cfg, d0)
            row = {"sample_id": pt["sample_id"], "d0": d0, "timings": timings, "parity": parity,
                   "selection_speedup": timings["reference_selection_s"] / max(timings["device_selection_s"], 1e-9)}
            ctx.append_jsonl("benchmark.jsonl", row)
            ctx.log(f"sample={pt['sample_id']} device selection={timings['device_selection_s']:.3f}s "
                    f"CPU reference={timings['reference_selection_s']:.3f}s parity={all(p['passed'] for p in parity.values())}")
            rows.append(row)
            del cpu, reference, actual
    finally:
        space.disable_grad()
    summary = {"model": L.name, "device": L.device, "parameters": space.total,
               "dense_gradient_bytes": space.total * 4, "released_cached_models": released,
               "passed": all(p["passed"] for r in rows for p in r["parity"].values()), "rows": rows}
    ctx.write_json("benchmark.json", summary)
    ctx.progress(len(pts), len(pts), "benchmark complete")
    return summary
