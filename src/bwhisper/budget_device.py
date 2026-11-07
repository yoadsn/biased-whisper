"""Device-native budget sweeps with nested rankings and per-budget checkpoints."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .budget import BUDGET_DEFAULTS, _ms, eval_d
from .config import RUNS_DIR
from .corpus import store
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase1 import decision_points
from .phase2 import _eval_grad
from .phase2_capture import measured

EXECUTION_KEYS = {"budget_backend", "resume_run", "release_unused_models", "release_model_after_run", "restart_of", "benchmark_points"}
METRICS = ("energy_raw", "energy_sel", "self_real", "xfer_cos", "xfer_real", "xfer_shuf")


def choose_points(cfg, total):
    pts = decision_points(cfg["phase1_runs"], cfg["verdicts"])
    rng = np.random.default_rng(int(cfg["seed"]))
    rng.shuffle(pts)
    ntr, nte = int(cfg["train_points"]), int(cfg["test_points"])
    if len(pts) < ntr + nte:
        ntr = max(3, int(len(pts) * 2 / 3))
        nte = len(pts) - ntr
    train, test = pts[:ntr], pts[ntr:ntr + nte]
    if not train or not test:
        raise ValueError("need nonempty training and test decision points")
    fractions = [float(f) for f in cfg["fractions"]]
    if not fractions or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError("fractions must be finite and in (0,1]")
    ks = sorted({max(1, int(round(f * total))) for f in fractions})
    return train, test, ks, rng


def point_key(pt):
    return f"{pt['run_id']}__{pt['sample_id']}"


def memory_guard(vector, max_k):
    if vector.device.type != "cuda":
        return
    device = vector.device
    free, _ = torch.cuda.mem_get_info(device)
    reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    # Conservative scratch allowance for dense scores, top-k/sort and prefix
    # arrays. Actual allocator peaks are checked by the real-grid benchmark.
    scratch = vector.numel() * 12 + max_k * 40
    if scratch > 0.9 * (free + reusable):
        raise RuntimeError(f"ranking scratch estimate {scratch / 1e9:.1f} GB exceeds available device memory; "
                           "free GPU memory or explicitly revise the grid (no automatic truncation)")


@torch.no_grad()
def raw_energy_curve(g, ks):
    squared = g.square()
    total = float(squared.sum()) + 1e-30
    if not math.isfinite(total):
        raise ValueError("nonfinite gradient energy")
    top = torch.topk(squared, max(ks), sorted=True).values
    del squared
    cumulative = top.double().cumsum(0)
    positions = torch.tensor([k - 1 for k in ks], device=g.device)
    energies = cumulative[positions].cpu().numpy() / total
    return energies, total


@torch.no_grad()
def ranked_support(g, space, normalize, max_k):
    score = torch.empty_like(g)
    if normalize == "none":
        torch.abs(g, out=score)
    elif normalize == "rms":
        rms = torch.stack([g[int(s):int(s + n)].square().mean().sqrt()
                           for s, n in zip(space.starts, space.sizes)])
        rms = torch.where(rms == 0, torch.ones_like(rms), rms)
        for i, (s, n) in enumerate(zip(space.starts, space.sizes)):
            sl = slice(int(s), int(s + n))
            torch.div(g[sl].abs(), rms[i], out=score[sl])
    else:
        raise ValueError("normalize must be none or rms")
    return torch.topk(score, max_k, sorted=True).indices


def sorted_support(space, ranked, k):
    idx = ranked[:k].sort().values
    return idx, space.sparse_bounds(idx, validate=False)


def apply_eval(L, space, audio, pt, kind, support, values):
    idx, bounds = support
    undo = space.add_sparse_device(idx, values, bounds)
    try:
        return eval_d(L, audio, pt, kind)
    finally:
        space.undo_sparse(undo)


def make_plan(cfg, space, model, train, test, ks):
    return {"schema": 1, "model": model, "signature": space.signature, "knobs": space.total,
            "science_config": {k: cfg[k] for k in BUDGET_DEFAULTS if k not in EXECUTION_KEYS and k != "model"},
            "train": [point_key(p) for p in train], "test": [point_key(p) for p in test], "ks": ks}


class Journal:
    def __init__(self, ctx, cfg, plan):
        self.ctx, self.plan, self.records, self.source = ctx, plan, {}, None
        rid = cfg.get("resume_run")
        if rid:
            if not isinstance(rid, str) or Path(rid).name != rid or rid in (".", ".."):
                raise ValueError("invalid resume_run")
            self.source = RUNS_DIR / rid
            state = json.loads((self.source / "job.json").read_text())
            if state["type"] != "budget_sweep" or state["status"] in ("running", "queued"):
                raise ValueError("resume source must be a stopped budget sweep")
            if not (self.source / "plan.json").exists():
                raise ValueError("legacy sweep has no resumable metrics; restart without resume_run")
            if json.loads((self.source / "plan.json").read_text()) != plan:
                raise ValueError("resume plan/configuration/model/point order mismatch")
            metrics_path = self.source / "measurements.jsonl"
            lines = metrics_path.read_text().splitlines(keepends=True) if metrics_path.exists() else []
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise ValueError("corrupt checkpoint record") from None
                    continue  # incomplete final append after a process crash
                self._validate(row)
                self.put({**row, "reused_from": rid})
        ctx.write_json("plan.json", plan)

    @staticmethod
    def key(row):
        return row["phase"], row["point"], int(row["k"])

    def _validate(self, row):
        phase = row.get("phase")
        expected = self.plan["test"] if phase == "test" else self.plan["train"]
        if phase not in ("train", "calibration", "test") or row.get("point") not in expected:
            raise ValueError("checkpoint point/phase mismatch")
        if (phase == "calibration" and row.get("k") != 0) or (phase != "calibration" and row.get("k") not in self.plan["ks"]):
            raise ValueError("checkpoint budget mismatch")
        if phase == "test" and "rng_after" not in row:
            raise ValueError("test checkpoint missing NumPy control RNG state")
        if not isinstance(row.get("metrics"), dict):
            raise ValueError("checkpoint missing metrics")
        expected_metrics = ({"energy_raw", "energy_sel", "self_real"} if phase == "train" else
                            {"xfer_cos", "xfer_real", "xfer_shuf"} if phase == "test" else
                            {str(k) for k in self.plan["ks"]})
        if set(row["metrics"]) != expected_metrics:
            raise ValueError("checkpoint metric schema mismatch")
        for name, value in row["metrics"].items():
            if value is None and name == "self_real":
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("nonfinite checkpoint metric")

    def get(self, phase, pt, k):
        return self.records.get((phase, point_key(pt), k))

    def put(self, row):
        self._validate(row)
        key = self.key(row)
        if key in self.records:
            raise ValueError("duplicate checkpoint record")
        self.ctx.append_jsonl("measurements.jsonl", row)
        self.records[key] = row

    def load_consensus(self, device):
        if self.source is None or not (self.source / "consensus.pt").exists():
            return None
        if not all(("train", key, k) in self.records for key in self.plan["train"] for k in self.plan["ks"]):
            return None
        value = torch.load(self.source / "consensus.pt", map_location=device, weights_only=True)
        if value.shape != (self.plan["knobs"],) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError("invalid consensus checkpoint")
        destination = self.ctx.dir / "consensus.pt"
        try:
            os.link(self.source / "consensus.pt", destination)
        except OSError:
            shutil.copy2(self.source / "consensus.pt", destination)
        return value


def checkpoint_consensus(ctx, consensus):
    tmp = ctx.dir / "consensus.pt.tmp"
    torch.save(consensus.detach().cpu(), tmp)
    tmp.replace(ctx.dir / "consensus.pt")


def curve_summary(cfg, space, L, train, test, ks, journal, supports, consensus, scales):
    rows = {k: {name: [] for name in METRICS} for k in ks}
    for row in journal.records.values():
        if row["phase"] in ("train", "test"):
            for name, value in row["metrics"].items():
                rows[row["k"]][name].append(value)
    curve = []
    encoder = [n.startswith("model.encoder.") for n in space.names]
    for k in ks:
        idx, bounds = supports[k]
        enc_count = sum((hi - lo) for t, (lo, hi) in enumerate(zip(bounds, bounds[1:])) if bool(encoder[t]))
        curve.append({"k": k, "fraction": k / space.total, "bytes_per_sample": k * 8,
                      "consensus_encoder_share": enc_count / k, "xfer_edit_l2": abs(scales[k]),
                      "xfer_edit_rms_per_param": abs(scales[k]) / math.sqrt(k),
                      **{name: _ms(values) for name, values in rows[k].items()}})
    return {"model": L.name, "param_filter": cfg["param_filter"], "knobs": space.total,
            "direction": cfg["direction"], "train": len(train), "test": len(test), "curve": curve,
            "budget_backend": "device", "checkpoint_records": len(journal.records)}


def run_device_budget(ctx: Ctx, cfg: dict):
    L = M.load_model(cfg["model"])
    if cfg.get("release_unused_models", True):
        ctx.log(f"released {M.unload_unused(L)} unused cached model replicas")
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    train, test, ks, rng = choose_points(cfg, space.total)
    plan = make_plan(cfg, space, L.name, train, test, ks)
    journal = Journal(ctx, cfg, plan)
    reused = len(journal.records)
    ctx.log(f"device budget model={L.name} device={L.device} knobs={space.total:,} "
            f"train={len(train)} test={len(test)} ks={ks} reused_metrics={reused}")
    total_steps = 2 * len(train) + len(test) + (len(train) + 2 * len(test)) * len(ks) + 2
    done = 0
    def progress(message, steps=0):
        nonlocal done
        done += steps
        ctx.progress(done, total_steps, message)
    audio = {}
    def get_audio(pt):
        sid = pt["sample_id"]
        if sid not in audio:
            audio[sid] = load_audio(store.get(sid)["audio_path"])
        return audio[sid]
    target = math.log((1 - float(cfg["mistake_prob"])) / float(cfg["mistake_prob"]))
    target *= 1 if cfg["direction"] == "to_digits" else -1
    consensus = journal.load_consensus(L.device)
    loaded_consensus = consensus is not None
    if consensus is None:
        consensus = torch.zeros(space.total, device=L.device)
    dots = {k: [] for k in ks}
    space.enable_grad()
    try:
        for i, pt in enumerate(train):
            ctx.check()
            progress(f"train {i + 1}/{len(train)} gradient")
            complete = all(journal.get("train", pt, k) for k in ks)
            if loaded_consensus:
                progress(f"reuse train {i + 1}/{len(train)}", 1 + len(ks))
                continue
            timings = {}
            d0, g = _eval_grad(L, space, get_audio(pt), pt, cfg["objective"], gradient_device=L.device, timings=timings)
            progress(f"train {i + 1}/{len(train)} gradient complete", 1)
            if not complete:
                memory_guard(g, max(ks))
                ctx.check()
                with measured(timings, "raw_energy_s", L.device):
                    raw, total = raw_energy_curve(g, ks)
                ctx.check()
                with measured(timings, "ranking_s", L.device):
                    ranked = ranked_support(g, space, cfg["normalize"], max(ks))
                for n, k in enumerate(ks):
                    ctx.check()
                    if journal.get("train", pt, k):
                        progress(f"reuse train {i + 1} k={k:,}", 1)
                        continue
                    progress(f"train {i + 1}/{len(train)} k={k:,} self effect")
                    stages = {}
                    with measured(stages, "support_sort_s", L.device):
                        support = sorted_support(space, ranked, k)
                    with measured(stages, "edit_eval_undo_s", L.device):
                        gs = g[support[0]]
                        energy = float(gs.square().sum())
                        delta = target - d0
                        effect = None
                        if energy > 0 and delta != 0:
                            values = gs * (delta / energy)
                            effect = (apply_eval(L, space, get_audio(pt), pt, cfg["objective"], support, values) - d0) / delta
                            del values
                    journal.put({"phase": "train", "point": point_key(pt), "k": k,
                                 "metrics": {"energy_raw": float(raw[n]), "energy_sel": energy / total, "self_real": effect},
                                 "timings": stages})
                    del support, gs
                    progress(f"train {i + 1}/{len(train)} k={k:,} saved", 1)
                del ranked
            else:
                progress(f"reuse train metrics {i + 1}/{len(train)}", len(ks))
            consensus += g / (g.norm() + 1e-30)
            del g
            ctx.append_jsonl("point_timings.jsonl", {"phase": "train", "point": point_key(pt), **timings})
        if not loaded_consensus:
            consensus /= len(train)
            ctx.check()
            progress("saving consensus checkpoint")
            checkpoint_consensus(ctx, consensus)
        ctx.check()
        progress("ranking consensus once for all budgets")
        with measured(timings := {}, "consensus_ranking_s", L.device):
            memory_guard(consensus, max(ks))
            ranked = ranked_support(consensus, space, cfg["normalize"], max(ks))
            supports = {}
            for k in ks:
                ctx.check()
                supports[k] = sorted_support(space, ranked, k)
        del ranked
        ctx.append_jsonl("point_timings.jsonl", {"phase": "consensus", **timings})
        progress("consensus supports ready", 1)
        norms = {k: float(consensus[supports[k][0]].norm()) for k in ks}
        for i, pt in enumerate(train):
            ctx.check()
            progress(f"calibration {i + 1}/{len(train)}")
            record = journal.get("calibration", pt, 0)
            if record is None:
                timings = {}
                _, g = _eval_grad(L, space, get_audio(pt), pt, cfg["objective"], gradient_device=L.device, timings=timings)
                with measured(timings, "calibration_dots_s", L.device):
                    values = {str(k): float(g[supports[k][0]] @ consensus[supports[k][0]]) / (norms[k] + 1e-30) for k in ks}
                del g
                record = {"phase": "calibration", "point": point_key(pt), "k": 0, "metrics": values, "timings": timings}
                journal.put(record)
            for k in ks:
                dots[k].append(record["metrics"][str(k)])
            progress(f"calibration {i + 1}/{len(train)} saved", 1)
        scales = {k: float(cfg["xfer_target"]) / float(np.mean(dots[k])) if abs(float(np.mean(dots[k]))) > 1e-12 else 0.0 for k in ks}
        ctx.write_json("scales.json", {str(k): scales[k] for k in ks})
        for i, pt in enumerate(test):
            ctx.check()
            progress(f"test {i + 1}/{len(test)} gradient")
            complete = all(journal.get("test", pt, k) for k in ks)
            if not complete:
                timings = {}
                d0, g = _eval_grad(L, space, get_audio(pt), pt, cfg["objective"], gradient_device=L.device, timings=timings)
                gn = float(g.norm()) + 1e-30
                ctx.append_jsonl("point_timings.jsonl", {"phase": "test", "point": point_key(pt), **timings})
            progress(f"test {i + 1}/{len(test)} gradient ready", 1)
            for k in ks:
                ctx.check()
                record = journal.get("test", pt, k)
                if record is not None:
                    rng.bit_generator.state = record["rng_after"]
                    progress(f"reuse test {i + 1} k={k:,}", 2)
                    continue
                stages = {}
                idx, _ = supports[k]
                with measured(stages, "projection_s", L.device):
                    ck = consensus[idx]
                    cn = norms[k]
                    cosine = float(g[idx] @ ck) / (gn * (cn + 1e-30))
                    values = ck / cn * scales[k] if cn > 0 else torch.zeros_like(ck)
                    del ck
                ctx.check()
                progress(f"test {i + 1}/{len(test)} k={k:,} real edit")
                with measured(stages, "real_eval_undo_s", L.device):
                    real = apply_eval(L, space, get_audio(pt), pt, cfg["objective"], supports[k], values) - d0
                progress(f"test {i + 1}/{len(test)} k={k:,} shuffle", 1)
                ctx.check()
                # Preserve the exact NumPy RNG/control ordering of the reference.
                # Only this selected control payload visits CPU, not full gradients.
                with measured(stages, "numpy_shuffle_transfer_s", L.device):
                    shuffled = torch.from_numpy(rng.permutation(values.cpu().numpy())).to(L.device)
                with measured(stages, "shuffled_eval_undo_s", L.device):
                    control = apply_eval(L, space, get_audio(pt), pt, cfg["objective"], supports[k], shuffled) - d0
                del values, shuffled
                journal.put({"phase": "test", "point": point_key(pt), "k": k,
                             "metrics": {"xfer_cos": cosine, "xfer_real": real, "xfer_shuf": control},
                             "rng_after": rng.bit_generator.state, "timings": stages})
                progress(f"test {i + 1}/{len(test)} k={k:,} saved", 1)
            if not complete:
                del g
        summary = curve_summary(cfg, space, L, train, test, ks, journal, supports, consensus, scales)
        summary["reused_checkpoint_records"] = reused
        ctx.write_json("summary.json", summary)
        progress("done", 1)
        assert done == total_steps
        return {"model": L.name, "knobs": space.total, "ks": ks, "budget_backend": "device",
                "checkpoint_records": len(journal.records), "reused_checkpoint_records": reused}
    finally:
        space.disable_grad()


@register("budget_benchmark", kind="gpu", defaults=BUDGET_DEFAULTS)
def benchmark(ctx: Ctx, cfg: dict):
    """Exercise the real largest grid endpoint without rerunning huge CPU sorts."""
    L = M.load_model(cfg["model"])
    M.unload_unused(L)
    space = ParamSpace(L.model, cfg["param_filter"], cfg["include_regex"], cfg["exclude_regex"])
    train, _, ks, rng = choose_points(cfg, space.total)
    pt = train[0]
    audio = load_audio(store.get(pt["sample_id"])["audio_path"])
    if str(L.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(L.device)
    space.enable_grad()
    timings, checks = {}, {}
    before = None
    try:
        d0, g = _eval_grad(L, space, audio, pt, cfg["objective"], gradient_device=L.device, timings=timings)
        reserve = torch.zeros_like(g)  # include the full sweep's live consensus footprint
        memory_guard(g, max(ks))
        with measured(timings, "raw_energy_s", L.device):
            energy, _ = raw_energy_curve(g, ks)
        checks["energy_monotone"] = bool(np.all(np.diff(energy) >= -1e-7) and np.all(np.isfinite(energy)))
        with measured(timings, "ranking_s", L.device):
            ranked = ranked_support(g, space, cfg["normalize"], max(ks))
        before = space.snapshot()  # exact restoration check on every selected weight
        effects = []
        for k in sorted({min(ks), max(ks)}):
            ctx.check()
            ctx.progress(0, 2, f"benchmark real largest-grid edit k={k:,}")
            with measured(stages := {}, "support_sort_s", L.device):
                support = sorted_support(space, ranked, k)
            values = g[support[0]]
            es = float(values.square().sum()) + 1e-30
            target = math.log((1 - float(cfg["mistake_prob"])) / float(cfg["mistake_prob"]))
            target *= 1 if cfg["direction"] == "to_digits" else -1
            values = values * ((target - d0) / es)
            with measured(stages, "edit_eval_undo_s", L.device):
                real = apply_eval(L, space, audio, pt, cfg["objective"], support, values)
            with measured(stages, "numpy_shuffle_transfer_s", L.device):
                shuffled = torch.from_numpy(rng.permutation(values.cpu().numpy())).to(L.device)
            with measured(stages, "shuffled_eval_undo_s", L.device):
                control = apply_eval(L, space, audio, pt, cfg["objective"], support, shuffled)
            checks[f"restored_k{k}"] = all(torch.equal(a, b) for a, b in zip(before, space.params))
            checks[f"finite_k{k}"] = math.isfinite(real) and math.isfinite(control)
            effects.append({"k": k, "timings": stages, "real_d": real, "control_d": control})
            ctx.log(f"k={k:,}: {stages}; model restored={checks[f'restored_k{k}']}")
            del values, shuffled, support
        summary = {"model": L.name, "device": L.device, "sample_id": pt["sample_id"], "ks": ks,
                   "timings": timings, "effects": effects, "checks": checks, "passed": all(checks.values()),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(L.device) if str(L.device).startswith("cuda") else None}
        ctx.write_json("benchmark.json", summary)
        ctx.progress(2, 2, "benchmark complete")
        return summary
    finally:
        if before is not None:
            space.restore_snapshot(before)
        space.disable_grad()
