"""Same-condition batched Phase 3 and its bounded reference-parity benchmark."""
from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .config import PHASE3_DEFAULTS, RUNS_DIR
from .corpus import store
from .jobs import Ctx, register
from .paramspace import ParamSpace
from .phase1 import decision_points
from .phase2 import point_objective
from .phase2_capture import measured
from .phase3 import _metrics
from .phase3_inference import FeatureBank, greedy_tokens, probe_values


@contextmanager
def stage(timings, name, device):
    part = {}
    with measured(part, name, device):
        yield
    timings[name] = timings.get(name, 0.0) + part[name]


def prepare(ctx, cfg, benchmark=False):
    if int(cfg["batch_size"]) < 1 or int(cfg["max_new_tokens"]) < 0:
        raise ValueError("batch_size must be positive and max_new_tokens nonnegative")
    if any(c not in ("shuffled_values", "random_indices") for c in cfg["controls"]):
        raise ValueError("unknown Phase 3 control")
    p2dir = RUNS_DIR / cfg["phase2_run"]
    p2cfg = json.loads((p2dir / "job.json").read_text())["config"]
    L = M.load_model(cfg["model"] or p2cfg.get("model"))
    M.unload_unused(L)
    space = ParamSpace(L.model, p2cfg["param_filter"], p2cfg["include_regex"], p2cfg["exclude_regex"])
    if json.loads((p2dir / "space.json").read_text())["signature"] != space.signature:
        raise RuntimeError("parameter space signature mismatch: phase-2 candidates were computed on a different model/filter")
    pts = decision_points(p2cfg["phase1_runs"], p2cfg["verdicts"], int(p2cfg["max_points"]),
                          sample_ids=p2cfg.get("sample_ids"))
    rng = np.random.default_rng(int(cfg["seed"]))
    cands = {}
    for name in cfg["nudge_types"]:
        path = p2dir / f"candidates_{name}.npz"
        if not path.exists():
            ctx.log(f"no candidates for {name}; skipping")
            continue
        with np.load(path) as z:
            if not len(z["idx"]):
                continue
            limit = cfg.get("candidate_limit")
            sl = slice(0, int(limit)) if limit else slice(None)
            idx, val = z["idx"][sl], z[cfg["stat"]][sl]
            order = np.argsort(idx)
            cands[name] = idx[order].astype(np.int64), val[order].astype(np.float32)
            ctx.log(f"{name}: {len(idx):,} candidates, |delta|_2={np.linalg.norm(val):.4g}")
    if not cands:
        raise RuntimeError("no candidate sets to test")
    if benchmark:
        # Benchmarks consume discovery/in-sample audio only, never the frozen
        # recording-held-out evaluation cohort.
        count = int(cfg["benchmark_samples"])
        if count < 1:
            raise ValueError("benchmark_samples must be positive")
        unique = list(dict.fromkeys(pt["sample_id"] for pt in pts))[:count]
        samples = [store.get(sid) for sid in unique]
        probe = pts[:count]
    else:
        eval_ids = cfg.get("eval_sample_ids")
        if eval_ids == []:
            raise ValueError("explicit evaluation cohort is empty")
        pool = store.query(source=cfg.get("eval_source") or None, ids=eval_ids,
                           digits_only=bool(cfg["require_ref_digits"]), limit=10**6)
        p2_ids = {pt["sample_id"] for pt in pts}
        if cfg["exclude_phase2_samples"]:
            pool = [sample for sample in pool if sample["id"] not in p2_ids]
        rng.shuffle(pool)
        samples = pool[:int(cfg["eval_limit"])]
        probe = pts[:30]
    if not samples:
        raise ValueError("no evaluation samples")
    ctx.log(f"model={L.name} device={L.device} eval={len(samples)} probe={len(probe)} batch_size={cfg['batch_size']}")
    return L, space, p2cfg, cands, rng, samples, probe


def conditions(cfg, cands, rng, total):
    yield "baseline", None, 0.0, "none", None, None
    for name, (support, values) in cands.items():
        for strength in cfg["strengths"]:
            strength = float(strength)
            yield name, name, strength, "nudge", support, values * strength
            for control in cfg["controls"]:
                idx, val = support, values
                if control == "shuffled_values":
                    val = rng.permutation(val)
                else:
                    idx = np.sort(rng.choice(total, len(idx), replace=False)).astype(np.int64)
                yield f"{name}|{control}", name, strength, control, idx, val * strength


def infer(ctx, L, samples, probe, bank, objective, cfg, timings, advance, batch_size=None,
          reference_audio=None, output=None):
    """All encoder/KV tensors are condition-local and bounded by batch size."""
    reference = reference_audio is not None
    batch_size = 1 if reference else int(batch_size or cfg["batch_size"])
    tokens, texts, ds = [], {}, []
    for start in range(0, len(samples), batch_size):
        ctx.check()
        batch = samples[start:start + batch_size]
        ids = [sample["id"] for sample in batch]
        with stage(timings, "encoder_s", L.device):
            enc = M.encode(L, reference_audio[ids[0]]) if reference else bank.encode(L, ids)
        with stage(timings, "decoder_s", L.device):
            generated = ([M.greedy(L, enc, list(L.sot_seq), max_new=int(cfg["max_new_tokens"])).tokens]
                         if reference else greedy_tokens(L, enc, int(cfg["max_new_tokens"]), ctx.check))
        del enc
        with stage(timings, "output_s", L.device):
            for sid, row in zip(ids, generated):
                text = L.decode(row).strip()
                texts[sid] = text
                tokens.append(row)
                if output is not None:
                    output(sid, text)
        advance(len(batch), "decode")
    with torch.inference_mode():
        for start in range(0, len(probe), batch_size):
            ctx.check()
            batch = probe[start:start + batch_size]
            ids = [pt["sample_id"] for pt in batch]
            with stage(timings, "probe_encoder_s", L.device):
                enc = M.encode(L, reference_audio[ids[0]]) if reference else bank.encode(L, ids)
            with stage(timings, "probe_objective_s", L.device):
                values = ([float(point_objective(L, enc, batch[0], objective))] if reference
                          else probe_values(L, enc, batch, objective))
                ds.extend(values)
            del enc
            advance(len(batch), "probe")
    return tokens, texts, ds


def run_batched(ctx: Ctx, cfg):
    L, space, p2cfg, cands, rng, samples, probe = prepare(ctx, cfg)
    timings = {}
    ctx.progress(0, 1, "precomputing weight-independent audio features")
    with stage(timings, "feature_precompute_s", L.device):
        bank = FeatureBank(L, samples + [store.get(p["sample_id"]) for p in probe], ctx)
    ctx.write_json("preparation_timings.json", timings)
    nconds = 1 + len(cands) * len(cfg["strengths"]) * (1 + len(cfg["controls"]))
    total, done = nconds * (len(samples) + len(probe)), 0
    table, base_texts = [], None
    if str(L.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(L.device)
    for name, typ, strength, kind, idx, val in conditions(cfg, cands, rng, space.total):
        ctx.check()
        undo, timings = None, {}
        def advance(count, label):
            nonlocal done
            done += count
            ctx.progress(done, total, f"{name} x{strength} {label}")
        try:
            with stage(timings, "edit_s", L.device):
                if idx is not None:
                    undo = space.add_sparse(idx, val)
            def output(sid, text):
                ctx.append_jsonl("outputs.jsonl", {"cond": name, "strength": strength, "sample_id": sid, "text": text})
            _, texts, ds = infer(ctx, L, samples, probe, bank, p2cfg["objective"], cfg, timings, advance, output=output)
        finally:
            with stage(timings, "undo_s", L.device):
                if undo is not None:
                    space.undo_sparse(undo)
        if typ is None:
            base_texts = texts
        ps = [1 / (1 + np.exp(-d)) for d in ds]
        info = {"n_params": len(idx), "delta_l2": float(np.linalg.norm(val))} if idx is not None else {}
        row = {"cond": name, "type": typ, "strength": strength, "kind": kind, **info,
               **_metrics(texts, base_texts), "probe_mean_logodds": float(np.mean(ds)) if ds else None,
               "probe_mean_p_digit": float(np.mean(ps)) if ps else None}
        table.append(row)
        ctx.write_json("table.json", table)
        ctx.append_jsonl("condition_timings.jsonl", {"cond": name, "strength": strength, **timings})
        ctx.log(f"{name} x{strength}: digit_share={row['digit_share']} timings={timings}")
    peak = torch.cuda.max_memory_allocated(L.device) if str(L.device).startswith("cuda") else None
    ctx.write_json("inference.json", {"backend": "batched", "batch_size": cfg["batch_size"],
                                     "peak_allocated_bytes": peak, "encoder_cache_across_conditions": False})
    return {"conditions": nconds, "eval_samples": len(samples), "inference_backend": "batched", "table": table}


BENCHMARK_DEFAULTS = {**PHASE3_DEFAULTS, "benchmark_samples": 8, "benchmark_batch_sizes": [1, 4, 8],
                      "strengths": [2.0]}


@register("phase3_benchmark", kind="gpu", defaults=BENCHMARK_DEFAULTS)
def benchmark(ctx: Ctx, cfg):
    sizes = list(dict.fromkeys(int(n) for n in cfg["benchmark_batch_sizes"]))
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError("benchmark_batch_sizes must be nonempty positive integers")
    L, space, p2cfg, cands, rng, samples, probe = prepare(ctx, cfg, benchmark=True)
    all_samples = samples + [store.get(pt["sample_id"]) for pt in probe]
    timings = {}
    with stage(timings, "feature_precompute_s", L.device):
        bank = FeatureBank(L, all_samples, ctx)
    audio = {sample["id"]: load_audio(sample["audio_path"]) for sample in all_samples}
    before = space.snapshot()
    checks, rows = {}, []
    total = (1 + len(cands) * len(cfg["strengths"]) * (1 + len(cfg["controls"]))) * (1 + len(sizes))
    try:
        for name, typ, strength, kind, idx, val in conditions(cfg, cands, rng, space.total):
            ctx.check()
            undo = None
            try:
                if idx is not None:
                    undo = space.add_sparse(idx, val)
                ctx.progress(len(rows), total, f"benchmark {name} reference")
                ref_time = {}
                ref_tokens, ref_texts, ref_ds = infer(ctx, L, samples, probe, bank, p2cfg["objective"], cfg,
                                                    ref_time, lambda *args: None, reference_audio=audio)
                rows.append({"cond": name, "strength": strength, "backend": "reference", "timings": ref_time})
                for size in sizes:
                    ctx.check()
                    ctx.progress(len(rows), total, f"benchmark {name} batch={size}")
                    if str(L.device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(L.device)
                    fast_time = {}
                    tokens, texts, ds = infer(ctx, L, samples, probe, bank, p2cfg["objective"], cfg,
                                              fast_time, lambda *args: None, batch_size=size)
                    key = f"{name}@{strength}/batch{size}"
                    checks[key + "/tokens"] = tokens == ref_tokens
                    checks[key + "/metrics"] = _metrics(texts, ref_texts) == _metrics(ref_texts, ref_texts)
                    checks[key + "/probe"] = bool(np.allclose(ds, ref_ds, rtol=2e-4, atol=2e-4))
                    rows.append({"cond": name, "strength": strength, "backend": "batched", "batch_size": size,
                                 "timings": fast_time, "tokens_equal": tokens == ref_tokens,
                                 "probe_max_abs_error": float(np.max(np.abs(np.array(ds) - ref_ds))) if ds else 0.0,
                                 "mismatched_sample_ids": [sample["id"] for sample, a, b in zip(samples, tokens, ref_tokens) if a != b],
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(L.device) if str(L.device).startswith("cuda") else None})
                    ctx.log(f"{key}: tokens_equal={tokens == ref_tokens} timings={fast_time}")
            finally:
                if undo is not None:
                    space.undo_sparse(undo)
            checks[f"{name}@{strength}/restored"] = all(torch.equal(a, b) for a, b in zip(before, space.params))
            ctx.write_json("benchmark_partial.json", {"checks": checks, "rows": rows})
        result = {"passed": all(checks.values()), "checks": checks, "rows": rows,
                  "model": L.name, "device": L.device, "samples": [s["id"] for s in samples],
                  "cohort": "phase2_discovery_only", "max_new_tokens": cfg["max_new_tokens"],
                  "preparation_timings": timings}
        ctx.write_json("benchmark.json", result)
        ctx.progress(len(rows), len(rows), "benchmark complete")
        return {"passed": result["passed"], "model": L.name, "comparisons": len(checks)}
    finally:
        space.restore_snapshot(before)
