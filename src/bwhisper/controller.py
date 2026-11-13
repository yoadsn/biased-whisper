"""Bounded, auditable controller experiments; review packets never exceed 10 sentences."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import beta

from .config import DATA, RUNS_DIR
from .corpus import store
from .hebrew_numbers import DIGIT_RE, is_degenerate
from .jobs import Ctx, register

CAMPAIGNS = DATA / "controller_campaigns"


def child(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid artifact name")
    return root / name


def sentence_cost(*texts):
    # Charge the longest side of the comparison. Overcounting abbreviations is
    # preferable to silently spending the owner's sentence-review budget.
    return max(max(1, len([s for s in re.split(r"(?<=[.!?])\s+|\n+", t.strip()) if s.strip()])) for t in texts)


@contextmanager
def ledger(campaign):
    directory = child(CAMPAIGNS, campaign)
    manifest = json.loads((directory / "campaign.json").read_text())
    with (directory / "review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "reviews.json"
        data = json.loads(path.read_text()) if path.exists() else {"spent": 0, "packets": {}}
        yield manifest, data
        tmp = directory / "reviews.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        tmp.replace(path)


@register("controller_review_batch", kind="io", defaults={
    "campaign": None, "phase3_run": None, "cond": None, "strength": None, "sample_ids": [],
})
def review_batch(ctx: Ctx, cfg):
    ids = cfg["sample_ids"]
    if not ids or len(ids) > 10 or len(set(ids)) != len(ids):
        raise ValueError("request 1-10 distinct sample IDs")
    source = child(RUNS_DIR, cfg["phase3_run"])
    job = json.loads((source / "job.json").read_text())
    if job["type"] != "phase3" or job["status"] != "done":
        raise ValueError("review requires a completed Phase 3 run")
    rows = [json.loads(line) for line in (source / "outputs.jsonl").read_text().splitlines()]
    baseline = {r["sample_id"]: r["text"] for r in rows if r["cond"] == "baseline"}
    chosen = {r["sample_id"]: r for r in rows
              if r["cond"] == cfg["cond"] and r["strength"] == cfg["strength"]}
    packet = []
    for sid in ids:
        sample = store.get(sid)
        row = chosen[sid]
        packet.append({"sample_id": sid, "source": sample["source"], "reference": sample["text"],
                       "baseline": baseline[sid], "output": row["text"],
                       "cost": sentence_cost(sample["text"], baseline[sid], row["text"])})
    cost = sum(r["cost"] for r in packet)
    if cost > 10:
        raise ValueError(f"packet contains {cost} sentences; split into batches of at most 10")
    key = f"{cfg['phase3_run']}|{cfg['cond']}|{cfg['strength']}"
    with ledger(cfg["campaign"]) as (manifest, data):
        cap = min(100, int(manifest["manual_review_cap"]))
        if data["spent"] + cost > cap:
            raise ValueError("review cap reached; consult owner before increasing the budget")
        seen = {(p["key"], sid) for p in data["packets"].values() for sid in p["sample_ids"]}
        if any((key, sid) in seen for sid in ids):
            raise ValueError("comparison already reserved/reviewed")
        allowed = set(manifest["calibration_ids"]) | set(manifest["confirmation_ids"])
        if not set(ids) <= allowed:
            raise ValueError("samples are outside the frozen campaign")
        calibration = set(ids) <= set(manifest["calibration_ids"])
        if not calibration:
            if not set(ids) <= set(manifest["confirmation_ids"]):
                raise ValueError("do not mix calibration and confirmation in one packet")
            selection_path = child(CAMPAIGNS, cfg["campaign"]) / "selection.json"
            if not selection_path.exists():
                raise ValueError("freeze a reviewed candidate before confirmation")
            selection = json.loads(selection_path.read_text())
            config = job["config"]
            if (selection["cond"] != cfg["cond"] or selection["strength"] != cfg["strength"] or
                    any(selection[k] != config.get(k) for k in ("phase2_run", "stat", "candidate_limit"))):
                raise ValueError("confirmation does not match the frozen candidate")
            if not set(ids) <= set(selection["confirmation_ids"]):
                raise ValueError("samples are outside the fixed confirmation review set")
            if any(set(ids) & set(p["sample_ids"]) for p in data["packets"].values()):
                raise ValueError("confirmation examples must not be reviewed twice")
            candidate = child(RUNS_DIR, config["phase2_run"]) / f"candidates_{cfg['cond']}.npz"
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != selection["candidate_sha256"]:
                raise ValueError("frozen candidate was modified")
        elif any(not set(p["sample_ids"]) <= set(manifest["calibration_ids"]) for p in data["packets"].values()):
            raise ValueError("do not retune calibration after opening confirmation")
        data["spent"] += cost
        data["packets"][ctx.job.id] = {"key": key, "sample_ids": ids, "cost": cost, "labels": None}
        payload = {"campaign": cfg["campaign"], "phase3_run": cfg["phase3_run"],
                   "cond": cfg["cond"], "strength": cfg["strength"],
                   "sentences_spent": data["spent"], "cap": cap, "rows": packet}
    # Commit the reservation first. A packet write failure must not expose text
    # without charging it, or silently refund an already reserved comparison.
    ctx.write_json("packet.json", payload)
    return {"sentences": cost, "spent": data["spent"], "cap": cap, "packet": "packet.json"}


@register("controller_review_submit", kind="io", defaults={"campaign": None, "packet_run": None, "labels": []})
def review_submit(ctx: Ctx, cfg):
    labels = cfg["labels"]
    packet_path = child(RUNS_DIR, cfg["packet_run"]) / "packet.json"
    packet = json.loads(packet_path.read_text())
    if packet["campaign"] != cfg["campaign"]:
        raise ValueError("packet belongs to a different campaign")
    ids = [r["sample_id"] for r in packet["rows"]]
    if sorted(r["sample_id"] for r in labels) != sorted(ids):
        raise ValueError("label every reserved sample exactly once")
    outputs = {r["sample_id"]: r["output"] for r in packet["rows"]}
    for r in labels:
        for flag in ("value_ok", "style_ok", "text_ok", "uncertain"):
            if type(r.get(flag)) is not bool:
                raise ValueError(f"{flag} must be boolean")
        if r["style_ok"] and "to_cardinal" in packet["cond"] and DIGIT_RE.search(outputs[r["sample_id"]]):
            raise ValueError("digits remain in a claimed all-cardinal output")
        if not isinstance(r.get("note"), str) or not r["note"].strip():
            raise ValueError("each correctness judgment requires a note")
        r["success"] = (r["value_ok"] and r["style_ok"] and r["text_ok"] and not r["uncertain"]
                        and not is_degenerate(outputs[r["sample_id"]]))
    with ledger(cfg["campaign"]) as (_, data):
        entry = data["packets"][cfg["packet_run"]]
        if entry["labels"] is not None:
            raise ValueError("packet already labeled; do not overwrite judgments")
        entry["labels"] = labels
        ctx.write_json("labels.json", labels)
    return {"reviewed": len(labels), "successes": sum(r["success"] for r in labels), "spent": data["spent"]}


def exact_interval(successes, total, confidence=0.95):
    """Two-sided exact interval; not a sequential stopping boundary."""
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial counts")
    a = (1 - confidence) / 2
    return [float(beta.ppf(a, successes, total - successes + 1)) if successes else 0.0,
             float(beta.ppf(1 - a, successes + 1, total - successes)) if successes < total else 1.0]


@register("controller_freeze", kind="io", defaults={"campaign": None, "packet_run": None})
def freeze_controller(ctx: Ctx, cfg):
    """Lock a reviewed calibration candidate before opening confirmation text."""
    directory = child(CAMPAIGNS, cfg["campaign"])
    selection_path = directory / "selection.json"
    packet = json.loads((child(RUNS_DIR, cfg["packet_run"]) / "packet.json").read_text())
    if packet["campaign"] != cfg["campaign"]:
        raise ValueError("packet campaign mismatch")
    with ledger(cfg["campaign"]) as (manifest, data):
        if selection_path.exists():
            raise ValueError("candidate already frozen; no confirmation-driven retuning")
        entry = data["packets"][cfg["packet_run"]]
        if entry["labels"] is None or not set(entry["sample_ids"]) <= set(manifest["calibration_ids"]):
            raise ValueError("choose a labeled calibration candidate")
        job = json.loads((child(RUNS_DIR, packet["phase3_run"]) / "job.json").read_text())
        config = job["config"]
        candidate = child(RUNS_DIR, config["phase2_run"]) / f"candidates_{packet['cond']}.npz"
        selection = {"phase2_run": config["phase2_run"], "cond": packet["cond"],
                     "strength": packet["strength"], "stat": config["stat"],
                     "candidate_limit": config.get("candidate_limit"),
                     "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                     "calibration_packet": cfg["packet_run"],
                     "confirmation_ids": manifest.get("confirmation_review_ids", manifest["confirmation_ids"])}
        ctx.write_json("selection.json", selection)
        selection_path.write_text(json.dumps(selection, indent=1))
    return selection


@register("controller_review_summary", kind="io", defaults={"campaign": None})
def review_summary(ctx: Ctx, cfg):
    with ledger(cfg["campaign"]) as (manifest, data):
        groups = {}
        for packet_id, entry in data["packets"].items():
            packet = json.loads((child(RUNS_DIR, packet_id) / "packet.json").read_text())
            phase = "calibration" if set(entry["sample_ids"]) <= set(manifest["calibration_ids"]) else "confirmation"
            job = json.loads((child(RUNS_DIR, packet["phase3_run"]) / "job.json").read_text())
            source = job.get("config", {}).get("phase2_run", packet["phase3_run"])
            key = f"{phase}/{source}/{packet['cond']}@{packet['strength']}"
            g = groups.setdefault(key, {"n": 0, "success": 0, "value_ok": 0, "uncertain": 0,
                                        "digit_target_n": 0, "digit_target_success": 0})
            rows = {r["sample_id"]: r for r in packet["rows"]}
            for label in entry["labels"] or []:
                g["n"] += 1
                g["success"] += label["success"]
                g["value_ok"] += label["value_ok"] and not label["uncertain"]
                g["uncertain"] += label["uncertain"]
                if DIGIT_RE.search(rows[label["sample_id"]]["baseline"]):
                    g["digit_target_n"] += 1
                    g["digit_target_success"] += label["success"]
        for g in groups.values():
            g["ci95"] = exact_interval(g["success"], g["n"]) if g["n"] else None
            n, k = g["digit_target_n"], g["digit_target_success"]
            g["digit_target_ci95"] = exact_interval(k, n) if n else None
            g["digit_target_lower95_one_sided"] = float(beta.ppf(.05, k, n-k+1)) if k else 0.
        result = {"sentences_spent": data["spent"], "cap": min(100, manifest["manual_review_cap"]),
                  "groups": groups, "inference": "Exact intervals are fixed-sample, not sequential stopping boundaries. Calibration is exploratory; reference/baseline agreement is not verified audio truth."}
        ctx.write_json("summary.json", result)
    return result


@register("controller_candidates", kind="io", defaults={"source_run": None, "variants": []})
def build_candidates(ctx: Ctx, cfg):
    """Compose/filter already-discovered sparse edits without loading a model."""
    source = child(RUNS_DIR, cfg["source_run"])
    job = json.loads((source / "job.json").read_text())
    if job["status"] != "done":
        raise ValueError("candidate source must be complete")
    meta = json.loads((source / "space.json").read_text())
    starts = np.concatenate([[0], np.cumsum(meta["sizes"])])
    variants = cfg["variants"]
    names = [v["name"] for v in variants]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("variant names must be nonempty, unique safe identifiers")
    edits = {}
    for variant in variants:
        pieces = []
        for part in variant["parts"]:
            edit = part["edit"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", edit):
                raise ValueError("invalid source edit")
            scale = float(part.get("scale", 1))
            if not math.isfinite(scale):
                raise ValueError("scale must be finite")
            with np.load(source / f"candidates_{edit}.npz", allow_pickle=False) as z:
                idx, val = z["idx"], z["mean_all"]
            which = np.searchsorted(starts[1:], idx, side="right")
            pattern = re.compile(part.get("include_regex", ".*"))
            keep = np.array([bool(pattern.search(n)) for n in meta["names"]])[which]
            pieces.append((idx[keep], val[keep] * scale))
        idx = np.concatenate([p[0] for p in pieces])
        values = np.concatenate([p[1] for p in pieces])
        idx, inverse = np.unique(idx, return_inverse=True)
        val = np.zeros(len(idx), dtype=np.float32)
        np.add.at(val, inverse, values)
        keep = val != 0
        idx, val = idx[keep], val[keep]
        if not len(idx) or not np.isfinite(val).all():
            raise ValueError("empty or nonfinite candidate")
        np.savez(ctx.dir / f"candidates_{variant['name']}.npz", idx=idx,
                 mean_all=val, mean_present=val, median_present=val)
        edits[variant["name"]] = {"params": len(idx), "delta_l2": float(np.linalg.norm(val))}
    # Phase 3 consumes the source discovery config, preserving its model/filter,
    # objective and sample-exclusion provenance. This is NOT a new attribution run.
    own = {k: ctx.job.config[k] for k in ("source_run", "variants") if k in ctx.job.config}
    ctx.job.config.update(job["config"])
    ctx.job.config.update(own)
    ctx.job.save()
    ctx.write_json("space.json", meta)
    ctx.write_json("summary.json", {"model": job["config"]["model"], "source_run": source.name,
                                     "variants": variants, "edits": edits})
    return {"source_run": source.name, "edits": edits}


@register("controller_refit", kind="gpu", defaults={"source_run": None, "release_model_after_run": True,
          "neuron_counts": [32, 64], "head_counts": [2, 4], "towers": ["decoder"],
          "maps": ["full", "decision"], "capture_sample_ids": None})
def refit_candidates(ctx: Ctx, cfg):
    """Re-rank saved attribution maps; no new captures or manual labeling.

    The original global full-span ranking can include acoustic/content units.
    Compare small decoder-only supports and the decision-position map, retaining
    the original output-gain translation and all parameter-space checks.
    """
    from . import model as M
    from .paramspace import ParamSpace
    from .phase2n import bh_fdr, unit_stats, _weight_edits, _save_candidates

    source = child(RUNS_DIR, cfg["source_run"])
    job = json.loads((source / "job.json").read_text())
    if job["type"] != "phase2n" or job["status"] != "done":
        raise ValueError("requires a completed original Phase 2N attribution run")
    pcfg = job["config"]
    counts = [("neurons", n) for n in cfg.get("neuron_counts", [32, 64])] + [
        ("heads", n) for n in cfg.get("head_counts", [2, 4])]
    towers = cfg.get("towers", ["decoder"])
    maps = cfg.get("maps", ["full", "decision"])
    if (not counts or any(type(n) is not int or not 1 <= n <= 1024 for _, n in counts) or
            not towers or not set(towers) <= {"encoder", "decoder"} or
            not maps or not set(maps) <= {"full", "decision"}):
        raise ValueError("invalid bounded refit counts/towers/maps")
    layout = json.loads((source / "components.json").read_text())
    saved_space = json.loads((source / "space.json").read_text())
    L = M.load_model(pcfg["model"])
    M.unload_unused(L)
    space = ParamSpace(L.model, pcfg["param_filter"], pcfg["include_regex"], pcfg["exclude_regex"])
    if space.signature != saved_space["signature"]:
        raise ValueError("source model/parameter signature mismatch")
    heads_dim = {"decoder": L.model.config.d_model // L.model.config.decoder_attention_heads,
                 "encoder": L.model.config.d_model // L.model.config.encoder_attention_heads}
    made = {}
    ctx.job.config.update(pcfg)
    ctx.job.save()
    ctx.write_json("space.json", saved_space)
    with np.load(source / "attributions.npz", allow_pickle=False) as captures:
        mask = slice(None)
        selected = cfg.get("capture_sample_ids")
        if selected is not None:
            if (len(selected) < 3 or any(type(sid) is not int or sid <= 0 for sid in selected) or
                    len(selected) != len(set(selected))):
                raise ValueError("capture_sample_ids must contain >=3 distinct positive IDs")
            ids = np.array([int(str(key).rsplit("__", 1)[1]) for key in captures["keys"]])
            mask = np.isin(ids, selected)
            if int(mask.sum()) != len(selected) or set(ids[mask]) != set(selected):
                raise ValueError("requested capture IDs are missing or duplicated in source")
            ctx.job.config.update(sample_ids=selected, max_points=len(selected))
            ctx.job.save()
        for map_name in maps:
            ctx.check()
            st = unit_stats(captures[map_name][mask])
            significant = bh_fdr(st["p"], float(pcfg["fdr"]))
            site_of = np.empty(len(st["mean"]), dtype=np.int64)
            for i, site in enumerate(layout):
                site_of[site["offset"]:site["offset"] + site["n"]] = i
            for tag, count in counts:
                selection = []
                for u in np.argsort(-np.abs(st["mean"])):
                    site = layout[site_of[u]]
                    if (not significant[u] or site["tower"] not in towers or
                            (site["kind"] == "mlp") != (tag == "neurons")):
                        continue
                    selection.append((site, int(u - site["offset"]), float(st["mean"][u])))
                    if len(selection) == count:
                        break
                if not selection:
                    continue
                idx, values = _weight_edits(L, space, layout, selection, float(pcfg["gain"]), heads_dim)
                scope = "dec" if towers == ["decoder"] else "enc" if towers == ["encoder"] else "all"
                name = f"{scope}_{map_name}_{tag}{count}"
                for direction, sign in (("to_cardinal", 1), ("to_digits", -1)):
                    _save_candidates(ctx.dir / f"candidates_{name}_{direction}.npz", idx, sign * values)
                made[name] = {"units": [{"site": s["name"], "index": u, "mean": m} for s, u, m in selection],
                              "params": len(idx), "delta_l2": float(np.linalg.norm(values))}
                ctx.log(f"{name}: {len(selection)} units / {len(idx)} parameters")
    ctx.write_json("summary.json", {"model": L.name, "source_run": source.name,
                                    "method": "saved-attribution bounded gain refit; no new discovery audio",
                                    "capture_sample_ids": selected,
                                    "edits": made})
    return {"source_run": source.name, "variants": list(made)}


@register("controller_residual_candidates", kind="io", defaults={"source_run": None})
def residual_candidates(ctx: Ctx, cfg):
    """Expose every saved decoder-layer gradient direction, not just max-t.

    Same calculation/scale as Phase 2N; values remain weight edits to fc2.bias.
    No model load, inference, new attribution or manual review is performed.
    """
    from .phase2n import _save_candidates

    source = child(RUNS_DIR, cfg["source_run"])
    job = json.loads((source / "job.json").read_text())
    if job["type"] != "phase2n" or job["status"] != "done":
        raise ValueError("requires a completed original Phase 2N run")
    pcfg = job["config"]
    meta = json.loads((source / "space.json").read_text())
    starts = np.concatenate([[0], np.cumsum(meta["sizes"])])
    with np.load(source / "attributions.npz", allow_pickle=False) as z:
        gradients = z["resid_grad"]
    normalized = gradients / (np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-30)
    direction = normalized.mean(0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-30
    projection = np.einsum("nld,ld->nl", gradients, direction).mean(0)
    made = {}
    for layer, vector in enumerate(direction):
        name = f"model.decoder.layers.{layer}.fc2.bias"
        if name not in meta["names"] or projection[layer] <= 0:
            continue
        tensor = meta["names"].index(name)
        if meta["sizes"][tensor] != len(vector):
            raise ValueError("residual width/parameter signature mismatch")
        alpha = float(pcfg["steer_target"]) / float(projection[layer])
        values = (-alpha * vector).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite residual candidate")
        idx = (starts[tensor] + np.arange(len(vector))).astype(np.int64)
        tag = f"steer_L{layer}"
        for direction_name, sign in (("to_cardinal", 1), ("to_digits", -1)):
            _save_candidates(ctx.dir / f"candidates_{tag}_{direction_name}.npz", idx, sign * values)
        made[tag] = {"param": name, "alpha": alpha, "params": len(idx)}
    if not made:
        raise ValueError("no usable residual candidates")
    ctx.job.config.update(pcfg)
    ctx.job.save()
    ctx.write_json("space.json", meta)
    ctx.write_json("summary.json", {"model": pcfg["model"], "source_run": source.name, "edits": made})
    return {"source_run": source.name, "variants": list(made)}
