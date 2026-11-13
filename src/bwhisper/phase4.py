"""Online frozen-teacher pseudo-labeling -> decoder LoRA, independent of 2N/3."""
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from . import model as M
from .audio import load_audio
from .config import DATA, RUNS_DIR
from .corpus import store
from .hebrew_numbers import DIGIT_RE, count_mentions, is_degenerate
from .jobs import register
from .lora import Adapter
from .phase1 import StyleTokens, analyse_sample

DEFAULTS = {
    "target_style": "cardinals", "teacher_source_run": None,
    "parallel": True, "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    "global_batch_size": 8, "resume_run": None, "checkpoint_every_steps": 100,
    "worker_timeout_seconds": 900, "cpu_threads_per_worker": 2,
    "nccl_p2p_level": "NVL",
    "model": None, "device": "auto", "cohort_run": None,
    "train_limit": 20000, "eval_limit": 50, "epochs": 1, "seed": 42,
    "rank": 8, "alpha": 16., "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "encoder_top_layers": 0, "learning_rate": 0.0001, "weight_decay": 0.01,
    "gradient_accumulation": 8, "max_grad_norm": 1., "warmup_steps": 20,
    "max_new_tokens": 192, "min_rest_match": 0.9,
    "include_natural_cardinal": True, "include_no_number": True,
    "value_filter": False, "bf16": True, "eval_every_steps": 0,
    "release_model_after_run": True,
}


def validate(cfg):
    if cfg.get("target_style", "cardinals") not in ("cardinals", "digits"):
        raise ValueError("target_style must be cardinals or digits")
    if cfg.get("target_style") == "digits" and (not cfg.get("parallel") or not cfg.get("teacher_source_run")):
        raise ValueError("digits training requires parallel mode and a completed cardinal teacher_source_run")
    if cfg.get("teacher_source_run") and cfg.get("target_style", "cardinals") != "digits":
        raise ValueError("teacher_source_run is only supported for cached digits training")
    for key in ("train_limit", "eval_limit", "rank", "gradient_accumulation", "max_new_tokens"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["epochs"]) is not int or not 1 <= cfg["epochs"] <= 3:
        raise ValueError("epochs must be 1..3")
    for key in ("learning_rate", "alpha", "max_grad_norm"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= cfg["min_rest_match"] <= 1 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("invalid filter/dropout")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("invalid weight_decay")
    for key in ("warmup_steps", "eval_every_steps", "encoder_top_layers"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    modules = cfg["target_modules"]
    if (not isinstance(modules, list) or not modules or len(set(modules)) != len(modules) or
            not set(modules) <= {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}):
        raise ValueError("target_modules must be distinct supported linear module names")


def run_dir(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid run ID")
    return RUNS_DIR / name


def recording(sample):
    meta = sample["meta"]
    identity = meta.get("recording") or meta.get("uuid")
    if not identity:
        raise ValueError(f"sample {sample['id']} lacks recording identity")
    return f"{sample['source']}:{identity}"


def select_cohorts(cfg):
    # Reuse the established split, never silently repartition old held-out audio.
    name = cfg["cohort_run"]
    if not name:
        paths = sorted(RUNS_DIR.glob("setup_scale-*/cohorts.json"))
        if not paths:
            raise ValueError("create recording-disjoint setup_scale cohorts first")
        name = paths[-1].parent.name
    path = run_dir(name) / "cohorts.json"
    split = json.loads(path.read_text())
    train_ids, eval_ids = split["discovery"], split["evaluation"]
    if (not train_ids or not eval_ids or set(train_ids) & set(eval_ids) or
            len(set(train_ids)) != len(train_ids) or len(set(eval_ids)) != len(eval_ids)):
        raise ValueError("invalid overlapping/empty cohorts")
    samples = {s["id"]: s for s in store.query(ids=train_ids + eval_ids, limit=10**9)}
    if set(samples) != set(train_ids + eval_ids):
        raise ValueError("cohort contains missing corpus IDs")
    if {recording(samples[i]) for i in train_ids} & {recording(samples[i]) for i in eval_ids}:
        raise ValueError("recording leakage across cohorts")
    # Preserve unopened controller confirmation recordings even from monitoring.
    protected = set()
    for manifest in (DATA / "controller_campaigns").glob("*/campaign.json"):
        for sid in json.loads(manifest.read_text()).get("confirmation_ids", []):
            sample = store.get(sid)
            if sample:
                protected.add(recording(sample))
    train_ids = [i for i in train_ids if recording(samples[i]) not in protected]
    eval_ids = [i for i in eval_ids if recording(samples[i]) not in protected]
    rng = random.Random(cfg["seed"])
    rng.shuffle(train_ids)
    rng.shuffle(eval_ids)
    # Monitoring uses one clip per recording, selected before any predictions.
    seen, evaluation = set(), []
    for sid in eval_ids:
        group = recording(samples[sid])
        if group not in seen:
            evaluation.append(sid)
            seen.add(group)
    train_ids, evaluation = train_ids[:cfg["train_limit"]], evaluation[:cfg["eval_limit"]]
    if not train_ids or not evaluation:
        raise ValueError("no training/monitoring samples after exclusions")
    return samples, {"cohort_run": name, "cohort_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "train_ids": train_ids, "eval_ids": evaluation,
                     "protected_recordings": len(protected), "seed": cfg["seed"]}


def training_target(row, cfg):
    """Return explicit tokens or an auditable rejection; no value gate by default."""
    verdict = row.get("verdict")
    if row.get("baseline_ended") is False:
        return None, "baseline_truncated"
    if verdict == "no_digits":
        text, tokens = row.get("baseline_text", ""), row.get("baseline_tokens", [])
        has_number = count_mentions(text)["word_mentions"] > 0
        if not cfg["include_natural_cardinal"] and has_number:
            return None, "natural_cardinal_disabled"
        if not cfg["include_no_number"] and not has_number:
            return None, "no_number_disabled"
        kind = "natural_cardinal" if has_number else "preservation"
        ended = row.get("baseline_ended", False)
    elif verdict == "cardinal":
        if row.get("diff", {}).get("rest_match", 0) < cfg["min_rest_match"]:
            return None, "rest_match"
        text, tokens = row.get("suppressed_text", ""), row.get("y_cardinal", [])
        ended, kind = row.get("suppressed_ended", False), "suppressed_cardinal"
        if cfg["value_filter"]:
            from .number_values import same_values
            if not same_values(row["baseline_text"], text):
                return None, "value_mismatch_or_unknown"
    else:
        return None, str(verdict or "missing_verdict")
    if not ended:
        return None, "truncated"
    if not text.strip() or not tokens:
        return None, "empty"
    if DIGIT_RE.search(text):
        return None, "remaining_digits"
    if is_degenerate(text):
        return None, "degenerate"
    return {"tokens": tokens, "text": text, "kind": kind}, "accepted"


def teacher_loss(L, features, tokens, forward_model=None):
    """Full cardinal continuation + EOS; forced language/task prefix is context only."""
    ids = list(L.sot_seq) + list(tokens) + [L.eos]
    if len(ids) - 1 > L.model.config.max_target_positions:
        raise ValueError("target exceeds decoder context")
    sequence = torch.tensor([ids], device=L.device)
    output = (forward_model or L.model)(input_features=features, decoder_input_ids=sequence[:, :-1], use_cache=False)
    start = len(L.sot_seq) - 1
    return torch.nn.functional.cross_entropy(output.logits[:, start:].float().reshape(-1, L.vocab),
                                             sequence[:, start + 1:].reshape(-1))


def prevalence(rows):
    n = len(rows)
    mentions = [count_mentions(r["text"]) for r in rows]
    digits = sum(m["digit_mentions"] for m in mentions)
    cardinal = sum(m["word_mentions"] for m in mentions)
    return {"n": n, "clips_with_digits": sum(bool(DIGIT_RE.search(r["text"])) for r in rows),
            "digit_clip_rate": sum(bool(DIGIT_RE.search(r["text"])) for r in rows) / n if n else None,
            "digit_mentions": digits, "cardinal_mentions": cardinal,
            "digit_mention_share": digits / (digits + cardinal) if digits + cardinal else None,
            "degenerate": sum(is_degenerate(r["text"]) for r in rows),
            "truncated": sum(not r["ended"] for r in rows),
            "empty": sum(not r["text"].strip() for r in rows)}


@register("phase4_evaluate", kind="gpu", defaults={"training_run": None, "checkpoint": "latest",
          "device": "auto", "release_model_after_run": True})
def evaluate_checkpoint(ctx, cfg):
    """Reload an adapter and repeat its frozen monitoring cohort, without a ban."""
    source = run_dir(cfg["training_run"])
    job = json.loads((source / "job.json").read_text())
    if job["type"] != "phase4" or job["status"] in ("running", "queued"):
        raise ValueError("select an inactive Phase 4 training run")
    name = cfg["checkpoint"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid checkpoint name")
    directory = source / name
    metadata = json.loads((directory / "adapter.json").read_text())
    if metadata["model"] != job["config"]["model"] or metadata.get("format") != "bwhisper-lora-v1":
        raise ValueError("adapter/base model provenance mismatch")
    manifest = json.loads((source / "manifest.json").read_text())
    ctx.write_json("manifest.json", manifest)
    L = M.load_model(metadata["model"], device=M.pick_device(cfg["device"]))
    M.unload_unused(L)
    adapter = Adapter(L.model, metadata["rank"], metadata["alpha"], metadata["dropout"],
                      metadata["modules"], metadata["encoder_top_layers"])
    base, edited = [], []
    try:
        adapter.load(directory)
        L.model.eval()
        for i, sid in enumerate(manifest["eval_ids"]):
            ctx.check()
            sample = store.get(sid)
            if not sample:
                raise ValueError(f"missing corpus sample {sid}")
            audio = load_audio(sample["audio_path"])
            with adapter.teacher():
                text, result = M.transcribe(L, audio, job["config"]["max_new_tokens"])
            base.append({"sample_id": sid, "text": text, "ended": result.ended})
            text, result = M.transcribe(L, audio, job["config"]["max_new_tokens"])
            edited.append({"sample_id": sid, "text": text, "ended": result.ended})
            ctx.append_jsonl("evaluation.jsonl", {**edited[-1], "baseline": base[-1]["text"]})
            ctx.progress(i + 1, len(manifest["eval_ids"]), "unsuppressed adapter evaluation")
        result = {"training_run": source.name, "checkpoint": name, "target_style": metadata.get("target_style", job["config"].get("target_style", "cardinals")), "baseline": prevalence(base),
                  "adapter": prevalence(edited), "suppression_at_evaluation": False}
        ctx.write_json("evaluation.json", result)
        return result
    finally:
        adapter.restore()


@register("phase4", kind="gpu", defaults=DEFAULTS)
def run_phase4(ctx, cfg):
    cfg.setdefault("target_style", "cardinals")
    validate(cfg)
    if cfg.get("parallel"):
        from .phase4_parallel import run_parallel
        return run_parallel(ctx, cfg)
    if cfg.get("resume_run"):
        raise ValueError("resume_run requires the parallel/cache backend")
    samples, manifest = select_cohorts(cfg)
    ctx.write_json("manifest.json", manifest)
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    L = M.load_model(cfg["model"], device=M.pick_device(cfg["device"]))
    M.unload_unused(L)
    cfg["model"], cfg["device"], cfg["cohort_run"] = L.name, str(L.device), manifest["cohort_run"]
    ctx.job.save()
    if cfg["max_new_tokens"] + len(L.sot_seq) > L.model.config.max_target_positions:
        raise ValueError("max_new_tokens exceeds decoder context including forced prefix")
    use_bf16 = cfg["bf16"] and str(L.device).startswith("cuda")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 unavailable; set bf16=false")
    def precision():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    adapter = Adapter(L.model, cfg["rank"], cfg["alpha"], cfg["dropout"],
                      cfg["target_modules"], cfg["encoder_top_layers"])
    started, step, seen, accepted, accumulated = time.monotonic(), 0, 0, 0, 0
    counts, targets, baseline = Counter(), {}, {}
    status = {"stage": "initializing", "epoch": 0, "step": 0, "counts": {},
              "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
              "train_samples": len(manifest["train_ids"]), "eval_samples": len(manifest["eval_ids"]),
              "review_budget": "No automatic correctness labels. Existing campaign spend remains 50/100 sentences."}
    total = len(manifest["train_ids"]) * cfg["epochs"]
    def publish(stage, **extra):
        status.update(stage=stage, step=step, seen=seen, accepted_presentations=accepted,
                      counts=dict(counts), elapsed_seconds=time.monotonic() - started, **extra)
        ctx.write_json("monitor.tmp.json", status)
        (ctx.dir / "monitor.tmp.json").replace(ctx.dir / "monitor.json")
        ctx.progress(seen, total, f"{stage}: epoch {status['epoch']}/{cfg['epochs']}, update {step}")

    def checkpoint(name):
        adapter.save(ctx.dir / name, {"format": "bwhisper-lora-v1", "model": L.name, "target_style": cfg["target_style"],
                     "step": step, "epoch": status["epoch"], "training_config": cfg})
        ctx.append_jsonl("checkpoints.jsonl", {"path": name, "step": step, "epoch": status["epoch"]})

    def evaluate(epoch):
        publish("evaluation")
        rows = []
        training = L.model.training
        L.model.eval()
        try:
            for sid in manifest["eval_ids"]:
                ctx.check()
                audio = load_audio(samples[sid]["audio_path"])
                if sid not in baseline:
                    with adapter.teacher():
                        text, result = M.transcribe(L, audio, cfg["max_new_tokens"])
                    baseline[sid] = {"sample_id": sid, "text": text, "ended": result.ended}
                    ctx.append_jsonl("baseline.jsonl", baseline[sid])
                text, result = M.transcribe(L, audio, cfg["max_new_tokens"])
                row = {"sample_id": sid, "text": text, "ended": result.ended,
                       "epoch": epoch, "step": step, "baseline": baseline[sid]["text"]}
                rows.append(row)
                ctx.append_jsonl("evaluation.jsonl", row)
                publish("evaluation", eval_done=len(rows), eval_total=len(manifest["eval_ids"]))
            metrics = {"epoch": epoch, "step": step, "baseline": prevalence(list(baseline.values())),
                       "adapter": prevalence(rows), "suppression_at_evaluation": False}
            ctx.append_jsonl("evaluations.jsonl", metrics)
            status["evaluation"] = metrics
            ctx.log(f"evaluation epoch {epoch}, step {step}: digit clip rate {metrics['adapter']['digit_clip_rate']:.3f}")
        finally:
            L.model.train(training)

    try:
        style = StyleTokens(L)  # fixed built-in rules, no mutable latest-satellite dependency
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        optimizer.zero_grad(set_to_none=True)
        publish("ready")
        ctx.log(f"Phase 4: frozen teacher, {status['trainable_parameters']:,} LoRA parameters; no Phase 2N/3 required")
        evaluate(0)
        def update():
            nonlocal accumulated, step
            for p in adapter.parameters():
                if p.grad is not None:
                    p.grad.div_(accumulated)
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
            lr = cfg["learning_rate"] * min(1., (step + 1) / max(1, cfg["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step += 1
            ctx.append_jsonl("metrics.jsonl", {"step": step, "epoch": status["epoch"],
                             "loss": status["loss"], "learning_rate": lr, "grad_norm": float(norm),
                             "seen": seen, "accepted": accepted, "elapsed_seconds": time.monotonic() - started})
            publish("training", learning_rate=lr)
            if cfg["eval_every_steps"] and step % cfg["eval_every_steps"] == 0:
                checkpoint(f"step-{step:06d}")
                evaluate(status["epoch"])

        for epoch in range(1, cfg["epochs"] + 1):
            status["epoch"] = epoch
            ids = list(manifest["train_ids"])
            rng.shuffle(ids)
            for sid in ids:
                ctx.check()
                seen += 1
                audio = None
                if sid not in targets:
                    publish("teacher_generation")
                    audio = load_audio(samples[sid]["audio_path"])
                    with adapter.teacher():
                        row = analyse_sample(L, audio, cfg["max_new_tokens"], style)
                    target, reason = training_target(row, cfg)
                    # Prevent unsupported special IDs / too-long pseudo-labels entering loss.
                    if target and (len(target["tokens"]) + len(L.sot_seq) > L.model.config.max_target_positions or
                                   any(t < 0 or t >= L.eos for t in target["tokens"])):
                        target, reason = None, "invalid_target_tokens"
                    targets[sid] = target
                    counts[reason if target is None else target["kind"]] += 1
                    ctx.append_jsonl("targets.jsonl", {"sample_id": sid, "verdict": row["verdict"],
                                     "reason": reason, "baseline": row.get("baseline_text"), "target": target})
                target = targets[sid]
                if target is None:
                    publish("filtering")
                    continue
                if audio is None:
                    audio = load_audio(samples[sid]["audio_path"])
                L.model.train()
                with precision():
                    loss = teacher_loss(L, M.features(L, audio), target["tokens"])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at sample {sid}")
                loss.backward()
                accepted += 1
                accumulated += 1
                status["loss"] = float(loss.detach())
                if accumulated >= cfg["gradient_accumulation"]:
                    ctx.check()
                    update()
                else:
                    publish("training")
            if accumulated:
                ctx.check()
                update()
            if not step:
                raise ValueError("no accepted training examples; inspect targets.jsonl")
            checkpoint(f"epoch-{epoch:02d}")
            evaluate(epoch)
        publish("complete")
        return {"model": L.name, "updates": step, "counts": dict(counts),
                "adapter": f"epoch-{cfg['epochs']:02d}", "evaluation": status["evaluation"],
                "trainable_parameters": status["trainable_parameters"]}
    finally:
        # Cancellation/failure retains the latest partial adapter, but never contaminates cached base ASR.
        try:
            if step:
                checkpoint("latest")
        finally:
            adapter.restore()
