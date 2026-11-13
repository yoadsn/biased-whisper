"""Spawn/torchrun entry point for Phase 4. No API imports or Job Manager."""
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import model as M
from .audio import load_audio
from .lora import Adapter
from .phase1 import StyleTokens, analyse_sample
from .phase4 import teacher_loss, training_target, prevalence
from .phase4_parallel import atomic_json, claim_task, save_cache


class Telemetry:
    def __init__(self, directory, rank, device, parent_pid=None):
        self.directory, self.rank, self.device = directory, rank, device
        self.state = {"rank": rank, "device": str(device), "stage": "loading", "done": 0,
                      "progress_at": time.time()}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.parent_pid = parent_pid
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.is_set():
            if self.parent_pid and not Path(f"/proc/{self.parent_pid}").exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return
            with self.lock:
                self.state.update(**self.memory())
                state = {**self.state, "heartbeat": time.time()}
                atomic_json(self.directory / f"rank-{self.rank}.json", state)
            self.stop.wait(1)

    def update(self, **values):
        with self.lock:
            self.state.update(progress_at=time.time(), **values)

    def memory(self):
        cuda = str(self.device).startswith("cuda")
        return {"memory_bytes": torch.cuda.memory_allocated(self.device) if cuda else 0,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else 0}

    def append(self, filename, value):
        # Shared scalar/artifact histories are written ONLY by student rank 0.
        with (self.directory / filename).open("a") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def close(self):
        self.stop.set()
        self.thread.join()
        atomic_json(self.directory / f"rank-{self.rank}.json", {**self.state, **self.memory(), "heartbeat": time.time()})


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def load(request, rank):
    cfg = request["cfg"]
    torch.set_num_threads(cfg["cpu_threads_per_worker"])
    torch.manual_seed(cfg["seed"])
    device = cfg["devices"][rank]
    torch.cuda.set_device(device)
    # Pin the teacher's language/task settings to the parent snapshot.
    M.load_settings = lambda: request["settings"]
    L = M.load_model(cfg["model"], device=device)
    if cfg["max_new_tokens"] + len(L.sot_seq) > L.model.config.max_target_positions:
        raise ValueError("max_new_tokens exceeds decoder context")
    return L


def generate_one(L, sample, cfg, style):
    # Feature extractors may support dithering. Pin any stochastic preprocessing
    # per sample so dynamic assignment cannot change a cached teacher outcome.
    seed = (int(cfg["seed"]) + sample["id"]) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    audio = load_audio(sample["audio_path"])
    audio_seconds = time.monotonic() - started
    synchronize(L.device)
    started = time.monotonic()
    with torch.no_grad():
        row = analyse_sample(L, audio, cfg["max_new_tokens"], style)
    synchronize(L.device)
    teacher_seconds = time.monotonic() - started
    target, reason = training_target(row, cfg)
    if target and (len(target["tokens"]) + len(L.sot_seq) > L.model.config.max_target_positions or
                   any(type(t) is not int or t < 0 or t >= L.eos for t in target["tokens"])):
        target, reason = None, "invalid_target_tokens"
    return {"sample_id": sample["id"], "verdict": row["verdict"], "reason": reason,
            "baseline": row.get("baseline_text"), "target": target,
            "audio_seconds": audio_seconds, "teacher_seconds": teacher_seconds}


def teacher(request, run, rank):
    directory = run / "teacher"
    telemetry = Telemetry(directory, rank, request["cfg"]["devices"][rank], request.get("parent_pid"))
    done, counts, seconds, audio_seconds = 0, Counter(), 0., 0.
    try:
        L = load(request, rank)
        style = StyleTokens(L)
        with sqlite3.connect(directory / "tasks.db", timeout=60) as db:
            while not (directory / "STOP").exists():
                sample = claim_task(db)
                if sample is None:
                    break
                telemetry.update(stage="teacher_generation", sample_id=sample["id"])
                row = generate_one(L, sample, request["cfg"], style)
                key = request["sample_keys"][str(sample["id"])]
                row.update(fingerprint=request["manifest"]["teacher_fingerprint"], sample_key=key)
                save_cache(Path(request["cache"]) / f"{sample['id']}-{key}.json", row)
                telemetry.append(f"outcomes-{rank}.jsonl", row)
                done += 1
                seconds += row["teacher_seconds"]
                audio_seconds += row["audio_seconds"]
                counts[row["target"]["kind"] if row["target"] else row["reason"]] += 1
                telemetry.update(done=done, counts=dict(counts), teacher_seconds=seconds, audio_seconds=audio_seconds)
        telemetry.update(stage="cancelled" if (directory / "STOP").exists() else "done")
    except BaseException as error:
        telemetry.update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        telemetry.close()


def global_batches(ids, batch_size, rank, world_size):
    """All ranks get the same number of slots; None is a zero-loss tail slot."""
    for start in range(0, len(ids), batch_size):
        real = ids[start:start + batch_size]
        padded = real + [None] * (batch_size - len(real))
        yield padded[rank::world_size], len(real)


def rng_state(device):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            "cuda": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def train_engine(L, request, run, rank, world, telemetry, feature_fn=None, evaluate_fn=None):
    """Shared CUDA/Gloo-test engine; per-example mean loss matches serial Phase 4."""
    cfg, manifest = request["cfg"], request["manifest"]
    device = L.device
    use_cuda = str(device).startswith("cuda")
    adapter = Adapter(L.model, cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["target_modules"], cfg["encoder_top_layers"])
    step, next_epoch, next_batch = 0, 1, 0
    seen, local_seen = 0, 0
    directory = run / "distributed"
    feature_fn = feature_fn or (lambda sid: M.features(L, load_audio(request["samples"][str(sid)]["audio_path"])))
    targets = {r["sample_id"]: r["target"] for r in request["accepted"]}
    ids = manifest["accepted_ids"]
    start = time.monotonic()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    # Construct LoRA identically on all ranks, then independent dropout streams.
    student = DDP(L.model, device_ids=[torch.device(device).index] if use_cuda else None,
                  broadcast_buffers=False, find_unused_parameters=False)
    torch.manual_seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank)
    np.random.seed((cfg["seed"] + rank) % (2**32))
    last_eval = None
    try:
        if request.get("resume_checkpoint"):
            source = Path(request["resume_checkpoint"])
            adapter.load(source)
            saved = torch.load(source / "training_state.pt", map_location="cpu", weights_only=True)
            if saved["training_fingerprint"] != manifest["training_fingerprint"] or saved["world_size"] != world:
                raise ValueError("incompatible resume training state")
            optimizer.load_state_dict(saved["optimizer"])
            step, next_epoch, next_batch, seen = (saved[k] for k in ("step", "next_epoch", "next_batch", "seen"))
            restore_rng(saved["rng"][rank], device)
        initial_seen, initial_step = seen, step

        def stop_requested():
            flag = torch.tensor(int((directory / "STOP").exists()), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return bool(flag.item())

        def checkpoint(label, epoch, batch):
            telemetry.update(stage="checkpoint", step=step)
            states = [None] * world if rank == 0 else None
            dist.gather_object(rng_state(device), states, dst=0)
            if rank == 0:
                # Never overwrite a resumable checkpoint in place. Commit a pointer last.
                name = f"{label}-step-{step:06d}"
                final = run / name
                if not final.exists():
                    temp = run / (name + ".partial")
                    adapter.save(temp, {"format": "bwhisper-lora-v1", "model": L.name, "step": step,
                                       "epoch": epoch, "target_style": cfg.get("target_style", "cardinals"), "training_config": cfg})
                    state = {"optimizer": optimizer.state_dict(), "step": step, "next_epoch": epoch,
                             "next_batch": batch, "seen": seen, "rng": states, "world_size": world,
                             "training_fingerprint": manifest["training_fingerprint"]}
                    torch.save(state, temp / "training_state.pt")
                    atomic_json(temp / "resume_meta.json", {"training_fingerprint": manifest["training_fingerprint"]})
                    temp.rename(final)
                atomic_json(run / "resume.json", {"path": name})
                telemetry.append("checkpoints.jsonl", {"path": name, "step": step, "epoch": epoch})
            dist.barrier()

        def evaluate(epoch):
            nonlocal last_eval
            telemetry.update(stage="evaluation" if rank == 0 else "waiting", step=step)
            if rank == 0:
                begin = time.monotonic()
                if evaluate_fn:
                    last_eval = evaluate_fn(L, adapter, epoch, step)
                else:
                    last_eval = evaluate_rank0(L, adapter, request, telemetry, epoch, step, directory)
                last_eval["evaluation_seconds"] = time.monotonic() - begin
                telemetry.append("evaluations.jsonl", last_eval)
            dist.barrier()

        evaluate(next_epoch - 1)
        for epoch in range(next_epoch, cfg["epochs"] + 1):
            order = list(ids)
            random.Random(cfg["seed"] + epoch).shuffle(order)
            batches = list(global_batches(order, cfg["global_batch_size"], rank, world))
            for batch_index, (slots, valid_count) in enumerate(batches):
                if epoch == next_epoch and batch_index < next_batch:
                    continue
                if stop_requested():
                    checkpoint("cancelled", epoch, batch_index)
                    telemetry.update(stage="cancelled")
                    return None
                telemetry.update(stage="training", step=step, epoch=epoch)
                L.model.train()
                optimizer.zero_grad(set_to_none=True)
                local_loss = 0.
                audio_seconds = forward_seconds = 0.
                synchronize(device)
                update_start = time.monotonic()
                for micro, sid in enumerate(slots):
                    valid = sid is not None
                    sid = sid if valid else order[0]
                    before = time.monotonic()
                    features = feature_fn(sid)
                    synchronize(device)
                    audio_seconds += time.monotonic() - before
                    context = student.no_sync() if micro + 1 < len(slots) else nullcontext()
                    precision = torch.autocast("cuda", dtype=torch.bfloat16) if cfg["bf16"] and use_cuda else nullcontext()
                    before = time.monotonic()
                    with context, precision:
                        loss = teacher_loss(L, features, targets[sid]["tokens"], forward_model=student)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite distributed loss")
                        # DDP averages gradients by world; undo that factor then
                        # normalize by the actual global tail size, not padded slots.
                        (loss * (world / valid_count if valid else 0.)).backward()
                    synchronize(device)
                    forward_seconds += time.monotonic() - before
                    local_loss += float(loss.detach()) if valid else 0.
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
                lr = cfg["learning_rate"] * min(1., (step + 1) / max(1, cfg["warmup_steps"]))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                step += 1
                seen += valid_count
                local_seen += sum(sid is not None for sid in slots)
                total_loss = torch.tensor(local_loss, device=device)
                dist.all_reduce(total_loss)
                synchronize(device)
                telemetry.update(done=local_seen, global_done=seen, step=step, epoch=epoch, audio_seconds=audio_seconds,
                                 student_seconds=forward_seconds, update_seconds=time.monotonic() - update_start)
                if rank == 0:
                    elapsed = time.monotonic() - start
                    telemetry.append("metrics.jsonl", {"step": step, "epoch": epoch, "loss": total_loss.item() / valid_count,
                        "learning_rate": lr, "grad_norm": float(norm), "seen": seen,
                        "accepted_presentations": seen, "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
                        "student_clips_per_second": (seen-initial_seen) / max(.001, elapsed), "updates_per_second": (step-initial_step) / max(.001, elapsed),
                        "audio_seconds_rank0": audio_seconds, "student_seconds_rank0": forward_seconds,
                        "update_seconds": time.monotonic() - update_start, "elapsed_seconds": elapsed})
                if step % cfg["checkpoint_every_steps"] == 0:
                    checkpoint("update", epoch, batch_index + 1)
                if cfg["eval_every_steps"] and step % cfg["eval_every_steps"] == 0:
                    evaluate(epoch)
            checkpoint(f"epoch-{epoch:02d}", epoch + 1, 0)
            evaluate(epoch)
        if rank == 0:
            atomic_json(directory / "result.json", {"model": L.name, "updates": step, "evaluation": last_eval,
                        "target_style": cfg.get("target_style", "cardinals"),
                        "adapter": json.loads((run / "resume.json").read_text())["path"],
                        "world_size": world, "global_batch_size": cfg["global_batch_size"]})
        telemetry.update(stage="done", step=step)
        return step
    finally:
        adapter.restore()


def evaluate_rank0(L, adapter, request, telemetry, epoch, step, directory):
    rows, baseline = [], []
    training = L.model.training
    L.model.eval()
    try:
        for sid in request["manifest"]["eval_ids"]:
            # All ranks continue to the next boundary on cancellation; no rank-only
            # exception that leaves peers waiting in a barrier.
            if (directory / "STOP").exists():
                break
            audio = load_audio(request["samples"][str(sid)]["audio_path"])
            with adapter.teacher():
                text, result = M.transcribe(L, audio, request["cfg"]["max_new_tokens"])
            baseline.append({"sample_id": sid, "text": text, "ended": result.ended})
            text, result = M.transcribe(L, audio, request["cfg"]["max_new_tokens"])
            rows.append({"sample_id": sid, "text": text, "ended": result.ended})
            telemetry.append("evaluation.jsonl", {**rows[-1], "baseline": baseline[-1]["text"], "epoch": epoch, "step": step})
            telemetry.update(stage="evaluation", eval_done=len(rows), eval_total=len(request["manifest"]["eval_ids"]))
        return {"epoch": epoch, "step": step, "baseline": prevalence(baseline), "adapter": prevalence(rows),
                "suppression_at_evaluation": False, "complete": len(rows) == len(request["manifest"]["eval_ids"])}
    finally:
        L.model.train(training)


def train(request, run):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    directory = run / "distributed"
    telemetry = Telemetry(directory, rank, request["cfg"]["devices"][rank], request.get("parent_pid"))
    try:
        L = load(request, rank)
        if request["cfg"]["bf16"] and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 unsupported on worker device")
        dist.init_process_group("nccl", timeout=timedelta(seconds=request["cfg"]["worker_timeout_seconds"]),
                                device_id=torch.device(L.device))
        train_engine(L, request, run, rank, world, telemetry)
    except BaseException as error:
        telemetry.update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        telemetry.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    mode, path = sys.argv[1:3]
    request_path = Path(path)
    request = json.loads(request_path.read_text())
    if mode == "teacher":
        teacher(request, request_path.parent, int(sys.argv[3]))
    elif mode == "train":
        train(request, request_path.parent)
    else:
        raise ValueError("unknown Phase 4 worker mode")
