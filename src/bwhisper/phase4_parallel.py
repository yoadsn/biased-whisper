"""Parent-only orchestration. Children never create a Job Manager or touch its files."""
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

from .config import DATA
from .jobs import Cancelled


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_identity(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in (".json", ".safetensors", ".bin", ".txt", ".model")))
    if not files:
        raise ValueError("checkpoint identity requires a local model snapshot")
    return {p.name: file_hash(p) for p in files}


def teacher_identity(model_path, cfg, language_settings):
    from . import phase1, phase4, model, hebrew_numbers, audio, number_values
    import torch
    import transformers
    keys = ("max_new_tokens", "min_rest_match", "include_natural_cardinal", "include_no_number", "value_filter")
    return {"version": 1, "model": checkpoint_identity(model_path),
            "settings": language_settings, "filters": {k: cfg[k] for k in keys}, "seed": cfg["seed"],
            "source": {m.__name__: file_hash(m.__file__) for m in
                       (phase1, phase4, model, hebrew_numbers, audio, number_values)},
            "worker": file_hash(Path(__file__).with_name("phase4_worker.py")),
            "libraries": {"torch": torch.__version__, "transformers": transformers.__version__},
            "precision": "base-float32", "satellites": "built-in"}


def sample_identity(sample):
    return digest({"id": sample["id"], "source": sample["source"], "meta": sample["meta"],
                   "audio": file_hash(sample["audio_path"])})


def digit_source(cfg, manifest, model_path, settings):
    """Explicitly derive a new objective from a frozen, checksum-verified old cache.

    Do not pretend old outcomes have today's generation-code identity. Their
    original fingerprint remains provenance of the accepted pair selection.
    """
    from .phase4 import run_dir
    directory = run_dir(cfg["teacher_source_run"])
    job = json.loads((directory / "job.json").read_text())
    if job["type"] != "phase4" or job["status"] != "done" or job["config"].get("target_style", "cardinals") != "cardinals":
        raise ValueError("teacher_source_run must be a completed cardinal Phase 4 run")
    request = json.loads((directory / "request.json").read_text())
    source_manifest = request["manifest"]
    for key in ("cohort_run", "cohort_sha256", "train_ids", "eval_ids", "seed"):
        if manifest.get(key) != source_manifest.get(key):
            raise ValueError(f"teacher source cohort mismatch: {key}")
    for key in ("seed", "max_new_tokens", "min_rest_match", "value_filter"):
        if cfg[key] != job["config"][key]:
            raise ValueError(f"teacher source selection mismatch: {key}")
    fingerprint = source_manifest["teacher_fingerprint"]
    # Resolve only under this project's cache, never a request-supplied filesystem path.
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("invalid source teacher fingerprint")
    cache = DATA / "phase4_cache" / fingerprint
    source_identity = json.loads((cache / "identity.json").read_text())
    if digest(source_identity) != fingerprint:
        raise ValueError("source teacher identity checksum mismatch")
    if source_identity["model"] != checkpoint_identity(model_path):
        raise ValueError("source teacher base/tokenizer content mismatch")
    if source_identity["settings"] != {k: settings[k] for k in ("language", "timestamps")}:
        raise ValueError("source teacher language/timestamp mismatch")
    identity = {"version": 1, "target_style": "digits", "source_teacher_fingerprint": fingerprint,
                "source_run": cfg["teacher_source_run"], "include_no_number": cfg["include_no_number"],
                "derivation_code": file_hash(__file__), "model": source_identity["model"],
                "settings": source_identity["settings"], "tokenization": "baseline-text-no-special-tokens-v1"}
    return identity, cache, request


def digit_target(row, cfg, tokenizer):
    from .hebrew_numbers import DIGIT_RE, is_degenerate
    old = row.get("target")
    if old is None:
        return None, row["reason"]
    if old["kind"] == "preservation":
        return (old, "accepted") if cfg["include_no_number"] else (None, "no_number_disabled")
    if old["kind"] != "suppressed_cardinal":
        return None, "natural_cardinal_excluded"
    text = row.get("baseline", "")
    if not text.strip() or not DIGIT_RE.search(text) or is_degenerate(text):
        return None, "invalid_digit_baseline"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if not tokens or any(type(t) is not int or not 0 <= t < eos for t in tokens):
        return None, "invalid_target_tokens"
    if tokenizer.decode(tokens, skip_special_tokens=False).strip() != text.strip():
        raise ValueError("digit baseline tokenization did not round-trip")
    return {"text": text, "tokens": tokens, "kind": "natural_digits"}, "accepted"


def cache_entry(path, fingerprint, sample_key):
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"corrupt teacher cache: {path}") from e
    content = {k: v for k, v in row.items() if k != "checksum"}
    if (row.get("fingerprint") != fingerprint or row.get("sample_key") != sample_key or
            row.get("checksum") != digest(content)):
        raise ValueError(f"incompatible/corrupt teacher cache: {path}")
    return row


def save_cache(path, row):
    atomic_json(path, {**row, "checksum": digest(row)})


@contextmanager
def cache_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("this teacher cache is already being prepared by another job") from None
        yield


def task_queue(path, samples):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, payload TEXT, claimed INTEGER DEFAULT 0)")
        db.executemany("INSERT INTO tasks(id,payload) VALUES (?,?)", [(s["id"], json.dumps(s)) for s in samples])


@contextmanager
def local_task_queue(path, samples):
    """SQLite's live locks/journals must stay off the NFS run directory.

    Workers retain their existing tasks.db interface via a symlink. SQLite
    resolves it to the local file, including for rollback journals. Once all
    workers have exited, retain a diagnostic snapshot in the run directory.
    Completed teacher outcomes, not this transient queue, drive recovery.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="phase4-tasks-", dir="/tmp/opencode") as scratch:
        local = Path(scratch) / "tasks.db"
        task_queue(local, samples)
        path.symlink_to(local)
        try:
            yield
        finally:
            snapshot = path.with_name("tasks.snapshot.tmp")
            try:
                with sqlite3.connect(local) as source, sqlite3.connect(snapshot) as target:
                    source.backup(target)
                snapshot.replace(path)
            finally:
                snapshot.unlink(missing_ok=True)


def claim_task(db):
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT id,payload FROM tasks WHERE claimed=0 ORDER BY id LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE tasks SET claimed=1 WHERE id=?", (row[0],))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return json.loads(row[1]) if row else None


def child_processes(pid):
    """Remember only descendants of a process we launched, including start IDs."""
    found = {}
    try:
        children = (Path(f"/proc/{pid}/task/{pid}/children").read_text()).split()
    except FileNotFoundError:
        return found
    for child in children:
        try:
            # Fields after the final ')' start at proc stat field 3; starttime is 22.
            identity = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            continue
        found[int(child)] = identity
        found.update(child_processes(int(child)))
    return found


def supervise(ctx, commands, directory, timeout, on_update, cancel_grace=90, environment=None):
    """Event-driven subprocess supervision: filesystem notifications + pidfds.

    Cooperative stop files allow synchronized student checkpoints. Failed/hung
    process groups are terminated, with bounded SIGKILL fallback. No shell fork
    inherits live CUDA state; Popen immediately execs a fresh interpreter.
    """
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError("inotify unavailable")
    libc.inotify_add_watch(fd, os.fsencode(directory), 0x8 | 0x80)
    workers, files = [], []
    launched_at = time.time()
    stop = directory / "STOP"
    cancelled_at = None
    terminal_at = None
    try:
        for index, command in enumerate(commands):
            log = (directory / f"process-{index}.log").open("w")
            files.append(log)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                   "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", **(environment or {})}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            workers.append(process)
        while True:
            if ctx.should_stop() and cancelled_at is None:
                stop.touch()
                cancelled_at = time.monotonic()
            states = on_update()
            now = time.time()
            errors = [s for s in states if s.get("error")]
            if errors:
                raise RuntimeError(f"worker failure: {errors[0]['error']}")
            for i, worker in enumerate(workers):
                code = worker.poll()
                if code not in (None, 0):
                    log = (directory / f"process-{i}.log").read_text()[-6000:]
                    raise RuntimeError(f"worker process {i} exited {code}: {log}")
            if all(w.poll() == 0 for w in workers):
                if cancelled_at is not None:
                    raise Cancelled()
                return
            if states and all(s.get("stage") in ("done", "cancelled", "waiting") for s in states):
                terminal_at = terminal_at or time.monotonic()
                if time.monotonic() - terminal_at > min(30, timeout):
                    raise TimeoutError("worker shutdown timed out")
            else:
                terminal_at = None
            if cancelled_at is not None and time.monotonic() - cancelled_at > cancel_grace:
                raise Cancelled()
            # Heartbeats run independently of slow model forwards; progress timeout
            # additionally catches a collective that is alive but no longer advancing.
            for state in states:
                if state.get("stage") not in ("done", "cancelled", "waiting") and now - state.get("progress_at", now) > timeout:
                    raise TimeoutError(f"worker {state.get('rank')} stalled during {state.get('stage')}")
            if not states and now - launched_at > timeout:
                raise TimeoutError("worker startup timed out")
            select.select([fd], [], [], .5)
            try:
                os.read(fd, 65536)
            except BlockingIOError:
                pass
    finally:
        descendants = {}
        for worker in workers:
            descendants.update(child_processes(worker.pid))
        for worker in workers:
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        # torchrun may create separate child sessions. Its normal SIGTERM handler
        # propagates shutdown; this also catches descendants after a hung launcher.
        for pid, identity in descendants.items():
            try:
                current = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                if current == identity:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        for file in files:
            file.close()
        os.close(fd)


def validate_parallel(cfg, cuda_count):
    devices = cfg.get("devices")
    if (not isinstance(devices, list) or not devices or len(devices) != len(set(devices)) or
            any(d not in [f"cuda:{i}" for i in range(cuda_count)] for d in devices)):
        raise ValueError("devices must be distinct available CUDA devices")
    batch = cfg["global_batch_size"]
    if type(batch) is not int or batch < len(devices) or batch % len(devices):
        raise ValueError("global_batch_size must be a positive multiple of the GPU count")
    for key in ("checkpoint_every_steps", "worker_timeout_seconds", "cpu_threads_per_worker"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["nccl_p2p_level"] not in ("auto", "NVL", "LOC", "PIX", "PXB", "PHB", "SYS"):
        raise ValueError("invalid nccl_p2p_level")


def run_parallel(ctx, cfg):
    import torch
    from . import model as M
    from .phase4 import select_cohorts, run_dir
    from .config import load_settings

    validate_parallel(cfg, torch.cuda.device_count())
    resume = None
    if cfg.get("resume_run"):
        previous = run_dir(cfg["resume_run"])
        job = json.loads((previous / "job.json").read_text())
        if job["status"] in ("running", "queued") or job["type"] != "phase4":
            raise ValueError("resume source must be an inactive Phase 4 run")
        if not job["config"].get("parallel") or not (previous / "resume.json").exists():
            raise ValueError("resume source has no distributed training checkpoint; leave resume_run empty to reuse the teacher cache")
        pointer = json.loads((previous / "resume.json").read_text())
        resume = previous / pointer["path"]
        if previous.resolve() not in resume.resolve().parents:
            raise ValueError("invalid resume checkpoint")
    M.unload_all()  # the parent must not pin an old cached inference model on worker devices
    samples, manifest = select_cohorts(cfg)
    cfg["cohort_run"] = manifest["cohort_run"]
    cfg["model"] = cfg["model"] or load_settings()["model_active"]
    status = {"stage": "fingerprinting", "epoch": 0, "step": 0, "seen": 0, "accepted_presentations": 0,
              "counts": {}, "train_samples": len(manifest["train_ids"]), "eval_samples": len(manifest["eval_ids"]),
               "target_style": cfg.get("target_style", "cardinals"),
               "trainable_parameters": 0, "world_size": len(cfg["devices"]), "global_batch_size": cfg["global_batch_size"]}
    start = time.monotonic()
    offsets = {}
    last_stage = None
    def publish(stage=None):
        nonlocal last_stage
        status["elapsed_seconds"] = time.monotonic() - start
        if stage:
            status["stage"] = stage
        if status["stage"] != last_stage:
            ctx.log(f"parallel Phase 4: {status['stage']}; devices={cfg['devices']}, global_batch={cfg['global_batch_size']}")
            last_stage = status["stage"]
        atomic_json(ctx.dir / "monitor.json", status)
        total = (status.get("accepted_targets", len(manifest["train_ids"])) * cfg["epochs"]
                 if status["stage"] not in ("fingerprinting", "teacher_generation") else len(manifest["train_ids"]))
        ctx.progress(status["seen"], total,
                     f"{status['stage']}: epoch {status['epoch']}/{cfg['epochs']}, update {status['step']}")
    publish()
    model_path = Path(cfg["model"])
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(cfg["model"]))
    settings = load_settings()
    model_config = json.loads((model_path / "config.json").read_text())
    if cfg["encoder_top_layers"] > model_config.get("encoder_layers", 0):
        raise ValueError("encoder_top_layers exceeds model depth")
    prefix_size = 3 if settings.get("timestamps") else 4
    if cfg["max_new_tokens"] + prefix_size > model_config.get("max_target_positions", 448):
        raise ValueError("max_new_tokens exceeds decoder context")
    source = None
    if cfg.get("target_style") == "digits":
        identity, source_cache, source = digit_source(cfg, manifest, model_path, settings)
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        identity = teacher_identity(model_path, cfg, {k: settings[k] for k in ("language", "timestamps")})
    fingerprint = digest(identity)
    cfg["model"] = str(model_path.resolve())
    ctx.job.save()
    cache = DATA / "phase4_cache" / fingerprint
    manifest.update(teacher_fingerprint=fingerprint, target_style=cfg.get("target_style", "cardinals"),
                    teacher_source_run=cfg.get("teacher_source_run"), world_size=len(cfg["devices"]),
                    global_batch_size=cfg["global_batch_size"])
    ctx.write_json("manifest.json", manifest)
    selected = [samples[i] for i in manifest["train_ids"]]
    keys = {}
    identity_samples = selected + [samples[i] for i in manifest["eval_ids"]]
    for i, sample in enumerate(identity_samples):
        ctx.check()
        keys[sample["id"]] = sample_identity(sample)
        if i % 100 == 0:
            status["fingerprinted_samples"] = i + 1
            publish()
    request = {"parent_pid": os.getpid(), "cfg": cfg, "manifest": manifest, "samples": {str(i): samples[i] for i in manifest["train_ids"] + manifest["eval_ids"]},
               "sample_keys": keys, "cache": str(cache), "settings": settings}
    with cache_lock(cache):
        atomic_json(cache / "identity.json", identity)
        if source is not None:
            source_fingerprint = source["manifest"]["teacher_fingerprint"]
            for i, sample in enumerate(selected):
                ctx.check()
                key = keys[sample["id"]]
                if source["sample_keys"].get(str(sample["id"])) != key:
                    raise ValueError(f"source sample identity changed: {sample['id']}")
                original = cache_entry(source_cache / f"{sample['id']}-{key}.json", source_fingerprint, key)
                if original is None:
                    raise ValueError(f"missing source teacher outcome: {sample['id']}")
                destination = cache / f"{sample['id']}-{key}.json"
                if cache_entry(destination, fingerprint, key) is None:
                    target, reason = digit_target(original, cfg, tokenizer)
                    if target and len(target["tokens"]) + prefix_size > model_config.get("max_target_positions", 448):
                        target, reason = None, "invalid_target_tokens"
                    save_cache(destination, {"sample_id": sample["id"], "fingerprint": fingerprint,
                        "sample_key": key, "baseline": original["baseline"], "target": target, "reason": reason,
                        "verdict": original["verdict"], "source_teacher_fingerprint": source_fingerprint,
                        "audio_seconds": 0., "teacher_seconds": 0.})
                if i % 100 == 0:
                    status["derived_samples"] = i + 1
                    publish("deriving_digit_targets")
            ctx.log(f"derived digit targets from {cfg['teacher_source_run']}; no teacher inference")
        pending, hits, cached_counts = [], 0, Counter()
        for sample in selected:
            row = cache_entry(cache / f"{sample['id']}-{keys[sample['id']]}.json", fingerprint, keys[sample['id']])
            if row is None:
                pending.append(sample)
            else:
                hits += 1
                cached_counts[row["target"]["kind"] if row["target"] else row["reason"]] += 1
        status.update(cache_hits=hits, cache_misses=len(pending))
        teacher_dir = ctx.dir / "teacher"
        teacher_dir.mkdir()
        atomic_json(ctx.dir / "request.json", request)

        def collect(directory, stage):
            states = [json.loads(p.read_text()) for p in sorted(directory.glob("rank-*.json"))]
            status["workers"] = states
            if stage == "teacher_generation":
                done = sum(s.get("done", 0) for s in states)
                status["seen"] = hits + done
                seconds = sum(s.get("teacher_seconds", 0) for s in states)
                status["teacher_worker_seconds"] = seconds
                status["teacher_clips_per_second"] = done / max(.001, time.monotonic() - teacher_start)
                status["accepted_targets_per_second"] = sum(sum(s.get("counts", {}).get(k, 0) for k in
                    ("natural_cardinal", "suppressed_cardinal", "preservation")) for s in states) / max(.001, time.monotonic() - teacher_start)
                status["counts"] = dict(sum((Counter(s.get("counts", {})) for s in states), cached_counts.copy()))
            else:
                for name in ("metrics.jsonl", "evaluations.jsonl", "checkpoints.jsonl", "baseline.jsonl", "evaluation.jsonl"):
                    path = directory / name
                    if path.exists():
                        with path.open() as f:
                            f.seek(offsets.get(name, 0))
                            while True:
                                pos = f.tell()
                                line = f.readline()
                                if not line or not line.endswith("\n"):
                                    offsets[name] = pos
                                    break
                                value = json.loads(line)
                                ctx.append_jsonl(name, value)
                                if name == "metrics.jsonl":
                                    status.update({k: value[k] for k in ("epoch", "step", "loss", "learning_rate", "seen", "accepted_presentations", "trainable_parameters", "student_clips_per_second", "updates_per_second")})
                                elif name == "evaluations.jsonl":
                                    status["evaluation"] = value
                                elif name == "checkpoints.jsonl":
                                    ctx.log(f"saved checkpoint {value['path']} at update {value['step']}")
                if states and states[0].get("stage") in ("evaluation", "checkpoint", "waiting"):
                    stage = states[0]["stage"]
            publish(stage)
            return states

        teacher_start = time.monotonic()
        if pending:
            commands = [[sys.executable, "-m", "bwhisper.phase4_worker", "teacher", str(ctx.dir / "request.json"), str(rank)]
                        for rank in range(len(cfg["devices"]))]
            with local_task_queue(teacher_dir / "tasks.db", pending):
                supervise(ctx, commands, teacher_dir, cfg["worker_timeout_seconds"], lambda: collect(teacher_dir, "teacher_generation"))
        ctx.check()
        rows = []
        for sample in selected:
            row = cache_entry(cache / f"{sample['id']}-{keys[sample['id']]}.json", fingerprint, keys[sample['id']])
            if row is None:
                raise ValueError(f"teacher outcome missing for sample {sample['id']}")
            rows.append(row)
            ctx.append_jsonl("targets.jsonl", row)
        accepted = [r for r in rows if r["target"] is not None]
        if not accepted:
            raise ValueError("no accepted teacher targets")
        ctx.log(f"teacher cache complete: {len(accepted)}/{len(rows)} accepted, {hits} cache hits")
        request["accepted"] = accepted
        manifest["accepted_ids"] = [r["sample_id"] for r in accepted]
        train_keys = ("rank", "alpha", "dropout", "target_modules", "encoder_top_layers", "learning_rate", "weight_decay",
                      "max_grad_norm", "warmup_steps", "global_batch_size", "seed", "bf16", "epochs")
        manifest["training_fingerprint"] = digest({"teacher": fingerprint, "keys": keys, "ids": manifest["accepted_ids"],
            "eval_ids": manifest["eval_ids"], "world_size": len(cfg["devices"]), "config": {k: cfg[k] for k in train_keys},
            "adapter_code": file_hash(Path(__file__).with_name("lora.py")),
            "target_content": digest([{k: r[k] for k in ("sample_id", "target")} for r in accepted])})
        request["manifest"] = manifest
        if resume is not None:
            meta = json.loads((resume / "resume_meta.json").read_text())
            if meta["training_fingerprint"] != manifest["training_fingerprint"]:
                raise ValueError("incompatible training resume (world size, inputs or training config changed)")
            request["resume_checkpoint"] = str(resume)
        atomic_json(ctx.dir / "request.json", request)
        ctx.write_json("manifest.json", manifest)
        status["counts"] = dict(Counter(r["target"]["kind"] if r["target"] else r["reason"] for r in rows))
        status.update(accepted_targets=len(accepted), seen=0, teacher_seconds=time.monotonic() - teacher_start)
        training_dir = ctx.dir / "distributed"
        training_dir.mkdir()
        publish("student_startup")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(len(cfg["devices"])),
                   "--module", "bwhisper.phase4_worker", "train", str(ctx.dir / "request.json")]
        environment = {} if cfg["nccl_p2p_level"] == "auto" else {"NCCL_P2P_LEVEL": cfg["nccl_p2p_level"]}
        supervise(ctx, [command], training_dir, cfg["worker_timeout_seconds"], lambda: collect(training_dir, "training"), environment=environment)
        collect(training_dir, "training")
        ctx.check()
        result = json.loads((training_dir / "result.json").read_text())
        publish("complete")
        return result
