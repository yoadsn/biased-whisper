"""Background job runner. Every phase action is a job with a directory under
data/runs/<id>/ holding job.json (state), log.txt and phase-specific outputs.
The UI, the CLI and humans/agents all read the same files, so state is shared.
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
import fcntl
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .config import RUNS_DIR, ensure_dirs

REGISTRY: dict[str, dict[str, Any]] = {}


@contextmanager
def gpu_reservation(job_type, config):
    """Cross-process advisory leases shared by every harness GPU job."""
    from .config import DATA, load_settings
    from .model import pick_device

    devices = config.get("devices") or []
    if job_type == "phase4" and not config.get("parallel"):
        devices = []
    if not devices:
        preferred = config.get("device") or load_settings()["device"]
        if preferred == "auto":
            preferred = load_settings()["device"]
        device = pick_device(preferred)
        devices = ["cuda:0" if device == "cuda" else device]
    directory = DATA / "gpu_locks"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for device in sorted(set(devices)):
            if not str(device).startswith("cuda"):
                continue
            # Map logical indices to physical visibility labels for cross-process leases.
            index = int(str(device).split(":")[1]) if ":" in str(device) else 0
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            physical = visible.split(",")[index] if visible else str(index)
            if not physical.replace("-", "").isalnum():
                raise ValueError("invalid CUDA visibility label")
            handle = stack.enter_context((directory / f"cuda-{physical}.lock").open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"GPU {device} is reserved by another harness job") from None
        yield


def register(name: str, kind: str = "gpu", defaults: dict | None = None):
    """kind: 'gpu' jobs run one at a time; 'io' jobs run on a separate pool."""

    def deco(fn):
        REGISTRY[name] = {"fn": fn, "kind": kind, "defaults": defaults or {}}
        return fn

    return deco


def _json_default(o):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class Cancelled(Exception):
    pass


class Ctx:
    def __init__(self, job: "Job"):
        self.job = job
        self.dir = job.dir

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.job.lock:
            with open(self.dir / "log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(f"[{self.job.id}] {msg}", flush=True)

    def progress(self, done: int, total: int, message: str | None = None) -> None:
        self.job.progress = {"done": done, "total": total}
        if message is not None:
            self.job.message = message
        self.job.save()

    def should_stop(self) -> bool:
        return self.job.cancel_requested()

    def check(self) -> None:
        if self.should_stop():
            raise Cancelled()

    def append_jsonl(self, name: str, obj: Any) -> None:
        with self.job.lock:
            with open(self.dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")

    def write_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))


class Job:
    def __init__(self, id: str, type: str, config: dict, dir: Path):
        self.id, self.type, self.config, self.dir = id, type, config, dir
        self.status = "queued"
        self.progress = {"done": 0, "total": 0}
        self.message = ""
        self.summary: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.lock = threading.RLock()
        self.cancel_flag = threading.Event()
        self.external = False  # created by another process (e.g. the CLI); state is read from job.json
        self.mtime = 0.0

    def cancel_requested(self) -> bool:
        # a marker file lets the UI cancel a run owned by another process
        return self.cancel_flag.is_set() or (self.dir / "cancel.flag").exists()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "config": self.config, "status": self.status,
            "progress": self.progress, "message": self.message, "summary": self.summary,
            "error": self.error, "created": self.created, "started": self.started, "finished": self.finished,
        }

    def save(self) -> None:
        with self.lock:
            tmp = self.dir / "job.json.tmp"
            tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
            tmp.replace(self.dir / "job.json")


class Manager:
    def __init__(self):
        ensure_dirs()
        self.jobs: dict[str, Job] = {}
        self.pools = {"gpu": ThreadPoolExecutor(1, thread_name_prefix="gpu"), "io": ThreadPoolExecutor(2, thread_name_prefix="io")}
        self._load_existing()

    _FIELDS = ("status", "progress", "message", "summary", "error", "created", "started", "finished")

    def _load_existing(self) -> None:
        for p in RUNS_DIR.glob("*/job.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            j = Job(d["id"], d["type"], d["config"], p.parent)
            for k in self._FIELDS:
                setattr(j, k, d.get(k))
            if j.status in ("running", "queued"):
                j.status, j.error = "interrupted", "server restarted while job was active"
                j.save()
            j.external = True
            j.mtime = p.stat().st_mtime
            self.jobs[j.id] = j

    def refresh(self) -> None:
        """Pick up runs written by other processes (CLI) after this server started, and their progress.
        Jobs started by this process are authoritative and never overwritten."""
        for p in RUNS_DIR.glob("*/job.json"):
            j = self.jobs.get(p.parent.name)
            if j is not None and not j.external:
                continue
            try:
                m = p.stat().st_mtime
                if j is not None and m == j.mtime:
                    continue
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue  # mid-write or removed; next poll retries
            if j is None:
                j = Job(d["id"], d["type"], d["config"], p.parent)
                j.external = True
                self.jobs[j.id] = j
            for k in self._FIELDS:
                setattr(j, k, d.get(k))
            j.mtime = m

    def new_job(self, type: str, config: dict) -> Job:
        if type not in REGISTRY:
            raise KeyError(f"unknown job type {type!r}; known: {sorted(REGISTRY)}")
        cfg = {**REGISTRY[type]["defaults"], **(config or {})}
        label = f"phase4-{cfg.get('target_style', 'cardinals')}" if type == "phase4" and cfg.get("target_style", "cardinals") in ("cardinals", "digits") else type
        jid = f"{label}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        d = RUNS_DIR / jid
        d.mkdir(parents=True, exist_ok=True)
        j = Job(jid, type, cfg, d)
        self.jobs[jid] = j
        j.save()
        return j

    def _run(self, j: Job) -> None:
        ctx = Ctx(j)
        j.status, j.started = "running", time.time()
        j.save()
        lease = ExitStack()
        try:
            if REGISTRY[j.type]["kind"] == "gpu":
                lease.enter_context(gpu_reservation(j.type, j.config))
            j.summary = REGISTRY[j.type]["fn"](ctx, j.config)
            j.status = "cancelled" if j.cancel_requested() else "done"
        except Cancelled:
            j.status = "cancelled"
        except Exception as e:  # noqa: BLE001
            j.status, j.error = "failed", f"{type(e).__name__}: {e}"
            ctx.log("FAILED\n" + traceback.format_exc())
        # Run after the job frame (including its tensors/exception traceback) has
        # unwound. Clearing the cache inside a job's finally still leaves its
        # locals alive, so empty_cache there cannot release the working set.
        if REGISTRY[j.type]["kind"] == "gpu" and j.config.get("release_model_after_run", False):
            from . import model as M

            try:
                cleanup = M.unload_all()
                ctx.write_json("model_cleanup.json", cleanup)
                ctx.log("released model cache and unused device memory after job")
            except Exception as e:  # noqa: BLE001
                j.status = "failed"
                j.error = (j.error + "; " if j.error else "") + f"model cleanup failed: {e}"
                ctx.log("CLEANUP FAILED\n" + traceback.format_exc())
        lease.close()
        j.finished = time.time()
        j.save()

    def submit(self, type: str, config: dict | None = None) -> Job:
        j = self.new_job(type, config or {})
        self.pools[REGISTRY[type]["kind"]].submit(self._run, j)
        return j

    def run_sync(self, type: str, config: dict | None = None) -> Job:
        j = self.new_job(type, config or {})
        self._run(j)
        return j

    def get(self, jid: str) -> Job | None:
        self.refresh()
        return self.jobs.get(jid)

    def list(self, type: str | None = None) -> list[dict]:
        self.refresh()
        js = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.to_dict() for j in js if type in (None, "", j.type)]

    def cancel(self, jid: str) -> bool:
        self.refresh()
        j = self.jobs.get(jid)
        if j and j.status in ("queued", "running"):
            if j.external:
                (j.dir / "cancel.flag").write_text("cancel requested\n")
            j.cancel_flag.set()
            return True
        return False

    def log_tail(self, jid: str, offset: int = 0) -> tuple[str, int]:
        p = RUNS_DIR / jid / "log.txt"
        if not p.exists():
            return "", 0
        b = p.read_bytes()
        return b[offset:].decode("utf-8", "replace"), len(b)


_manager: Manager | None = None


def manager() -> Manager:
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


def read_jsonl(path: Path, offset: int = 0, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
