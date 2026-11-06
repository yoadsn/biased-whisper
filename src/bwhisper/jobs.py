"""Background job runner. Every phase action is a job with a directory under
data/runs/<id>/ holding job.json (state), log.txt and phase-specific outputs.
The UI, the CLI and humans/agents all read the same files, so state is shared.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .config import RUNS_DIR, ensure_dirs

REGISTRY: dict[str, dict[str, Any]] = {}


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
        return self.job.cancel_flag.is_set()

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

    def _load_existing(self) -> None:
        for p in RUNS_DIR.glob("*/job.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            j = Job(d["id"], d["type"], d["config"], p.parent)
            for k in ("status", "progress", "message", "summary", "error", "created", "started", "finished"):
                setattr(j, k, d.get(k))
            if j.status in ("running", "queued"):
                j.status, j.error = "interrupted", "server restarted while job was active"
                j.save()
            self.jobs[j.id] = j

    def new_job(self, type: str, config: dict) -> Job:
        if type not in REGISTRY:
            raise KeyError(f"unknown job type {type!r}; known: {sorted(REGISTRY)}")
        cfg = {**REGISTRY[type]["defaults"], **(config or {})}
        jid = f"{type}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
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
        try:
            j.summary = REGISTRY[j.type]["fn"](ctx, j.config)
            j.status = "cancelled" if j.cancel_flag.is_set() else "done"
        except Cancelled:
            j.status = "cancelled"
        except Exception as e:  # noqa: BLE001
            j.status, j.error = "failed", f"{type(e).__name__}: {e}"
            ctx.log("FAILED\n" + traceback.format_exc())
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
        return self.jobs.get(jid)

    def list(self, type: str | None = None) -> list[dict]:
        js = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.to_dict() for j in js if type in (None, "", j.type)]

    def cancel(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j and j.status in ("queued", "running"):
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
