"""FastAPI app: thin HTTP layer over the same job registry the CLI uses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import jobs
from . import phase1 as _p1  # noqa: F401  (registers job types)
from . import phase2 as _p2  # noqa: F401
from . import phase3 as _p3  # noqa: F401
from . import phase1_5 as _p15  # noqa: F401
from . import budget as _bud  # noqa: F401
from . import phase2n as _p2n  # noqa: F401
from .config import KNOWLEDGE_DIR, RUNS_DIR, WEB_DIR, ensure_dirs, load_settings, save_settings
from .corpus import explore, hf_cache, store
from .corpus import jobs as _corpus_jobs  # noqa: F401
from .jobs import REGISTRY, manager, read_jsonl
from .model import pick_device
from .paramspace import LightSpace, PRESETS
from .phase1 import decision_points, load_verdicts, set_verdict

ensure_dirs()
app = FastAPI(title="biased-whisper harness")


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _job_dir(jid: str) -> Path:
    d = RUNS_DIR / jid
    if not d.is_dir() or ".." in jid:
        raise HTTPException(404, "no such run")
    return d


# ---------------------------------------------------------------- settings / meta
@app.get("/api/settings")
def get_settings():
    s = load_settings()
    return {"settings": s, "resolved_device": pick_device(s["device"]), "param_filters": list(PRESETS)}


@app.post("/api/settings")
def post_settings(patch: dict = Body(...)):
    return save_settings(patch)


@app.get("/api/job-types")
def job_types():
    return {k: {"kind": v["kind"], "defaults": v["defaults"]} for k, v in REGISTRY.items()}


# ---------------------------------------------------------------- corpus
@app.get("/api/corpus")
def corpus_list(source: str | None = None, q: str | None = None, digits_only: bool = False,
                limit: int = 50, offset: int = 0, order: str = "id"):
    rows = store.query(source=source or None, q=q or None, digits_only=digits_only, limit=limit, offset=offset, order=order)
    for r in rows:
        r.pop("meta", None)
    return {"items": rows, "total": store.count(source or None, digits_only)}


@app.get("/api/corpus/stats")
def corpus_stats():
    return store.stats()


@app.get("/api/corpus/hf-datasets")
def corpus_hf():
    return hf_cache.list_datasets()


@app.get("/api/corpus/{sample_id}")
def corpus_get(sample_id: int):
    s = store.get(sample_id)
    if not s:
        raise HTTPException(404)
    return s


@app.get("/api/corpus/{sample_id}/audio")
def corpus_audio(sample_id: int):
    s = store.get(sample_id)
    if not s or not Path(s["audio_path"]).exists():
        raise HTTPException(404)
    return FileResponse(s["audio_path"], media_type="audio/wav")


@app.put("/api/corpus/{sample_id}")
def corpus_edit(sample_id: int, body: dict = Body(...)):
    store.update_text(sample_id, body["text"])
    return store.get(sample_id)


@app.delete("/api/corpus")
def corpus_delete(ids: list[int] = Body(...)):
    return {"deleted": store.delete(ids)}


# ---------------------------------------------------------------- explore.ivrit.ai
@app.get("/api/explore/config")
def explore_cfg():
    return explore.public_cfg()


@app.post("/api/explore/config")
def explore_cfg_set(patch: dict = Body(...)):
    return explore.save_cfg(patch)


@app.post("/api/explore/curl")
def explore_curl(body: dict = Body(...)):
    return explore.import_curl(body["search_curl"], body.get("audio_curl") or None)


@app.get("/api/explore/check")
def explore_check():
    return explore.check_auth()


@app.get("/api/explore/probe")
def explore_probe():
    return explore.probe()


# ---------------------------------------------------------------- jobs
@app.get("/api/jobs")
def jobs_list(type: str | None = None):
    return manager().list(type)


@app.post("/api/jobs")
def jobs_submit(body: dict = Body(...)):
    try:
        j = manager().submit(body["type"], body.get("config") or {})
    except KeyError as e:
        raise HTTPException(400, str(e))
    return j.to_dict()


@app.get("/api/jobs/{jid}")
def jobs_get(jid: str):
    j = manager().get(jid)
    if not j:
        raise HTTPException(404)
    return j.to_dict()


@app.post("/api/jobs/{jid}/cancel")
def jobs_cancel(jid: str):
    return {"cancelled": manager().cancel(jid)}


@app.get("/api/jobs/{jid}/log")
def jobs_log(jid: str, offset: int = 0):
    text, off = manager().log_tail(jid, offset)
    return {"text": text, "offset": off}


@app.get("/api/jobs/{jid}/rows")
def jobs_rows(jid: str, file: str = "results.jsonl", offset: int = 0, limit: int = 200, verdict: str | None = None):
    d = _job_dir(jid)
    if "/" in file or ".." in file:
        raise HTTPException(400)
    rows = read_jsonl(d / file)
    if verdict:
        rows = [r for r in rows if r.get("verdict") == verdict]
    return {"total": len(rows), "items": rows[offset : offset + limit]}


@app.get("/api/jobs/{jid}/file/{name}")
def jobs_file(jid: str, name: str):
    d = _job_dir(jid)
    if "/" in name or ".." in name or not (d / name).exists() or not name.endswith(".json"):
        raise HTTPException(404)
    return JSONResponse(json.loads((d / name).read_text()))


# ---------------------------------------------------------------- phase specific
@app.get("/api/phase1/{jid}/verdicts")
def p1_verdicts(jid: str):
    return load_verdicts(jid)


@app.post("/api/phase1/{jid}/verdict")
def p1_set_verdict(jid: str, body: dict = Body(...)):
    _job_dir(jid)
    return set_verdict(jid, int(body["sample_id"]), body.get("verdict"))


@app.get("/api/phase1/usable")
def p1_usable(runs: str, verdicts: str = "cardinal,human_good"):
    pts = decision_points([r for r in runs.split(",") if r], verdicts.split(","))
    return {"count": len(pts)}


@app.get("/api/phase2/{jid}/candidates")
def p2_candidates(jid: str, type: str, offset: int = 0, limit: int = 100, tower: str | None = None):
    d = _job_dir(jid)
    f = d / f"candidates_{type}.npz"
    if not f.exists():
        raise HTTPException(404, "no candidates for this nudge type")
    z = np.load(f)
    sp = LightSpace(json.loads((d / "space.json").read_text()))
    n = len(z["idx"])
    items = []
    for j in range(offset, min(n, offset + limit)):
        loc = sp.locate(int(z["idx"][j]))
        items.append({**loc, "votes": int(z["votes"][j]), "null_votes": int(z["null_votes"][j]),
                      **{k: float(z[k][j]) for k in ("mean_all", "mean_present", "median_present", "std_present", "sign_consistency")},
                      "count": int(z["count"][j])})
    return {"total": n, "items": items}


# ---------------------------------------------------------------- knowledge
def _split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end > 0:
            try:
                return yaml.safe_load(text[4:end]) or {}, text[end + 4 :].lstrip("\n")
            except yaml.YAMLError:
                pass
    return {}, text


@app.get("/api/knowledge")
def knowledge_list():
    out = []
    for p in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        rel = p.relative_to(KNOWLEDGE_DIR).as_posix()
        fm, _ = _split_frontmatter(p.read_text())
        out.append({"path": rel, "reserved": p.name in ("index.md", "log.md"), "type": fm.get("type"),
                    "title": fm.get("title") or rel, "description": fm.get("description"), "tags": fm.get("tags") or [],
                    "status": fm.get("status")})
    return out


@app.get("/api/knowledge/doc")
def knowledge_doc(path: str):
    p = (KNOWLEDGE_DIR / path).resolve()
    if KNOWLEDGE_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    fm, body = _split_frontmatter(p.read_text())
    return {"path": path, "frontmatter": fm, "body": body}


# ---------------------------------------------------------------- static UI
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
