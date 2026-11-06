"""Corpus source: explore.ivrit.ai (Google-login protected).

The site's API is not known a priori, so this client is *configuration
driven*. Workflow:

 1. Log in with a browser, copy the request as cURL for (a) a search request
    and (b) an audio-segment request, paste them in the UI ("Explore" card).
    ``import_curl`` extracts cookie + headers + URL templates.
 2. ``probe`` can also crawl the site's own same-origin JS to list ``/api/...``
    paths it calls.
 3. ``fetch_batch`` runs searches with a seed, maps JSON fields via
    ``cfg['mapping']`` and downloads audio ranges.

Cookies are stored in data/secrets/ (gitignored, never committed).
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from ..audio import crop_to_wav
from ..config import SECRETS_DIR, ensure_dirs, load_settings
from . import store

CFG_PATH = SECRETS_DIR / "explore.json"

DEFAULT_CFG: dict[str, Any] = {
    "cookie": "",
    "headers": {},
    # URL templates; {q} {mode} {seed} {page} {offset} {limit} are substituted
    "search_url": "",
    "audio_url": "",  # may contain {source_id} {start} {end} {id}; "" -> use mapping.audio_url_key
    "mapping": {
        "results_path": "results",  # dotted path to the list in the search JSON
        "text": "text",
        "id": "id",
        "start": "start",
        "end": "end",
        "audio_url_key": "",  # field holding a ready-to-use audio url
        "source_id": "source_id",
    },
    "page_size": 40,
    "query": r"\d",
    "mode": "regex",
}


def load_cfg() -> dict:
    ensure_dirs()
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    if CFG_PATH.exists():
        stored = json.loads(CFG_PATH.read_text())
        mapping = {**cfg["mapping"], **stored.get("mapping", {})}
        cfg.update(stored)
        cfg["mapping"] = mapping
    return cfg


def save_cfg(patch: dict) -> dict:
    cfg = load_cfg()
    for k, v in patch.items():
        if k == "mapping":
            cfg["mapping"].update(v)
        elif k in DEFAULT_CFG:
            cfg[k] = v
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    CFG_PATH.chmod(0o600)
    return public_cfg(cfg)


def public_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or load_cfg()
    c = dict(cfg)
    c["cookie"] = f"<{len(cfg['cookie'])} chars>" if cfg["cookie"] else ""
    c["headers"] = {k: ("<hidden>" if k.lower() in ("authorization", "x-csrf-token") else v) for k, v in cfg["headers"].items()}
    return c


def parse_curl(cmd: str) -> dict:
    """Parse a 'Copy as cURL' command into url, headers and cookie."""
    cmd = cmd.replace("\\\n", " ")
    toks = shlex.split(cmd)
    url, headers, cookie = None, {}, ""
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("-H", "--header") and i + 1 < len(toks):
            k, _, v = toks[i + 1].partition(":")
            if k.strip().lower() == "cookie":
                cookie = v.strip()
            elif k.strip().lower() not in ("host", "content-length", "accept-encoding"):
                headers[k.strip()] = v.strip()
            i += 2
            continue
        if t in ("-b", "--cookie") and i + 1 < len(toks):
            cookie = toks[i + 1]
            i += 2
            continue
        if t.startswith("http"):
            url = t
        i += 1
    return {"url": url, "headers": headers, "cookie": cookie}


def import_curl(search_curl: str, audio_curl: str | None = None) -> dict:
    s = parse_curl(search_curl)
    patch: dict[str, Any] = {"cookie": s["cookie"], "headers": s["headers"]}
    if s["url"]:
        u = urlparse(s["url"])
        qs = dict(parse_qsl(u.query))
        for k, v in list(qs.items()):
            if v == r"\d" or v == "\\d":
                qs[k] = "{q}"
            elif k in ("mode",):
                qs[k] = "{mode}"
            elif k == "seed":
                qs[k] = "{seed}"
            elif k in ("page",):
                qs[k] = "{page}"
            elif k in ("offset",):
                qs[k] = "{offset}"
            elif k in ("limit", "size", "page_size"):
                qs[k] = "{limit}"
        patch["search_url"] = urlunparse((u.scheme, u.netloc, u.path, "", "&".join(f"{k}={v}" for k, v in qs.items()), ""))
    if audio_curl:
        a = parse_curl(audio_curl)
        patch["audio_url"] = a["url"] or ""
        if not patch["cookie"]:
            patch["cookie"] = a["cookie"]
    return save_cfg(patch)


def client(cfg: dict | None = None) -> httpx.Client:
    cfg = cfg or load_cfg()
    h = {"User-Agent": "Mozilla/5.0 biased-whisper-harness", **cfg["headers"]}
    if cfg["cookie"]:
        h["Cookie"] = cfg["cookie"]
    return httpx.Client(headers=h, follow_redirects=False, timeout=60)


def _is_login_redirect(r: httpx.Response) -> bool:
    return r.status_code in (301, 302, 303, 307) and "login" in r.headers.get("location", "")


def check_auth() -> dict:
    cfg = load_cfg()
    base = load_settings()["explore_base_url"]
    if not cfg["cookie"]:
        return {"ok": False, "reason": "no cookie configured"}
    with client(cfg) as c:
        r = c.get(base + "/")
        if _is_login_redirect(r):
            return {"ok": False, "reason": "cookie rejected (redirected to login)"}
        return {"ok": r.status_code == 200, "status": r.status_code}


def probe() -> dict:
    """Crawl the site's same-origin scripts and list api-looking paths."""
    cfg = load_cfg()
    base = load_settings()["explore_base_url"]
    found: dict[str, set] = {"api_paths": set(), "scripts": set()}
    with client(cfg) as c:
        r = c.get(base + "/search?q=%5Cd&mode=regex")
        if _is_login_redirect(r):
            return {"ok": False, "reason": "not authenticated: configure cookie first"}
        html = r.text
        for src in re.findall(r'<script[^>]+src="([^"]+)"', html):
            found["scripts"].add(src)
        for src in list(found["scripts"]):
            if src.startswith("http") and not src.startswith(base):
                continue
            try:
                js = c.get(src if src.startswith("http") else base + src).text
            except Exception:
                continue
            for m in re.findall(r"""["'`](/(?:api|search|audio|segment|stream|media)[^"'`\s]{0,120})["'`]""", js):
                found["api_paths"].add(m)
        for m in re.findall(r"""["'](/(?:api)[^"'\s]{0,120})["']""", html):
            found["api_paths"].add(m)
    return {"ok": True, "scripts": sorted(found["scripts"]), "api_paths": sorted(found["api_paths"])}


def _dig(obj: Any, path: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        obj = obj[int(part)] if isinstance(obj, list) else obj.get(part)
        if obj is None:
            return None
    return obj


def fetch_batch(
    seed: int,
    n: int,
    log: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda a, b: None,
    max_pages: int = 50,
) -> dict:
    """Run searches for ``seed`` until ``n`` new samples were stored."""
    cfg = load_cfg()
    m = cfg["mapping"]
    if not cfg["search_url"]:
        raise RuntimeError("explore search_url not configured (paste a cURL of a search request in the UI)")
    base = load_settings()["explore_base_url"]
    added = dup = 0
    ps = cfg["page_size"]
    with client(cfg) as c:
        for page in range(max_pages):
            if added >= n or should_stop():
                break
            url = cfg["search_url"].format(q=cfg["query"], mode=cfg["mode"], seed=seed, page=page, offset=page * ps, limit=ps)
            if url.startswith("/"):
                url = base + url
            r = c.get(url, headers={"Accept": "application/json"})
            if _is_login_redirect(r):
                raise RuntimeError("cookie rejected: re-login and re-import cURL")
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                raise RuntimeError(f"search did not return JSON (first 200 chars): {r.text[:200]!r}")
            items = _dig(data, m["results_path"]) if m["results_path"] else data
            if not isinstance(items, list) or not items:
                log(f"page {page}: no results (path={m['results_path']!r}); keys={list(data)[:10] if isinstance(data, dict) else type(data)}")
                break
            for it in items:
                if added >= n or should_stop():
                    break
                text = _dig(it, m["text"]) or ""
                sid = str(_dig(it, m["id"]) or "")
                st, en = _dig(it, m["start"]), _dig(it, m["end"])
                if not text or not sid:
                    continue
                key = f"{sid}@{st}-{en}"
                if store.exists("explore", key):
                    dup += 1
                    continue
                if cfg["audio_url"]:
                    aurl = cfg["audio_url"].format(id=sid, source_id=_dig(it, m["source_id"]), start=st, end=en)
                elif m["audio_url_key"]:
                    aurl = _dig(it, m["audio_url_key"])
                else:
                    raise RuntimeError("audio_url not configured")
                if aurl.startswith("/"):
                    aurl = base + aurl
                ar = c.get(aurl)
                ar.raise_for_status()
                tmp = store.audio_path_for("explore", key).with_suffix(".src")
                tmp.write_bytes(ar.content)
                dst = store.audio_path_for("explore", key)
                try:
                    dur = crop_to_wav(tmp, dst, None, None)
                finally:
                    tmp.unlink(missing_ok=True)
                store.add_sample("explore", key, text, dst, dur, st, en, meta={"raw": it}, batch=f"explore:seed{seed}")
                added += 1
                progress(added, n)
            log(f"seed {seed} page {page}: added {added}, duplicates {dup}")
    return {"added": added, "duplicates": dup}
