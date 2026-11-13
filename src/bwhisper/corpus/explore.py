"""Corpus source: explore.ivrit.ai (Google-login protected).

Supports two fast episode-grouped audio extraction modes:
  - mode="range" (default): HTTP Range requests + Ogg Opus page parser + local mini-Ogg decode
  - mode="episode": full episode streaming GET + local Ogg Opus slice/decode
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import shlex
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, quote, urlparse, urlunparse

import httpx

from ..audio import (
    crop_to_wav,
    crop_url_to_wav,
    decode_mini_ogg_to_wav,
    ogg_page_time_bounds,
    parse_ogg_pages,
    parse_opus_header,
    select_ogg_pages_for_range,
)
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


# Search regexes per experiment axis. "latin" asks for a Hebrew word next to a Latin word, or a prefixed Latin
# word (ה-Facebook), so clips are Hebrew sentences with an English word inside rather than all-English audio.
QUERY_PRESETS = {
    "digits": r"\d",
    "latin": r"[א-ת]\S* \S*[A-Za-z]{3,}\S* [א-ת]|[א-ת]-[A-Za-z]{3,}",
}


def search_query(content: str, override: str = "") -> str | None:
    """Regex for one fetch: explicit override, else the preset for `content`, else None = saved config query."""
    if override.strip():
        return override.strip()
    if content == "saved":
        return None
    if content not in QUERY_PRESETS:
        raise ValueError(f"content must be one of {sorted(QUERY_PRESETS) + ['saved']}, got {content!r}")
    return QUERY_PRESETS[content]


def template_search_url(url: str) -> str:
    """Make the q= parameter a {q} placeholder; a copied cURL carries whatever the user last searched for."""
    return re.sub(r"([?&])q=(?!\{q\})[^&]*", r"\1q={q}", url) if url else url


class ServerStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str, fatal: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.fatal = fatal


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
    c["headers"] = {
        k: ("<hidden>" if k.lower() in ("authorization", "x-csrf-token", "cookie") else v)
        for k, v in cfg["headers"].items()
    }
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
        if t == "--url" and i + 1 < len(toks):
            url = toks[i + 1]
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
            if k in ("q", "query") or v == r"\d" or v == "\\d":
                qs[k] = "{q}"
            elif k in ("mode",):
                qs[k] = "{mode}"
            elif k == "seed":
                qs[k] = "{seed}"
            elif k in ("page",):
                qs[k] = "{page1}"
            elif k in ("offset",):
                qs[k] = "{offset}"
            elif k in ("limit", "size", "page_size", "per_page"):
                qs[k] = "{limit}"
        patch["search_url"] = urlunparse(
            (u.scheme, u.netloc, u.path, "", "&".join(f"{k}={v}" for k, v in qs.items()), "")
        )
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
            for m in re.findall(
                r"""["'`](/(?:api|search|audio|segment|stream|media)[^"'`\s]{0,120})["'`]""",
                js,
            ):
                found["api_paths"].add(m)
        for m in re.findall(r"""["'](/(?:api)[^"'\s]{0,120})["']""", html):
            found["api_paths"].add(m)
    return {"ok": True, "scripts": sorted(found["scripts"]), "api_paths": sorted(found["api_paths"])}


def dedupe_episode_hits(
    episode: dict[str, Any],
    min_dur: float = 1.5,
    max_dur: float = 28.0,
    seen_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Deduplicate hits in an episode by (uuid, segment_idx) and filter by duration [min_dur, max_dur]
    and presence of Hebrew characters."""
    uuid = str(episode.get("uuid") or "").strip()
    if not uuid:
        return []
    seen_local: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in episode.get("hits") or []:
        seg_idx = h.get("segment_idx")
        if seg_idx is None:
            continue
        key = f"{uuid}:{seg_idx}"
        if key in seen_local or (seen_keys is not None and key in seen_keys):
            continue
        seen_local.add(key)
        try:
            start = float(h.get("start_sec", 0.0))
            end = float(h.get("end_sec", 0.0))
        except (TypeError, ValueError):
            continue
        dur = end - start
        text = (h.get("text") or "").strip()
        if not text or dur < min_dur or dur > max_dur:
            continue
        if not any("\u0590" <= ch <= "\u05ff" for ch in text):
            continue
        out.append(
            {
                "key": key,
                "uuid": uuid,
                "segment_idx": int(seg_idx),
                "start_sec": start,
                "end_sec": end,
                "dur": dur,
                "text": text,
                "title": episode.get("title") or "",
                "source": episode.get("source") or "",
                "date": episode.get("date") or "",
                "episode_idx": episode.get("episode_idx"),
            }
        )
    return out


def order_non_adjacent(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order candidate segments so mutually non-adjacent segments (|i - j| > 1) come first,
    followed by any remaining adjacent segments as fallback."""
    primary: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for s in segments:
        if all(abs(s["segment_idx"] - p["segment_idx"]) > 1 for p in primary):
            primary.append(s)
        else:
            fallback.append(s)
    return primary + fallback


def select_episode_segments(
    episode: dict[str, Any],
    per_episode: int = 6,
    min_dur: float = 1.5,
    max_dur: float = 28.0,
    seen_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `per_episode` deduplicated, duration-filtered segments from `episode`,
    preferring non-adjacent segment indices."""
    if per_episode <= 0:
        return []
    cands = dedupe_episode_hits(episode, min_dur=min_dur, max_dur=max_dur, seen_keys=seen_keys)
    ordered = order_non_adjacent(cands)
    chosen = ordered[:per_episode]
    if seen_keys is not None:
        for s in chosen:
            seen_keys.add(s["key"])
    return chosen


def select_page_segments(
    page_data: dict[str, Any],
    per_episode: int = 6,
    min_dur: float = 1.5,
    max_dur: float = 28.0,
    seen_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select up to `per_episode` segments per episode across a search response page."""
    seen = seen_keys if seen_keys is not None else set()
    out: list[dict[str, Any]] = []
    for ep in page_data.get("episodes") or []:
        out.extend(
            select_episode_segments(
                ep,
                per_episode=per_episode,
                min_dur=min_dur,
                max_dur=max_dur,
                seen_keys=seen,
            )
        )
    return out


def _check_http_status(r: httpx.Response, context: str = "audio") -> None:
    if _is_login_redirect(r) or r.status_code in (401, 403):
        raise ServerStatusError(r.status_code, f"{context} authentication rejected (HTTP {r.status_code})", fatal=True)
    if r.status_code == 429 or r.status_code >= 500:
        raise ServerStatusError(r.status_code, f"{context} server HTTP {r.status_code}", fatal=False)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"{context} unexpected HTTP {r.status_code}")


def _parse_total_size(r: httpx.Response) -> int:
    cr = r.headers.get("content-range") or r.headers.get("Content-Range") or ""
    if "/" in cr:
        tail = cr.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    cl = r.headers.get("content-length") or r.headers.get("Content-Length") or ""
    if cl.isdigit():
        return int(cl)
    return len(r.content)


def crop_episode_segments_range(
    c: httpx.Client,
    aurl: str,
    duration_sec: float | None,
    segments: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], float]], int]:
    """Fetch only the needed Ogg Opus byte windows for `segments` of one episode using
    HTTP Range requests, decode each locally via a mini-Ogg stream, and return
    ([(seg, wav_dur), ...], bytes_transferred)."""
    if not segments:
        return [], 0

    bytes_transferred = 0
    # 1. Fetch first 16 KB for OpusHead + OpusTags headers and total file size from Content-Range
    r0 = c.get(aurl, headers={"Range": "bytes=0-16383"})
    _check_http_status(r0, "range header")
    bytes_transferred += len(r0.content)
    file_size = _parse_total_size(r0)
    try:
        header_bytes, pre_skip, serial = parse_opus_header(r0.content)
    except ValueError:
        # Some episodes have a larger OpusTags page (e.g. embedded cover art); fetch 128 KB
        r0 = c.get(aurl, headers={"Range": "bytes=0-131071"})
        _check_http_status(r0, "range header 128k")
        bytes_transferred += len(r0.content)
        header_bytes, pre_skip, serial = parse_opus_header(r0.content)

    # If the server ignored Range and returned the full file (status 200), decode directly from r0.content!
    if r0.status_code == 200 and len(r0.content) >= file_size and file_size > 16384:
        all_pages = parse_ogg_pages(r0.content, expected_serial=serial)
        results: list[tuple[dict[str, Any], float]] = []
        for s in segments:
            sel = select_ogg_pages_for_range(
                all_pages, pre_skip, s["c_start"], s["c_end"], is_file_start=True, is_file_end=True
            )
            if sel is not None:
                dur = decode_mini_ogg_to_wav(header_bytes, sel[0], sel[1], s["c_start"], s["c_end"], s["dst"])
                results.append((s, dur))
        return results, bytes_transferred

    est_dur = float(duration_sec) if (duration_sec and duration_sec > 1.0) else max(s["c_end"] for s in segments) * 1.2
    bps = file_size / max(1.0, est_dur)
    bps = max(1500.0, min(16000.0, bps))

    # 2. Sort segments by time and merge segments within ~600 KB into shared windows (up to ~1 MB span)
    sorted_segs = sorted(segments, key=lambda x: x["c_start"])
    clusters: list[list[dict[str, Any]]] = []
    for s in sorted_segs:
        if not clusters:
            clusters.append([s])
            continue
        prev_cluster = clusters[-1]
        gap_bytes = (s["c_start"] - prev_cluster[-1]["c_end"]) * bps
        span_bytes = (s["c_end"] - prev_cluster[0]["c_start"]) * bps
        if gap_bytes <= 500_000 and span_bytes <= 1_000_000:
            prev_cluster.append(s)
        else:
            clusters.append([s])

    results = []
    for cluster in clusters:
        t_min = min(s["c_start"] for s in cluster)
        t_max = max(s["c_end"] for s in cluster)
        need_start = max(0.0, t_min - 0.15)
        est_start = int(need_start * bps)
        est_end = int(t_max * bps)

        win_start = max(0, est_start - 150_000)
        win_end = min(file_size - 1, max(est_end + 150_000, win_start + 350_000))
        pages = []

        for _it in range(3):
            if win_start == 0 and win_end < len(r0.content):
                buf = r0.content
            else:
                rw = c.get(aurl, headers={"Range": f"bytes={win_start}-{win_end}"})
                _check_http_status(rw, "range window")
                bytes_transferred += len(rw.content)
                buf = rw.content

            pages = parse_ogg_pages(buf, expected_serial=serial)
            bounds = ogg_page_time_bounds(pages, pre_skip)
            if bounds is None:
                win_start = max(0, win_start - 250_000)
                win_end = min(file_size - 1, win_end + 250_000)
                continue

            t_first, t_last = bounds
            start_ok = (win_start == 0) or (t_first <= need_start)
            end_ok = (win_end >= file_size - 1) or (t_last >= t_max)
            if start_ok and end_ok:
                break

            obs_bps = (
                (win_end - win_start) / max(1.0, t_last - t_first)
                if (t_last - t_first) > 2.0
                else bps
            )
            obs_bps = max(1500.0, min(16000.0, obs_bps))
            # Estimate byte offset of need_start using observed (win_start, t_first)
            target_byte = int(win_start + (need_start - t_first) * obs_bps)
            new_start = max(0, target_byte - 150_000)
            span_bytes = int((t_max - need_start) * obs_bps) + 320_000
            win_start = min(new_start, max(0, file_size - 350_000))
            win_end = min(file_size - 1, max(win_start + span_bytes, win_start + 350_000))

        is_fstart = win_start == 0
        is_fend = win_end >= file_size - 1
        for s in cluster:
            sel = select_ogg_pages_for_range(
                pages,
                pre_skip,
                s["c_start"],
                s["c_end"],
                is_file_start=is_fstart,
                is_file_end=is_fend,
            )
            if sel is None:
                continue
            dur = decode_mini_ogg_to_wav(
                header_bytes, sel[0], sel[1], s["c_start"], s["c_end"], s["dst"]
            )
            results.append((s, dur))

    return results, bytes_transferred


def crop_episode_segments_full(
    c: httpx.Client,
    aurl: str,
    segments: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], float]], int]:
    """Fallback mode='episode': download the whole episode once via a streaming GET,
    then extract all selected segments locally."""
    if not segments:
        return [], 0

    chunks: list[bytes] = []
    with c.stream("GET", aurl) as r:
        _check_http_status(r, "episode download")
        for chunk in r.iter_bytes(chunk_size=262144):
            chunks.append(chunk)
    buf = b"".join(chunks)
    bytes_transferred = len(buf)

    results: list[tuple[dict[str, Any], float]] = []
    try:
        header_bytes, pre_skip, serial = parse_opus_header(buf)
        pages = parse_ogg_pages(buf, expected_serial=serial)
        for s in segments:
            sel = select_ogg_pages_for_range(
                pages, pre_skip, s["c_start"], s["c_end"], is_file_start=True, is_file_end=True
            )
            if sel is not None:
                dur = decode_mini_ogg_to_wav(
                    header_bytes, sel[0], sel[1], s["c_start"], s["c_end"], s["dst"]
                )
                results.append((s, dur))
        if len(results) == len(segments):
            return results, bytes_transferred
    except Exception:
        results.clear()

    # Fallback via temporary file in system temp dir
    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tf:
        tf.write(buf)
        tmp_path = Path(tf.name)
    try:
        for s in segments:
            dur = crop_to_wav(tmp_path, s["dst"], s["c_start"], s["c_end"])
            results.append((s, dur))
    finally:
        tmp_path.unlink(missing_ok=True)
    return results, bytes_transferred


def fetch_episodes_batch(
    seed: int,
    n: int,
    per_episode: int = 6,
    pad: float = 0.25,
    min_dur: float = 1.5,
    max_dur: float = 28.0,
    mode: str = "range",
    workers: int = 4,
    log: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda a, b: None,
    max_pages: int = 40,
    episode_counts: dict[str, int] | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Page through explore.ivrit.ai search results for `seed` (search regex `query`, default the saved one), grouping by episode and
    running up to `workers` parallel episode workers (each with a persistent keep-alive
    httpx.Client) using either `mode='range'` or `mode='episode'`."""
    cfg = load_cfg()
    if query:
        cfg["query"] = query
    base = load_settings()["explore_base_url"].rstrip("/")
    search_tpl = template_search_url(
        cfg.get("search_url")
        or f"{base}/api/search?q={{q}}&mode={{mode}}&page={{page1}}&per_page={{limit}}&shuffle=true&seed={{seed}}"
    )
    ps = int(cfg.get("page_size") or 40)
    workers = max(1, min(8, int(workers)))
    episode_counts = episode_counts if episode_counts is not None else {}

    added = 0
    dup = 0
    episodes_seen = 0
    bytes_transferred = 0
    seen_keys: set[str] = set()
    t_start_all = time.time()

    err_lock = threading.Lock()
    consecutive_errors = 0
    abort_error: ServerStatusError | None = None

    thread_local = threading.local()
    worker_clients: list[httpx.Client] = []
    clients_lock = threading.Lock()

    def get_worker_client() -> httpx.Client:
        cl = getattr(thread_local, "client", None)
        if cl is None:
            cl = client(cfg)
            thread_local.client = cl
            with clients_lock:
                worker_clients.append(cl)
        return cl

    def process_episode(ep: dict[str, Any], ep_segs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[tuple[dict[str, Any], float]], int]:
        nonlocal consecutive_errors, abort_error
        with err_lock:
            if abort_error is not None:
                raise abort_error
        if should_stop():
            return ep, [], 0

        uuid = ep["uuid"]
        aurl = f"{base}/audio/{uuid}.opus"
        ep_dur = ep.get("duration_sec")
        w_client = get_worker_client()

        for attempt in (1, 2):
            try:
                if mode == "episode":
                    cropped, b_read = crop_episode_segments_full(w_client, aurl, ep_segs)
                else:
                    cropped, b_read = crop_episode_segments_range(w_client, aurl, ep_dur, ep_segs)
                with err_lock:
                    consecutive_errors = 0
                return ep, cropped, b_read
            except ServerStatusError as se:
                with err_lock:
                    consecutive_errors += 1
                    cnt = consecutive_errors
                    if se.fatal or cnt >= 3:
                        abort_error = se
                        raise
                time.sleep(1.5 ** cnt)
            except Exception as e:  # noqa: BLE001
                for s in ep_segs:
                    Path(s["dst"]).unlink(missing_ok=True)
                if attempt == 2:
                    log(f"seed {seed}: skipping episode {uuid} ({type(e).__name__}: {str(e)[:120]})")
                    return ep, [], 0
                time.sleep(0.3)
        return ep, [], 0

    try:
        with client(cfg) as search_client, ThreadPoolExecutor(max_workers=workers, thread_name_prefix="explore") as pool:
            for page in range(max_pages):
                if added >= n or should_stop() or abort_error is not None:
                    break
                page1 = page + 1
                url = search_tpl.format(
                    q=quote(cfg["query"]),
                    mode=cfg["mode"],
                    seed=seed,
                    page=page,
                    page1=page1,
                    offset=page * ps,
                    limit=ps,
                )
                if url.startswith("/"):
                    url = base + url
                r = search_client.get(url, headers={"Accept": "application/json"})
                _check_http_status(r, f"search seed {seed} page {page1}")
                try:
                    data = r.json()
                except Exception:
                    raise RuntimeError(f"search did not return JSON on seed {seed} page {page1}") from None

                episodes = data.get("episodes") or []
                if not isinstance(episodes, list) or not episodes:
                    log(f"seed {seed} page {page1}: no episodes returned")
                    break

                # Prepare episode batches needed to reach `n - added`
                ep_tasks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
                queued_segs = 0
                queued_by_episode: dict[str, int] = {}
                need = n - added

                for ep in episodes:
                    if queued_segs >= need:
                        break
                    episodes_seen += 1
                    uuid = ep.get("uuid", "")
                    remaining = per_episode - episode_counts.get(uuid, 0) - queued_by_episode.get(uuid, 0)
                    if remaining <= 0:
                        continue
                    all_cands = dedupe_episode_hits(ep, min_dur=min_dur, max_dur=max_dur, seen_keys=seen_keys)
                    fresh_cands: list[dict[str, Any]] = []
                    for cand in all_cands:
                        seen_keys.add(cand["key"])
                        if store.exists("explore", cand["key"]):
                            dup += 1
                        else:
                            fresh_cands.append(cand)

                    ordered = order_non_adjacent(fresh_cands)
                    chosen: list[dict[str, Any]] = []
                    selected_indices: list[int] = []
                    for cand in ordered:
                        if len(chosen) >= remaining or (queued_segs + len(chosen)) >= need:
                            break
                        if selected_indices and any(abs(cand["segment_idx"] - idx) <= 1 for idx in selected_indices):
                            rem_non_adj = [
                                x for x in ordered
                                if x["segment_idx"] not in selected_indices
                                and all(abs(x["segment_idx"] - idx) > 1 for idx in selected_indices)
                            ]
                            if rem_non_adj:
                                continue
                        c_start = max(0.0, cand["start_sec"] - pad)
                        c_end = cand["end_sec"] + pad
                        ep_dur = ep.get("duration_sec")
                        if isinstance(ep_dur, (int, float)) and ep_dur > c_start:
                            c_end = min(c_end, float(ep_dur))
                        cand = dict(cand)
                        cand["c_start"] = c_start
                        cand["c_end"] = c_end
                        cand["dst"] = store.audio_path_for("explore", cand["key"])
                        chosen.append(cand)
                        selected_indices.append(cand["segment_idx"])

                    if chosen:
                        ep_tasks.append((ep, chosen))
                        queued_segs += len(chosen)
                        queued_by_episode[uuid] = queued_by_episode.get(uuid, 0) + len(chosen)

                futures = [pool.submit(process_episode, ep, segs) for ep, segs in ep_tasks]
                for fut in as_completed(futures):
                    ep, cropped, b_read = fut.result()
                    bytes_transferred += b_read
                    for cand, dur in cropped:
                        if added >= n:
                            Path(cand["dst"]).unlink(missing_ok=True)
                            continue
                        meta = {
                            "uuid": cand["uuid"],
                            "title": cand["title"],
                            "source": cand["source"],
                            "date": cand["date"],
                            "segment_idx": cand["segment_idx"],
                            "episode_idx": cand["episode_idx"],
                            "seed": seed,
                            "query": cfg["query"],
                        }
                        ins = store.add_sample(
                            "explore",
                            cand["key"],
                            cand["text"],
                            cand["dst"],
                            dur,
                            cand["start_sec"],
                            cand["end_sec"],
                            meta=meta,
                            batch=f"explore:seed{seed}",
                        )
                        if ins is not None:
                            added += 1
                            uuid = cand["uuid"]
                            episode_counts[uuid] = episode_counts.get(uuid, 0) + 1
                            progress(added, n)
                        else:
                            dup += 1

                elapsed = max(1e-3, time.time() - t_start_all)
                log(
                    f"seed {seed} page {page1} ({mode}): added={added}/{n} dup={dup} "
                    f"episodes={episodes_seen} MB={bytes_transferred / 1e6:.2f} elapsed={elapsed:.1f}s"
                )
                if not data.get("has_more"):
                    break
    finally:
        for cl in worker_clients:
            try:
                cl.close()
            except Exception:
                pass

    elapsed_total = time.time() - t_start_all
    return {
        "added": added,
        "duplicates_skipped": dup,
        "episodes_seen": episodes_seen,
        "bytes_transferred": bytes_transferred,
        "elapsed_sec": round(elapsed_total, 2),
        "seed": seed,
        "mode": mode,
    }


# Keep fetch_batch as an alias so any legacy caller continues to work
def fetch_batch(
    seed: int,
    n: int,
    log: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda a, b: None,
    max_pages: int = 40,
) -> dict[str, Any]:
    return fetch_episodes_batch(
        seed=seed, n=n, log=log, should_stop=should_stop, progress=progress, max_pages=max_pages
    )
