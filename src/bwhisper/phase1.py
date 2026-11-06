"""Phase 1 - prove the model knows both forms.

For each corpus utterance: free greedy decode; if a digit token appears,
rewind to just before the first digit token, ban *all* digit-related tokens
and continue. If the model then spells the number out and keeps the rest of
the sentence intact, it demonstrably "knows both forms" at that decision point.

Each result row doubles as a Phase-2 *decision point* (prefix ids + the digit
token + the cardinal first token).
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from . import model as M
from .audio import load_audio
from .config import PHASE1_DEFAULTS, RUNS_DIR
from .corpus import store
from .hebrew_numbers import auto_verdict, count_mentions
from .jobs import Ctx, read_jsonl, register


_PREFIX_RE = re.compile(r"\s?[והבכלמש]{1,3}")


def is_hyphen_text(t: str) -> bool:
    t = t.strip()
    return bool(t) and (t in ("-", "־") or t.endswith(("-", "־")))


def is_prefix_text(t: str) -> bool:
    return bool(_PREFIX_RE.fullmatch(t))


_FUSE_PROBES = ("עשרים", "שלוש", "אלפיים", "מאה", "חמש", "שמונים")


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode (the byte-level BPE alphabet Whisper uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DEC = _byte_decoder()


def token_bytes(tokenizer, i: int) -> bytes:
    tok = tokenizer.convert_ids_to_tokens(i)
    try:
        return bytes(_BYTE_DEC[c] for c in tok)
    except (KeyError, TypeError):
        return b""


class StyleTokens:
    """Token sets for span-level decisions.

    hyphen   tokens that are a hyphen ('-'), plus raw-byte tokens of 0xD6 (maqaf '־' and niqqud are
             byte-split by the BPE and never appear in normal output); banned after a prefix token
    prefix   one-to-three-letter prefix tokens (' ב', ' ו', ' מה', ...)
    merging  prefix tokens whose fused number words start with a *different* token (' ב' -> ' בא',
             ' בש'); at the span start these are banned so the word path can fuse the prefix
    symbols  digit-style symbols ('%' and satellite symbols), banned on the word path
    """

    def __init__(self, L: M.Loaded, satellites: dict | None = None):
        tk = L.tokenizer
        self.hyphen, self.prefix, self.symbols, self.d6 = set(), set(), set(), set()
        for i in range(min(L.vocab, len(tk))):
            t = L.decode([i], skip_special=False)
            if is_hyphen_text(t) and not any(c.isdigit() for c in t):
                self.hyphen.add(i)
            elif is_prefix_text(t):
                self.prefix.add(i)
            if "%" in t and not any("\u0590" <= c <= "\u05ff" for c in t):
                self.symbols.add(i)
            b = token_bytes(tk, i)
            if b and 0xD6 in b and i < L.eos:
                self.d6.add(i)
        self.hyphen |= self.d6
        self.merging = set()
        for p in self.prefix:
            pt = L.decode([p], skip_special=False)
            firsts = [tk.encode(pt + w, add_special_tokens=False)[0] for w in _FUSE_PROBES]
            if sum(f != p for f in firsts) > len(firsts) / 2:
                self.merging.add(p)
        self.sat_before = set()
        for e in (satellites or {}).get("before", []):
            self.sat_before.add(int(e["id"]))
        for e in (satellites or {}).get("after", []):
            t = e.get("tok", "")
            if t.strip() and not any(c.isalpha() or c.isdigit() for c in t):  # symbols only, never words
                self.symbols.add(int(e["id"]))

    def span_start(self, tokens: list[int], d_pos: int, decode) -> int:
        s = d_pos
        if s > 0 and (tokens[s - 1] in self.hyphen or tokens[s - 1] in self.sat_before):
            s -= 1
            if s > 0 and (tokens[s - 1] in self.prefix or tokens[s - 1] in self.sat_before) and tokens[s] in self.hyphen:
                s -= 1
        return s

    def step_ban(self, prefix_len: int | None = None, span_tok: int | None = None):
        hy = sorted(self.hyphen)
        first_ban = [span_tok] if span_tok is not None and span_tok in self.merging else []

        def f(history: list[int]) -> list[int]:
            out = list(first_ban) if prefix_len is not None and len(history) == prefix_len else []
            if history and history[-1] in self.prefix:
                out += hy
            return out

        return f


def first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def reconvergence(a: list[int], b: list[int], start: int, min_equal: int = 3) -> tuple[int, int]:
    """End (exclusive) of the divergent region starting at ``start`` in a and b."""
    sm = SequenceMatcher(a=a[start:], b=b[start:], autojunk=False)
    for blk in sm.get_matching_blocks():
        if blk.size >= min_equal:
            return start + blk.a, start + blk.b
    return len(a), len(b)


def latest_satellites() -> dict | None:
    runs = sorted(RUNS_DIR.glob("phase1_5-*/satellites.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def analyse_sample(L: M.Loaded, audio, max_new: int, style: StyleTokens | None = None) -> dict:
    """style=None -> v1 (rewind to first digit token). Otherwise v2: rewind to the start of the style
    span (prefix + hyphen), ban digits, hyphen-after-prefix and digit-style symbols."""
    enc = M.encode(L, audio)
    base = M.greedy(L, enc, list(L.sot_seq), max_new=max_new)
    text = L.decode(base.tokens).strip()
    out: dict = {
        "baseline_text": text,
        "baseline_tokens": base.tokens,
        "mentions": count_mentions(text),
        "has_digits": False,
        "span_mode": style is not None,
    }
    d_pos = next((i for i, t in enumerate(base.tokens) if bool(L.digit_mask[t])), None)
    if d_pos is None:
        out["verdict"] = "no_digits"
        return out
    out["has_digits"] = True
    if style is None:
        rw, extra, sb = d_pos, None, None
    else:
        rw = style.span_start(base.tokens, d_pos, L.decode)
        extra = sorted(style.symbols | style.d6)
        sb = style.step_ban(len(L.sot_seq) + rw, base.tokens[rw] if rw < d_pos else None)
    prefix = list(L.sot_seq) + base.tokens[:rw]
    sup = M.greedy(L, enc, prefix, max_new=max_new, ban_digits=True, extra_ban=extra, step_ban=sb)
    y_d = base.tokens
    y_c = base.tokens[:rw] + sup.tokens
    dec = first_divergence(y_d, y_c)
    if dec >= len(y_d) or dec >= len(y_c):
        out["verdict"] = "no_divergence"
        return out
    end_d, end_c = reconvergence(y_d, y_c, dec)
    dec_prefix = list(L.sot_seq) + y_d[:dec]
    dist_free = M.step_distribution(L, enc, dec_prefix)
    dist_ban = M.step_distribution(L, enc, list(L.sot_seq) + y_c[:dec], ban_digits=True)
    sup_text = L.decode(y_c).strip()
    verdict, diff = auto_verdict(text, sup_text)
    if style is not None and verdict == "cardinal" and any(t in style.symbols for t in y_c):
        verdict = "partial"
    out.update(
        {
            "decision_index": dec,
            "first_digit_index": d_pos,
            "rewind_index": rw,
            "span_offset": d_pos - dec,
            "prefix_ids": dec_prefix,
            "prefix_text": L.decode(y_d[:dec]),
            "hyphen_committed": bool(d_pos > 0 and is_hyphen_text(L.decode([y_d[d_pos - 1]], skip_special=False))),
            "digit_token": {"id": y_d[dec], "tok": L.decode([y_d[dec]], skip_special=False)},
            "cardinal_token": {"id": y_c[dec], "tok": L.decode([y_c[dec]], skip_special=False)},
            "y_digit": y_d,
            "y_cardinal": y_c,
            "div_digit": [dec, end_d],
            "div_cardinal": [dec, end_c],
            "div_digit_text": L.decode(y_d[dec:end_d]),
            "div_cardinal_text": L.decode(y_c[dec:end_c]),
            "p_digit": dist_free["p_digit"],
            "dist_free": dist_free["top"],
            "dist_suppressed": dist_ban["top"],
            "suppressed_text": sup_text,
            "suppressed_tokens": sup.tokens,
            "suppressed_ended": sup.ended,
            "verdict": verdict,
            "diff": diff,
        }
    )
    return out


@register("phase1", kind="gpu", defaults=PHASE1_DEFAULTS)
def run_phase1(ctx: Ctx, cfg: dict) -> dict:
    samples = select_samples(cfg)
    devices = cfg.get("devices") or []
    if not isinstance(devices, list) or len(set(devices)) != len(devices):
        raise ValueError("devices must be a list of unique CUDA devices")
    if devices:
        import torch

        available = {f"cuda:{i}" for i in range(torch.cuda.device_count())}
        if set(devices) - available:
            raise ValueError(f"requested devices unavailable: {set(devices) - available}")
        if not samples:
            raise ValueError("no samples selected")
        devices = devices[:len(samples)]
        counts = [0] * len(devices)
        lock = threading.Lock()

        class WorkerCtx:
            def __init__(self, index):
                self.index = index

            def __getattr__(self, key):
                return getattr(ctx, key)

            def progress(self, done, total, message=None):
                with lock:
                    counts[self.index] = done
                    ctx.progress(sum(counts), len(samples), message)

        def worker(i):
            return _run_samples(WorkerCtx(i), cfg, samples[i::len(devices)], devices[i])

        ctx.log(f"parallel phase1: {len(devices)} independent replicas on {devices}")
        with ThreadPoolExecutor(len(devices), thread_name_prefix="phase1-gpu") as pool:
            summaries = list(pool.map(worker, range(len(devices))))
        verdicts = Counter()
        for summary in summaries:
            verdicts.update(summary["verdicts"])
        n_digits = sum(s["with_digits"] for s in summaries)
        return {"samples": len(samples), "verdicts": dict(verdicts), "with_digits": n_digits,
                "mean_p_digit_at_decision": sum((s["mean_p_digit_at_decision"] or 0) * s["with_digits"]
                                                for s in summaries) / n_digits if n_digits else None,
                "model": summaries[0]["model"], "devices": devices}
    return _run_samples(ctx, cfg, samples)


def select_samples(cfg: dict) -> list[dict]:
    if cfg.get("sample_ids") is not None:
        ids = [int(i) for i in cfg["sample_ids"]]
        if not ids:
            return []
        samples = store.query(ids=ids, limit=10**6)
        if {s["id"] for s in samples} != set(ids):
            raise ValueError("some requested corpus samples do not exist")
        return samples
    return store.query(digits_only=cfg["require_ref_digits"], limit=int(cfg["limit"]), order="random")


def _run_samples(ctx: Ctx, cfg: dict, samples: list[dict], device: str | None = None) -> dict:
    L = M.load_model(cfg["model"], device=device, extra_digit_chars=cfg.get("extra_digit_chars", ""))
    ctx.log(f"model={L.name} device={L.device} digit_tokens={len(L.digit_ids)}")
    style = None
    if cfg.get("span_mode", True):
        sat = None
        if cfg.get("satellites_run"):
            sat = json.loads((RUNS_DIR / cfg["satellites_run"] / "satellites.json").read_text())
        else:
            sat = latest_satellites()
        style = StyleTokens(L, sat)
        ctx.log(f"span mode: hyphen={len(style.hyphen)} prefix={len(style.prefix)} symbols={len(style.symbols)} "
                f"satellites={'yes' if sat else 'none (built-in rules only)'}")
    ctx.log(f"{len(samples)} samples")
    verdicts: Counter = Counter()
    p_digits = []
    for i, s in enumerate(samples):
        ctx.check()
        ctx.progress(i, len(samples), f"sample {s['id']}")
        try:
            audio = load_audio(s["audio_path"])
            r = analyse_sample(L, audio, int(cfg["max_new_tokens"]), style)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"sample {s['id']} error: {e}")
            r = {"verdict": "error", "error": str(e)}
        r.update({"sample_id": s["id"], "ref_text": s["text"], "duration": s["duration"], "model": L.name})
        ctx.append_jsonl("results.jsonl", r)
        verdicts[r["verdict"]] += 1
        if r.get("has_digits") and r.get("p_digit") is not None:
            p_digits.append(r["p_digit"])
        ctx.log(f"#{s['id']} {r['verdict']}: {r.get('baseline_text', '')[:60]}")
    ctx.progress(len(samples), len(samples))
    return {
        "samples": len(samples),
        "verdicts": dict(verdicts),
        "with_digits": sum(1 for _ in p_digits),
        "mean_p_digit_at_decision": (sum(p_digits) / len(p_digits)) if p_digits else None,
        "model": L.name,
    }


def load_verdicts(run_id: str) -> dict[int, str]:
    p = RUNS_DIR / run_id / "verdicts.json"
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def set_verdict(run_id: str, sample_id: int, verdict: str | None) -> dict:
    v = load_verdicts(run_id)
    if verdict in (None, ""):
        v.pop(sample_id, None)
    else:
        v[sample_id] = verdict
    (RUNS_DIR / run_id / "verdicts.json").write_text(json.dumps(v))
    return v


def decision_points(run_ids: list[str], accept: list[str], limit: int | None = None) -> list[dict]:
    """Phase-1 rows usable as Phase-2 decision points. A human verdict overrides the auto one:
    'human_good' is accepted, 'human_bad' rejected."""
    pts = []
    for rid in run_ids:
        hv = load_verdicts(rid)
        for r in read_jsonl(RUNS_DIR / rid / "results.jsonl"):
            if not r.get("has_digits") or not r.get("cardinal_token"):
                continue
            h = hv.get(r["sample_id"])
            v = "human_good" if h == "good" else ("human_bad" if h == "bad" else r["verdict"])
            if v in accept:
                r["effective_verdict"] = v
                r["run_id"] = rid
                pts.append(r)
    return pts[:limit] if limit else pts
