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
from collections import Counter

from . import model as M
from .audio import load_audio
from .config import PHASE1_DEFAULTS, RUNS_DIR
from .corpus import store
from .hebrew_numbers import auto_verdict, count_mentions
from .jobs import Ctx, read_jsonl, register


def analyse_sample(L: M.Loaded, audio, max_new: int, min_digit_tokens: int = 1) -> dict:
    enc = M.encode(L, audio)
    base = M.greedy(L, enc, list(L.sot_seq), max_new=max_new)
    text = L.decode(base.tokens).strip()
    out: dict = {
        "baseline_text": text,
        "baseline_tokens": base.tokens,
        "mentions": count_mentions(text),
        "has_digits": False,
    }
    d_pos = next((i for i, t in enumerate(base.tokens) if bool(L.digit_mask[t])), None)
    if d_pos is None:
        out["verdict"] = "no_digits"
        return out
    out["has_digits"] = True
    prefix = list(L.sot_seq) + base.tokens[:d_pos]
    dig_tok = base.tokens[d_pos]
    dist_free = M.step_distribution(L, enc, prefix)
    dist_ban = M.step_distribution(L, enc, prefix, ban_digits=True)
    sup = M.greedy(L, enc, prefix, max_new=max_new, ban_digits=True)
    sup_text = L.decode(base.tokens[:d_pos] + sup.tokens).strip()
    verdict, diff = auto_verdict(text, sup_text)
    out.update(
        {
            "decision_index": d_pos,
            "prefix_ids": prefix,
            "prefix_text": L.decode(base.tokens[:d_pos]),
            "hyphen_committed": bool(d_pos > 0 and L.decode([base.tokens[d_pos - 1]], skip_special=False).rstrip().endswith(("-", "־"))),
            "digit_token": {"id": dig_tok, "tok": L.decode([dig_tok], skip_special=False)},
            "cardinal_token": {"id": sup.tokens[0], "tok": L.decode([sup.tokens[0]], skip_special=False)} if sup.tokens else None,
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
    L = M.load_model(cfg["model"], extra_digit_chars=cfg.get("extra_digit_chars", ""))
    ctx.log(f"model={L.name} device={L.device} digit_tokens={len(L.digit_ids)}")
    if cfg.get("sample_ids"):
        samples = store.query(ids=[int(i) for i in cfg["sample_ids"]], limit=10**6)
    else:
        samples = store.query(digits_only=cfg["require_ref_digits"], limit=int(cfg["limit"]), order="random")
    ctx.log(f"{len(samples)} samples")
    verdicts: Counter = Counter()
    p_digits = []
    for i, s in enumerate(samples):
        ctx.check()
        ctx.progress(i, len(samples), f"sample {s['id']}")
        try:
            audio = load_audio(s["audio_path"])
            r = analyse_sample(L, audio, int(cfg["max_new_tokens"]))
        except Exception as e:  # noqa: BLE001
            ctx.log(f"sample {s['id']} error: {e}")
            r = {"verdict": "error", "error": str(e)}
        r.update({"sample_id": s["id"], "ref_text": s["text"], "duration": s["duration"], "model": L.name})
        ctx.append_jsonl("results.jsonl", r)
        verdicts[r["verdict"]] += 1
        if r.get("has_digits"):
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
