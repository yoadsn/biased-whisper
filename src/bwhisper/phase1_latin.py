"""Phase 1 on the Latin axis - does the model know a Hebrew spelling for a Latin-script word?

Same paired-decoding idea as the digits Phase 1: free greedy decode; if a Latin-letter token appears,
rewind to the start of its style span (Hebrew prefix + hyphen), ban *all* Latin-letter tokens and continue.
If the model then writes the word in Hebrew letters and keeps the rest of the sentence, it knows both forms.
Whether the Hebrew spelling is a transliteration is screened with a consonant-skeleton similarity
(`latin_script.span_verdict`); it is a heuristic, so every pair is stored for review.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from difflib import SequenceMatcher

import torch

from . import model as M
from .audio import load_audio
from .config import PHASE1_LATIN_DEFAULTS
from .corpus import store
from .jobs import Ctx, register
from .latin_script import LATIN_RE, has_latin, latin_kind, latin_words, span_verdict
from .phase1 import StyleTokens, first_divergence, reconvergence


def latin_token_ids(L: M.Loaded) -> list[int]:
    """Vocabulary ids whose text contains a Latin letter (specials excluded). Cached on the model."""
    if "latin_ids" not in L.extra:
        limit = min(L.vocab, len(L.tokenizer), L.eos)
        L.extra["latin_ids"] = [i for i in range(limit) if LATIN_RE.search(L.decode([i], skip_special=False))]
        mask = torch.zeros(L.vocab, dtype=torch.bool)
        mask[L.extra["latin_ids"]] = True
        L.extra["latin_mask"] = mask.to(L.device)
    return L.extra["latin_ids"]


@torch.inference_mode()
def latin_distribution(L: M.Loaded, enc, prefix: list[int], ban_latin: bool = False, topk: int = 8) -> dict:
    logits, _ = M.decoder_logits(L, enc, torch.tensor([prefix], device=L.device))
    logits = logits.float()
    if L.suppress_ids:
        logits[torch.tensor(L.suppress_ids, device=L.device)] = float("-inf")
    mask = L.extra["latin_mask"]
    if ban_latin:
        logits = logits.masked_fill(mask, float("-inf"))
    lp = torch.log_softmax(logits, -1)
    top = torch.topk(lp, topk)
    return {
        "p_latin": float(lp[mask].exp().sum()),
        "top": [{"id": int(i), "tok": L.decode([int(i)], skip_special=False), "p": float(v.exp()),
                 "latin": bool(mask[int(i)])} for v, i in zip(top.values, top.indices)],
    }


def analyse_sample(L: M.Loaded, audio, max_new: int, style: StyleTokens, threshold: float) -> dict:
    ids = latin_token_ids(L)
    enc = M.encode(L, audio)
    base = M.greedy(L, enc, list(L.sot_seq), max_new=max_new)
    text = L.decode(base.tokens).strip()
    out: dict = {"baseline_text": text, "baseline_tokens": base.tokens, "baseline_ended": base.ended,
                 "baseline_latin_words": latin_words(text), "has_latin": False}
    l_pos = next((i for i, t in enumerate(base.tokens) if bool(L.extra["latin_mask"][t])), None)
    if l_pos is None:
        out["verdict"] = "no_latin"
        return out
    out["has_latin"] = True
    rw = style.span_start(base.tokens, l_pos, L.decode)
    extra = sorted(set(ids) | style.d6)
    sb = style.step_ban(len(L.sot_seq) + rw, base.tokens[rw] if rw < l_pos else None)
    sup = M.greedy(L, enc, list(L.sot_seq) + base.tokens[:rw], max_new=max_new, extra_ban=extra, step_ban=sb)
    y_l = base.tokens
    y_h = base.tokens[:rw] + sup.tokens
    dec = first_divergence(y_l, y_h)
    if dec >= len(y_l) or dec >= len(y_h):
        out["verdict"] = "no_divergence"
        return out
    end_l, end_h = reconvergence(y_l, y_h, dec)
    sup_text = L.decode(y_h).strip()
    latin_span, hebrew_span = L.decode(y_l[dec:end_l]), L.decode(y_h[dec:end_h])
    verdict, score = span_verdict(latin_span, hebrew_span, threshold)
    first_word = (latin_words(latin_span) or [""])[0]
    wa, wb = text.split(), sup_text.split()
    out.update({
        "decision_index": dec, "first_latin_index": l_pos, "rewind_index": rw,
        "latin_span": latin_span, "hebrew_span": hebrew_span, "latin_kind": latin_kind(first_word) if first_word else None,
        "translit_score": round(score, 3), "verdict": verdict,
        "rest_match": round(SequenceMatcher(a=wa, b=wb, autojunk=False).ratio(), 3),
        "suppressed_text": sup_text, "suppressed_tokens": sup.tokens, "suppressed_ended": sup.ended,
        "prefix_ids": list(L.sot_seq) + y_l[:dec], "latin_token": {"id": y_l[dec], "tok": L.decode([y_l[dec]], skip_special=False)},
        "hebrew_token": {"id": y_h[dec], "tok": L.decode([y_h[dec]], skip_special=False)},
        "dist_free_at_decision": latin_distribution(L, enc, list(L.sot_seq) + y_l[:dec]),
        "dist_free_at_first_latin": latin_distribution(L, enc, list(L.sot_seq) + y_l[:l_pos]),
        "dist_banned": latin_distribution(L, enc, list(L.sot_seq) + y_h[:dec], ban_latin=True),
    })
    return out


def select_samples(cfg: dict) -> list[dict]:
    if cfg.get("sample_ids") is not None:
        return store.query(ids=[int(i) for i in cfg["sample_ids"]], limit=10**6)
    pat = re.compile(cfg["ref_regex"])
    rows = [s for s in store.query(source=cfg.get("source"), limit=10**6) if pat.search(s["text"])]
    if cfg.get("order") == "random":
        random.Random(int(cfg["seed"])).shuffle(rows)
    return rows[: int(cfg["limit"])]


@register("phase1_latin", kind="gpu", defaults=PHASE1_LATIN_DEFAULTS)
def run_phase1_latin(ctx: Ctx, cfg: dict) -> dict:
    samples = select_samples(cfg)
    L = M.load_model(cfg["model"], device=cfg.get("device"))
    ids = latin_token_ids(L)
    style = StyleTokens(L, None)
    ctx.log(f"model={L.name} device={L.device} latin_tokens={len(ids)} samples={len(samples)}")
    verdicts: Counter = Counter()
    kinds: Counter = Counter()
    ref_latin = base_latin = 0
    p_latin = []
    for i, s in enumerate(samples):
        ctx.check()
        ctx.progress(i, len(samples), f"sample {s['id']}")
        try:
            r = analyse_sample(L, load_audio(s["audio_path"]), int(cfg["max_new_tokens"]), style, float(cfg["translit_threshold"]))
        except Exception as e:  # noqa: BLE001
            ctx.log(f"sample {s['id']} error: {e}")
            r = {"verdict": "error", "error": str(e)}
        r.update({"sample_id": s["id"], "ref_text": s["text"], "ref_latin_words": latin_words(s["text"]),
                  "duration": s["duration"], "model": L.name})
        ctx.append_jsonl("results.jsonl", r)
        verdicts[r["verdict"]] += 1
        ref_latin += has_latin(s["text"])
        base_latin += bool(r.get("has_latin"))
        if r.get("latin_kind"):
            kinds[f"{r['latin_kind']}:{r['verdict']}"] += 1
        if r.get("dist_free_at_first_latin"):
            p_latin.append(r["dist_free_at_first_latin"]["p_latin"])
        ctx.log(f"#{s['id']} {r['verdict']} {r.get('latin_span', '')!r} -> {r.get('hebrew_span', '')!r} ({r.get('translit_score', '')})")
    ctx.progress(len(samples), len(samples))
    return {"samples": len(samples), "model": L.name, "verdicts": dict(verdicts), "by_kind": dict(kinds),
            "ref_has_latin": ref_latin, "baseline_has_latin": base_latin,
            "mean_p_latin_at_first_latin": sum(p_latin) / len(p_latin) if p_latin else None}
