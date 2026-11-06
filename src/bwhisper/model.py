"""Whisper loading + a small, explicit greedy decoder.

We do not use ``generate`` because Phase 1 needs to rewind to an arbitrary
step and suppress token sets, and Phase 2 needs exact control of the forward
pass for gradients. The decoder here is validated against ``generate`` in
tests/test_decoder.py.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from .config import SAMPLE_RATE, load_settings


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Loaded:
    name: str
    device: str
    model: "torch.nn.Module"
    tokenizer: object
    feature_extractor: object
    digit_ids: torch.Tensor  # 1-D long, on CPU
    digit_mask: torch.Tensor  # bool [vocab], on device
    suppress_ids: list[int]
    suppress_begin: list[int]
    eos: int
    sot_seq: list[int]  # forced prefix: sot, lang, task, notimestamps
    vocab: int
    extra: dict = field(default_factory=dict)

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special)


_cache: dict[tuple, Loaded] = {}
_lock = threading.RLock()


def digit_token_ids(tokenizer, vocab: int, extra_chars: str = "") -> list[int]:
    ids = []
    for i in range(vocab):
        s = tokenizer.decode([i], skip_special_tokens=False)
        if any(c.isdigit() for c in s) or (extra_chars and any(c in s for c in extra_chars)):
            ids.append(i)
    return ids


def load_model(name: str | None = None, device: str | None = None, extra_digit_chars: str = "") -> Loaded:
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    st = load_settings()
    name = name or st["model_active"]
    device = pick_device(device or st["device"])
    key = (name, device, extra_digit_chars)
    with _lock:
        if key in _cache:
            return _cache[key]
        try:
            model = WhisperForConditionalGeneration.from_pretrained(name, local_files_only=True)
            tok = WhisperTokenizer.from_pretrained(name, local_files_only=True)
            fe = WhisperFeatureExtractor.from_pretrained(name, local_files_only=True)
        except Exception:
            model = WhisperForConditionalGeneration.from_pretrained(name)
            tok = WhisperTokenizer.from_pretrained(name)
            fe = WhisperFeatureExtractor.from_pretrained(name)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        vocab = model.config.vocab_size
        d_ids = digit_token_ids(tok, min(vocab, len(tok)), extra_digit_chars)
        mask = torch.zeros(vocab, dtype=torch.bool)
        mask[d_ids] = True
        gc = model.generation_config
        lang = st["language"]
        sot = [
            tok.convert_tokens_to_ids("<|startoftranscript|>"),
            tok.convert_tokens_to_ids(f"<|{lang}|>"),
            tok.convert_tokens_to_ids("<|transcribe|>"),
        ]
        if not st.get("timestamps"):
            sot.append(tok.convert_tokens_to_ids("<|notimestamps|>"))
        L = Loaded(
            name=name,
            device=device,
            model=model,
            tokenizer=tok,
            feature_extractor=fe,
            digit_ids=torch.tensor(d_ids, dtype=torch.long),
            digit_mask=mask.to(device),
            suppress_ids=list(getattr(gc, "suppress_tokens", None) or []),
            suppress_begin=list(getattr(gc, "begin_suppress_tokens", None) or []),
            eos=model.config.eos_token_id,
            sot_seq=sot,
            vocab=vocab,
        )
        _cache[key] = L
        return L


def unload_all() -> None:
    with _lock:
        _cache.clear()
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def features(L: Loaded, audio: np.ndarray) -> torch.Tensor:
    f = L.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    return f.to(L.device)


def encode(L: Loaded, audio: np.ndarray, grad: bool = False) -> torch.Tensor:
    with torch.set_grad_enabled(grad):
        return L.model.model.encoder(features(L, audio)).last_hidden_state


def decoder_logits(L: Loaded, enc: torch.Tensor, ids: torch.Tensor, past=None):
    """Run the decoder on ``ids`` [1, T]; returns last-position logits [vocab] and cache."""
    out = L.model.model.decoder(
        input_ids=ids, encoder_hidden_states=enc, past_key_values=past, use_cache=True
    )
    logits = L.model.proj_out(out.last_hidden_state[:, -1])[0]
    return logits, out.past_key_values


def prefix_logits(L: Loaded, enc: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Full (cache-free) forward over a prefix; differentiable. Returns [vocab] logits."""
    t = torch.tensor([ids], device=L.device)
    out = L.model.model.decoder(input_ids=t, encoder_hidden_states=enc, use_cache=False)
    return L.model.proj_out(out.last_hidden_state[:, -1])[0]


@dataclass
class Step:
    token: int
    logprob: float
    p_digit: float


@dataclass
class DecodeResult:
    tokens: list[int]  # generated tokens only (no forced prefix), EOS excluded
    steps: list[Step]
    ended: bool


@torch.inference_mode()
def greedy(
    L: Loaded,
    enc: torch.Tensor,
    prefix: list[int],
    max_new: int = 96,
    ban_digits: bool = False,
    extra_ban: list[int] | None = None,
    apply_whisper_suppress: bool = True,
    step_ban=None,
) -> DecodeResult:
    """``step_ban(history) -> list[int]`` may ban extra tokens at a step given all tokens so far
    (prefix + generated), e.g. a hyphen right after a one-letter prefix."""
    """Greedy decode continuing from ``prefix`` (which includes the forced SOT tokens)."""
    ban = torch.zeros(L.vocab, dtype=torch.bool, device=L.device)
    if apply_whisper_suppress and L.suppress_ids:
        ban[torch.tensor(L.suppress_ids, device=L.device)] = True
    if extra_ban:
        ban[torch.tensor(extra_ban, device=L.device)] = True
    if ban_digits:
        ban |= L.digit_mask
    begin = torch.zeros_like(ban)
    if L.suppress_begin:
        begin[torch.tensor(L.suppress_begin, device=L.device)] = True
    # Forced-prefix specials (sot/lang/task/notimestamps) must never be generated
    specials = torch.tensor([i for i in L.sot_seq], device=L.device)
    ban[specials] = True
    # Other special tokens (timestamps, etc.) above eos are never wanted in notimestamps mode
    ban[L.eos + 1 :] = True

    toks: list[int] = []
    steps: list[Step] = []
    ids = torch.tensor([prefix], device=L.device)
    past = None
    ended = False
    n_prefix_gen = len(prefix) - len(L.sot_seq)
    for i in range(max_new):
        logits, past = decoder_logits(L, enc, ids, past)
        logits = logits.float()
        logits = logits.masked_fill(ban, float("-inf"))
        if n_prefix_gen + i == 0:
            logits = logits.masked_fill(begin, float("-inf"))
        if step_ban is not None:
            extra = step_ban(prefix + toks)
            if extra:
                logits[torch.tensor(extra, device=L.device)] = float("-inf")
        lp = torch.log_softmax(logits, -1)
        t = int(torch.argmax(lp))
        p_dig = float(lp[L.digit_mask].exp().sum()) if not ban_digits else 0.0
        if t == L.eos:
            ended = True
            break
        toks.append(t)
        steps.append(Step(t, float(lp[t]), p_dig))
        ids = torch.tensor([[t]], device=L.device)
    return DecodeResult(toks, steps, ended)


@torch.inference_mode()
def transcribe(L: Loaded, audio: np.ndarray, max_new: int = 96, ban_digits: bool = False) -> tuple[str, DecodeResult]:
    enc = encode(L, audio)
    r = greedy(L, enc, list(L.sot_seq), max_new=max_new, ban_digits=ban_digits)
    return L.decode(r.tokens).strip(), r


@torch.inference_mode()
def step_distribution(L: Loaded, enc: torch.Tensor, prefix: list[int], topk: int = 8, ban_digits: bool = False):
    """Probability view at a decision point: top-k tokens + digit-mass."""
    logits, _ = decoder_logits(L, enc, torch.tensor([prefix], device=L.device))
    logits = logits.float()
    if L.suppress_ids:
        logits[torch.tensor(L.suppress_ids, device=L.device)] = float("-inf")
    if ban_digits:
        logits = logits.masked_fill(L.digit_mask, float("-inf"))
    lp = torch.log_softmax(logits, -1)
    top = torch.topk(lp, topk)
    return {
        "p_digit": float(lp[L.digit_mask].exp().sum()),
        "top": [
            {"id": int(i), "tok": L.decode([int(i)], skip_special=False), "p": float(v.exp()), "digit": bool(L.digit_mask[int(i)])}
            for v, i in zip(top.values, top.indices)
        ],
    }


def seq_logprobs(L: Loaded, enc: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Teacher-forced log-probs: out[i] = log p(ids[i+1] | ids[:i+1]); differentiable. Shape [len(ids)-1]."""
    t = torch.tensor([ids], device=L.device)
    out = L.model.model.decoder(input_ids=t[:, :-1], encoder_hidden_states=enc, use_cache=False)
    lp = torch.log_softmax(L.model.proj_out(out.last_hidden_state)[0].float(), -1)
    return lp.gather(1, t[0, 1:].unsqueeze(1)).squeeze(1)
