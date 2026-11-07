"""Batched inference for Phase 3; never retain encoded states across edits."""
from __future__ import annotations

import numpy as np
import torch

from . import model as M
from .audio import load_audio
from .phase2 import point_objective, resolve_kind


class FeatureBank:
    """Only weight-independent CPU mel features survive a condition boundary."""

    def __init__(self, L, samples, ctx):
        self.values = {}
        for sample in samples:
            ctx.check()
            sid = sample["id"]
            if sid not in self.values:
                audio = load_audio(sample["audio_path"])
                # Individual extraction exactly matches M.features (including
                # padding/truncation), but avoids persistent GPU feature caches.
                self.values[sid] = L.feature_extractor(
                    audio, sampling_rate=M.SAMPLE_RATE, return_tensors="pt"
                ).input_features.cpu()

    @torch.inference_mode()
    def encode(self, L, ids):
        features = torch.cat([self.values[sid] for sid in ids]).to(L.device)
        return L.model.model.encoder(features).last_hidden_state


@torch.inference_mode()
def greedy_tokens(L, enc, max_new, check=lambda: None):
    """Same suppression/forced prefix as M.greedy; no unused Step diagnostics.

    All rows retain their KV-cache slot. Finished rows feed EOS; their later
    tokens are discarded. No row can influence another row's causal state.
    """
    if max_new < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    batch = enc.shape[0]
    ban = torch.zeros(L.vocab, dtype=torch.bool, device=L.device)
    if L.suppress_ids:
        ban[torch.tensor(L.suppress_ids, device=L.device)] = True
    ban[torch.tensor(L.sot_seq, device=L.device)] = True
    ban[L.eos + 1:] = True
    begin = ban.clone()
    if L.suppress_begin:
        begin[torch.tensor(L.suppress_begin, device=L.device)] = True
    ids = torch.tensor([L.sot_seq], device=L.device).expand(batch, -1)
    finished = torch.zeros(batch, dtype=torch.bool, device=L.device)
    output = torch.empty((batch, max_new), dtype=torch.long, device=L.device)
    past, count = None, 0
    for step in range(max_new):
        if step % 8 == 0:
            check()
        out = L.model.model.decoder(input_ids=ids, encoder_hidden_states=enc,
                                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = L.model.proj_out(out.last_hidden_state[:, -1]).float()
        logits.masked_fill_(begin if step == 0 else ban, float("-inf"))
        # Keep reference log-softmax rounding/tie behavior; skip only diagnostics.
        token = torch.log_softmax(logits, -1).argmax(-1)
        token = torch.where(finished, L.eos, token)
        output[:, step] = token
        finished |= token == L.eos
        ids = token[:, None]
        count = step + 1
        if bool(finished.all()):  # one scalar sync per batch, not 3 per row/token
            break
    result = []
    for row in output[:, :count].cpu().tolist():
        result.append(row[:row.index(L.eos)] if L.eos in row else row)
    return result


@torch.inference_mode()
def probe_values(L, enc, pts, kind):
    """Batch variable-length sequence contrasts; retain all legacy objectives.

    Right padding cannot affect earlier causal positions. Select only the
    divergent target positions before the vocabulary projection, avoiding a
    batch x full-sequence x vocabulary temporary.
    """
    if not pts:
        return []
    if any(resolve_kind(pt, kind) != "seq_contrast" for pt in pts):
        return torch.stack([point_objective(L, enc[i:i + 1], pt, kind)
                            for i, pt in enumerate(pts)]).cpu().tolist()
    scores = []
    for style in ("digit", "cardinal"):
        sequences = [list(L.sot_seq) + pt["y_digit" if style == "digit" else "y_cardinal"][:pt["div_" + style][1]]
                     for pt in pts]
        length = max(len(seq) - 1 for seq in sequences)
        ids = torch.full((len(pts), length), L.eos, dtype=torch.long, device=L.device)
        row_ids, positions, targets = [], [], []
        for row, (pt, seq) in enumerate(zip(pts, sequences)):
            ids[row, :len(seq) - 1] = torch.tensor(seq[:-1], device=L.device)
            a, b = pt["div_" + style]
            start, end = len(L.sot_seq) + a - 1, len(L.sot_seq) + b - 1
            row_ids.extend([row] * (end - start))
            positions.extend(range(start, end))
            targets.extend(seq[start + 1:end + 1])
        hidden = L.model.model.decoder(input_ids=ids, encoder_hidden_states=enc, use_cache=False).last_hidden_state
        rows = torch.tensor(row_ids, dtype=torch.long, device=L.device)
        pos = torch.tensor(positions, dtype=torch.long, device=L.device)
        target = torch.tensor(targets, dtype=torch.long, device=L.device)
        lp = torch.log_softmax(L.model.proj_out(hidden[rows, pos]).float(), -1)
        values = lp.gather(1, target[:, None]).squeeze(1)
        # Sum each span independently, matching the reference reduction order.
        counts = np.bincount(row_ids, minlength=len(pts)).tolist()
        offset, sums = 0, []
        for count in counts:
            sums.append(values[offset:offset + count].sum())
            offset += count
        scores.append(torch.stack(sums))
    return (scores[0] - scores[1]).cpu().tolist()
