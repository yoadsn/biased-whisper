"""A flat, stable, addressable view over a *subset* of a model's parameters.

Flat index <-> (parameter name, local index) is deterministic given the model
architecture and the filter, so suspect sets can be stored as bare integer
arrays and re-applied later (Phase 3) or on a bigger model of the same family
with the same filter.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import torch

PRESETS = {
    "bitfit": lambda n: n.endswith(".bias") or "layer_norm" in n,
    "decoder": lambda n: n.startswith("model.decoder."),
    "decoder_no_embed": lambda n: n.startswith("model.decoder.") and "embed_" not in n,
    "encoder": lambda n: n.startswith("model.encoder.") and "embed_positions" not in n,
    "all_no_embed": lambda n: "embed_tokens" not in n and "embed_positions" not in n,
    "all": lambda n: True,
}


def tower_of(name: str) -> str:
    if name.startswith("model.encoder."):
        return "encoder"
    if name.startswith("model.decoder."):
        return "decoder"
    return "other"


def group_of(name: str) -> str:
    """Coarse group used for summaries: tower.layer.module-kind."""
    m = re.match(r"model\.(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$", name)
    if m:
        tower, layer, mod, kind = m.groups()
        return f"{tower}.L{int(layer):02d}.{mod}.{kind}"
    return name


class ParamSpace:
    def __init__(self, model: torch.nn.Module, preset: str = "bitfit", include_regex: str = "", exclude_regex: str = ""):
        if preset not in PRESETS:
            raise ValueError(f"unknown param_filter {preset!r}; options: {list(PRESETS)}")
        self.preset = preset
        inc = re.compile(include_regex) if include_regex else None
        exc = re.compile(exclude_regex) if exclude_regex else None
        self.names: list[str] = []
        self.params: list[torch.nn.Parameter] = []
        for n, p in model.named_parameters():
            if not PRESETS[preset](n):
                continue
            if inc and not inc.search(n):
                continue
            if exc and exc.search(n):
                continue
            self.names.append(n)
            self.params.append(p)
        self.shapes = [tuple(p.shape) for p in self.params]
        self.sizes = np.array([p.numel() for p in self.params], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]) if len(self.sizes) else np.zeros(0, np.int64)
        self.total = int(self.sizes.sum())
        self.has_encoder = any(n.startswith("model.encoder.") for n in self.names)
        self.signature = hashlib.sha1(
            "|".join(f"{n}:{tuple(p.shape)}" for n, p in zip(self.names, self.params)).encode()
        ).hexdigest()[:16]

    # ---- gradient plumbing
    def enable_grad(self) -> None:
        for p in self.params:
            p.requires_grad_(True)
            p.grad = None

    def disable_grad(self) -> None:
        for p in self.params:
            p.requires_grad_(False)
            p.grad = None

    def flat_grad(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """Pack gradients on the requested device; CPU remains the legacy default."""
        parts = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float().to(device)
                 for p in self.params]
        return torch.cat(parts)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    # ---- addressing
    def tensor_ids(self, flat_idx: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.starts, flat_idx, side="right") - 1

    def locate(self, flat_idx: int) -> dict:
        t = int(self.tensor_ids(np.array([flat_idx]))[0])
        local = int(flat_idx - self.starts[t])
        shape = self.shapes[t]
        return {
            "flat": int(flat_idx),
            "module": self.names[t],
            "tower": tower_of(self.names[t]),
            "group": group_of(self.names[t]),
            "local": local,
            "index": [int(x) for x in np.unravel_index(local, shape)],
            "shape": list(shape),
        }

    def rms_per_tensor(self, flat: torch.Tensor) -> np.ndarray:
        out = np.empty(len(self.sizes), dtype=np.float32)
        for i, (s, n) in enumerate(zip(self.starts, self.sizes)):
            seg = flat[int(s) : int(s + n)]
            out[i] = float(seg.pow(2).mean().sqrt())
        out[out == 0] = 1.0
        return out

    def energy_per_tensor(self, flat: torch.Tensor) -> np.ndarray:
        return np.array([float(flat[int(s) : int(s + n)].pow(2).sum()) for s, n in zip(self.starts, self.sizes)], dtype=np.float64)

    # ---- editing
    def apply_flat(self, delta: torch.Tensor, scale: float = 1.0) -> None:
        """Add a dense flat delta (CPU) to every parameter in the space."""
        with torch.no_grad():
            for p, s, n in zip(self.params, self.starts, self.sizes):
                p.add_(delta[int(s) : int(s + n)].view(p.shape).to(p.device, p.dtype), alpha=scale)

    def snapshot(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.params]

    def restore_snapshot(self, snap: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, s in zip(self.params, snap):
                p.copy_(s)

    def add_sparse(self, idx: np.ndarray, delta: np.ndarray):
        """Add delta at flat indices (sorted ascending). Returns an undo token."""
        idx = np.asarray(idx, dtype=np.int64)
        delta = np.asarray(delta, dtype=np.float32)
        if idx.ndim != 1 or delta.shape != idx.shape:
            raise ValueError("sparse indices/values must be matching vectors")
        bounds = self.sparse_bounds(idx)
        undo = []
        try:
            with torch.no_grad():
                for t, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                    if lo == hi:
                        continue
                    p = self.params[t]
                    local = torch.from_numpy(idx[lo:hi] - self.starts[t]).to(p.device)
                    flat = p.view(-1)
                    old = flat[local].clone()
                    undo.append((t, local, old))
                    flat.index_copy_(0, local, old + torch.from_numpy(delta[lo:hi]).to(p.device, p.dtype))
        except Exception:
            self.undo_sparse(undo)
            raise
        return undo

    def sparse_bounds(self, idx, validate: bool = True) -> list[int]:
        """Split sorted unique indices with O(tensors log k) boundary lookups."""
        keys = np.concatenate([self.starts, [self.total]])
        if isinstance(idx, torch.Tensor):
            if idx.ndim != 1 or idx.dtype != torch.int64:
                raise ValueError("device sparse indices must be a 1-D int64 tensor")
            if validate and idx.numel() and (int(idx[0]) < 0 or int(idx[-1]) >= self.total
                                              or not bool(torch.all(idx[1:] > idx[:-1]))):
                raise ValueError("sparse indices must be sorted, unique and in range")
            return torch.searchsorted(idx, torch.as_tensor(keys, device=idx.device)).cpu().tolist()
        if validate and len(idx) and (idx[0] < 0 or idx[-1] >= self.total or np.any(idx[1:] <= idx[:-1])):
            raise ValueError("sparse indices must be sorted, unique and in range")
        return np.searchsorted(idx, keys).tolist()

    def add_sparse_device(self, idx: torch.Tensor, delta: torch.Tensor, bounds=None):
        """Apply a sorted support without exporting indices/values to the host."""
        if idx.ndim != 1 or delta.shape != idx.shape:
            raise ValueError("sparse indices/values must be matching vectors")
        bounds = self.sparse_bounds(idx) if bounds is None else bounds
        if len(bounds) != len(self.params) + 1 or bounds[0] != 0 or bounds[-1] != len(idx):
            raise ValueError("invalid sparse tensor boundaries")
        undo = []
        try:
            with torch.no_grad():
                for t, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                    if lo == hi:
                        continue
                    p = self.params[t]
                    local = (idx[lo:hi] - int(self.starts[t])).to(p.device)
                    flat = p.view(-1)
                    old = flat[local].clone()
                    undo.append((t, local, old))
                    flat.index_copy_(0, local, old + delta[lo:hi].to(p.device, p.dtype))
        except Exception:
            self.undo_sparse(undo)
            raise
        return undo

    def undo_sparse(self, undo) -> None:
        with torch.no_grad():
            for t, local, old in undo:
                self.params[t].view(-1)[local] = old


class LightSpace(ParamSpace):
    """Addressing only (no model): rebuilt from a run's space.json."""

    def __init__(self, d: dict):
        self.names = d["names"]
        self.shapes = [tuple(s) for s in d["shapes"]] if "shapes" in d else [(int(n),) for n in d["sizes"]]
        self.sizes = np.array(d["sizes"], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]) if len(self.sizes) else np.zeros(0, np.int64)
        self.total = int(self.sizes.sum())
        self.params = []
