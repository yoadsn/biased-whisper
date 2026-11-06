"""Phase-2 aggregation: counts -> common set -> random-group voting -> FDR.

Pure numpy; operates on per-sample sparse nudge files (idx, val, rms) so it
can be unit-tested without a model.

Pipeline per nudge type
  1. counts[p] = number of samples whose top-`sample_pct` set contains p
  2. union = {p: counts>0}; common set C = top `common_pct` of union by count
     (stable: ties broken by flat index)
  3. each sample -> dense vector over C (0 where the sample's set lacks p)
  4. voting: draw `group_size` samples, rescale each to [-1,1] (divide by max|v|),
     sum; params in the top `agree_top_pct` of |sum| (and nonzero) get +1
  5. null: identical procedure with each sample's vector randomly permuted over
     C (destroys cross-sample alignment, keeps per-sample magnitude profile)
  6. threshold = smallest vote count whose empirical FDR (null>=t / real>=t) <= fdr
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np


def load_sample(path: Path) -> dict:
    z = np.load(path)
    return {"idx": z["idx"].astype(np.int64), "val": z["val"], "rms": z["rms"], "energy": z["energy"] if "energy" in z else None}


def normalized(vals: np.ndarray, idx: np.ndarray, rms: np.ndarray, tensor_ids: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rms":
        return vals / rms[tensor_ids]
    return vals


class Aggregator:
    def __init__(self, files: list[Path], total: int, tensor_id_fn: Callable[[np.ndarray], np.ndarray], normalize: str = "rms", cache_gb: float = 2.0):
        self.files, self.total, self.tid, self.mode = files, total, tensor_id_fn, normalize
        self.cache_bytes = int(cache_gb * 1e9)
        self.C: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    # 1-2
    def build_common_set(self, common_pct: float, on_file: Callable[[int], None] = lambda i: None) -> dict:
        counts = np.zeros(self.total, dtype=np.uint16)
        for i, f in enumerate(self.files):
            counts[load_sample(f)["idx"]] += 1
            on_file(i)
        union = np.flatnonzero(counts)
        n_common = max(1, int(math.ceil(len(union) * common_pct / 100.0)))
        order = np.argsort(-counts[union].astype(np.int64), kind="stable")[:n_common]
        self.C = np.sort(union[order])
        self.counts = counts
        return {"union": int(len(union)), "common": int(len(self.C)), "min_count_in_common": int(counts[self.C].min()) if len(self.C) else 0,
                "max_count": int(counts.max())}

    # 3
    def dense(self, i: int, raw: bool = False) -> np.ndarray:
        if not raw and i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        s = load_sample(self.files[i])
        pos = np.searchsorted(self.C, s["idx"])
        pos[pos >= len(self.C)] = len(self.C) - 1
        ok = self.C[pos] == s["idx"]
        v = np.zeros(len(self.C), dtype=np.float32)
        vals = s["val"] if raw else normalized(s["val"], s["idx"], s["rms"], self.tid(s["idx"]), self.mode)
        v[pos[ok]] = vals[ok]
        if not raw:
            self._cache[i] = v
            while self._cache and len(self._cache) * v.nbytes > self.cache_bytes:
                self._cache.popitem(last=False)
        return v

    # 4-5
    def vote(self, rounds: int, group_size: int, top_pct: float, seed: int, null: bool = False,
             on_round: Callable[[int], None] = lambda r: None, should_stop: Callable[[], bool] = lambda: False) -> dict:
        n, C = len(self.files), len(self.C)
        if n < group_size:
            raise ValueError(f"need at least group_size={group_size} samples, have {n}")
        rng = np.random.default_rng(seed + (7919 if null else 0))
        k = max(1, int(round(C * top_pct / 100.0)))
        votes = np.zeros(C, dtype=np.int32)
        halves = [np.zeros(C, dtype=np.int32), np.zeros(C, dtype=np.int32)]
        perms: dict[int, np.ndarray] = {}
        done = 0
        for r in range(rounds):
            if should_stop():
                break
            pick = rng.choice(n, group_size, replace=False)
            s = np.zeros(C, dtype=np.float32)
            for j in pick:
                v = self.dense(int(j))
                if null:
                    if j not in perms:
                        perms[int(j)] = np.random.default_rng([seed, int(j)]).permutation(C).astype(np.int32)
                    v = v[perms[int(j)]]
                m = float(np.abs(v).max())
                if m > 0:
                    s += v / m
            a = np.abs(s)
            nnz = int(np.count_nonzero(a))
            kk = min(k, nnz)
            if kk > 0:
                top = np.argpartition(-a, kk - 1)[:kk]
                votes[top] += 1
                halves[r % 2][top] += 1
            done += 1
            if r % 50 == 0:
                on_round(r)
        return {"votes": votes, "halves": halves, "rounds": done, "k": k}

    # stats over all samples (raw nudge values)
    def stats(self, cand_pos: np.ndarray) -> dict:
        n = len(self.files)
        sum_all = np.zeros(len(self.C), dtype=np.float64)
        cnt = np.zeros(len(self.C), dtype=np.int32)
        for i in range(n):
            v = self.dense(i, raw=True)
            sum_all += v
            cnt += (v != 0)
        mat = np.stack([self.dense(i, raw=True)[cand_pos] for i in range(n)]) if len(cand_pos) else np.zeros((n, 0), np.float32)
        present = mat != 0
        cp = np.maximum(present.sum(0), 1)
        mean_present = mat.sum(0) / cp
        masked = np.where(present, mat, np.nan)
        with np.errstate(all="ignore"):
            median_present = np.nan_to_num(np.nanmedian(masked, axis=0)) if mat.shape[1] else np.zeros(0)
            std_present = np.nan_to_num(np.nanstd(masked, axis=0)) if mat.shape[1] else np.zeros(0)
        mean_all = sum_all[cand_pos] / n
        sign_cons = np.where(present, np.sign(mat) == np.sign(mean_present), False).sum(0) / cp
        return {
            "mean_all": mean_all.astype(np.float32), "mean_present": mean_present.astype(np.float32),
            "median_present": median_present.astype(np.float32), "std_present": std_present.astype(np.float32),
            "count": present.sum(0).astype(np.int32), "sign_consistency": sign_cons.astype(np.float32),
        }


def pick_threshold(real: np.ndarray, null: np.ndarray, fdr: float) -> tuple[int | None, float | None]:
    rs, ns = np.sort(real), np.sort(null)
    best, best_fdr = None, None
    for t in np.unique(real)[::-1]:
        if t <= 0:
            break
        nr = len(rs) - np.searchsorted(rs, t, "left")
        nn = len(ns) - np.searchsorted(ns, t, "left")
        f = nn / max(nr, 1)
        if f <= fdr:
            best, best_fdr = int(t), float(f)
    return best, best_fdr


def concentration(votes: np.ndarray) -> dict:
    """Descriptive: how concentrated are votes (the 'top 10% hold X%' idea)."""
    v = np.sort(votes.astype(np.float64))[::-1]
    tot = v.sum()
    if tot == 0 or len(v) == 0:
        return {"gini": 0.0, "top1_share": 0.0, "top10_share": 0.0}
    cum = np.cumsum(v) / tot
    n = len(v)
    asc = v[::-1]
    gini = float((2 * np.sum((np.arange(1, n + 1)) * asc) / (n * tot)) - (n + 1) / n)
    return {"gini": gini, "top1_share": float(cum[max(0, int(n * 0.01) - 1)]), "top10_share": float(cum[max(0, int(n * 0.10) - 1)])}


def jaccard_topk(a: np.ndarray, b: np.ndarray, frac: float = 0.01) -> float:
    k = max(1, int(len(a) * frac))
    ta, tb = set(np.argpartition(-a, k - 1)[:k].tolist()), set(np.argpartition(-b, k - 1)[:k].tolist())
    return len(ta & tb) / max(1, len(ta | tb))
