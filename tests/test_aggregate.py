import numpy as np

from bwhisper.aggregate import Aggregator, pick_threshold


def make_files(tmp_path, n=12, total=2000, signal=60, seed=0):
    """Plant `signal` params that consistently carry a sign across samples; rest is noise."""
    rng = np.random.default_rng(seed)
    planted = np.sort(rng.choice(total, signal, replace=False))
    files = []
    for i in range(n):
        noise_idx = rng.choice(total, 150, replace=False)
        idx = np.union1d(planted, noise_idx)
        val = rng.standard_normal(len(idx)).astype(np.float32) * 0.3
        m = np.isin(idx, planted)
        val[m] = np.sign(planted[np.searchsorted(planted, idx[m])] % 2 - 0.5) * (1.5 + rng.random(m.sum()))
        p = tmp_path / f"s{i}.npz"
        np.savez(p, idx=idx.astype(np.uint32), val=val, rms=np.ones(1, np.float32), energy=np.zeros(1))
        files.append(p)
    return files, planted


def test_planted_signal_recovered(tmp_path):
    files, planted = make_files(tmp_path)
    agg = Aggregator(files, 2000, lambda idx: np.zeros(len(idx), np.int64), normalize="none")
    info = agg.build_common_set(common_pct=100)
    assert info["union"] >= len(planted)
    real = agg.vote(400, 3, 20.0, seed=1)
    null = agg.vote(400, 3, 20.0, seed=1, null=True)
    thr, fdr = pick_threshold(real["votes"], null["votes"], 0.05)
    assert thr is not None
    found = agg.C[real["votes"] >= thr]
    recall = np.isin(planted, found).mean()
    precision = np.isin(found, planted).mean()
    assert recall > 0.8 and precision > 0.8, (recall, precision, thr, len(found))


def test_pure_noise_yields_no_candidates(tmp_path):
    rng = np.random.default_rng(3)
    files = []
    for i in range(10):
        idx = np.sort(rng.choice(3000, 300, replace=False))
        p = tmp_path / f"n{i}.npz"
        np.savez(p, idx=idx.astype(np.uint32), val=rng.standard_normal(300).astype(np.float32), rms=np.ones(1, np.float32), energy=np.zeros(1))
        files.append(p)
    agg = Aggregator(files, 3000, lambda idx: np.zeros(len(idx), np.int64), normalize="none")
    agg.build_common_set(100)
    real = agg.vote(300, 3, 30.0, seed=1)
    null = agg.vote(300, 3, 30.0, seed=1, null=True)
    thr, _ = pick_threshold(real["votes"], null["votes"], 0.05)
    n = 0 if thr is None else int((real["votes"] >= thr).sum())
    assert n < 0.1 * len(agg.C)
