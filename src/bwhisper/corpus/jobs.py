from __future__ import annotations

import random
import time
from collections import Counter

from ..jobs import Ctx, register
from . import explore, hf_cache, pockettts, store

EXPLORE_FETCH_DEFAULTS = {
    "seeds": [234608111],
    "per_seed": 120,
    "target": 40,
    "per_episode": 6,
    "pad": 0.25,
    "mode": "range",
    "workers": 4,
    "content": "digits",  # digits | latin | saved (use the query stored in the explore config)
    "query": "",  # optional search regex; overrides content
}


@register("corpus_import_pockettts", kind="io", defaults=pockettts.DEFAULTS)
def import_pockettts(ctx: Ctx, cfg: dict) -> dict:
    return pockettts.import_aligned(ctx, cfg)


@register("corpus_import_hf", kind="io",
          defaults={"dataset": None, "limit": 100, "seed": 0, "min_dur": 6.0, "max_dur": 25.0,
                    "content": "digits"})  # digits | latin
def import_hf(ctx: Ctx, cfg: dict) -> dict:
    return hf_cache.import_from_cache(
        dataset=cfg.get("dataset"), limit=int(cfg["limit"]), seed=int(cfg["seed"]),
        min_dur=float(cfg["min_dur"]), max_dur=float(cfg["max_dur"]),
        content=cfg.get("content", "digits"), log=ctx.log, should_stop=ctx.should_stop, progress=ctx.progress,
    )


@register("corpus_fetch_explore", kind="io", defaults=EXPLORE_FETCH_DEFAULTS)
def fetch_explore(ctx: Ctx, cfg: dict) -> dict:
    auth = explore.check_auth()
    if not auth["ok"]:
        raise RuntimeError(f"Explore authentication required: {auth.get('reason', auth.get('status'))}")
    raw_seeds = cfg.get("seeds")
    if isinstance(raw_seeds, (int, str)):
        base_seeds = [int(s.strip()) for s in str(raw_seeds).split(",") if s.strip()]
    elif isinstance(raw_seeds, list) and raw_seeds:
        base_seeds = [int(s) for s in raw_seeds]
    else:
        base_seeds = list(EXPLORE_FETCH_DEFAULTS["seeds"])

    per_seed = max(1, int(cfg.get("per_seed", 120)))
    target = max(1, int(cfg.get("target", 40)))
    per_episode = max(1, int(cfg.get("per_episode", 6)))
    pad = float(cfg.get("pad", 0.25))
    mode = str(cfg.get("mode") or "range")
    workers = max(1, int(cfg.get("workers", 4)))
    query = explore.search_query(cfg.get("content", "digits"), cfg.get("query") or "")

    rng = random.Random(base_seeds[-1])
    seed_queue = list(base_seeds)
    seen_seeds = set(seed_queue)

    total_added = 0
    total_dup = 0
    total_episodes = 0
    total_bytes = 0
    seeds_used: list[int] = []
    episode_counts = Counter(s["meta"].get("uuid") for s in store.query(source="explore", limit=10**9))
    t0 = time.time()

    ctx.progress(0, target, f"starting explore fetch ({mode}, {workers} workers)")
    while total_added < target and len(seeds_used) < 100 and not ctx.should_stop():
        if not seed_queue:
            nxt = rng.randint(100_000_000, 999_999_999)
            while nxt in seen_seeds:
                nxt = rng.randint(100_000_000, 999_999_999)
            seen_seeds.add(nxt)
            seed_queue.append(nxt)

        seed = seed_queue.pop(0)
        want = min(per_seed, target - total_added)
        base_done = total_added

        res = explore.fetch_episodes_batch(
            seed=seed,
            n=want,
            per_episode=per_episode,
            pad=pad,
            mode=mode,
            workers=workers,
            episode_counts=episode_counts,
            query=query,
            log=ctx.log,
            should_stop=ctx.should_stop,
            progress=lambda a, _b, bd=base_done, s=seed: ctx.progress(
                bd + a, target, f"seed {s} ({bd + a}/{target})"
            ),
        )
        seeds_used.append(seed)
        total_added += int(res.get("added", 0))
        total_dup += int(res.get("duplicates_skipped", 0))
        total_episodes += int(res.get("episodes_seen", 0))
        total_bytes += int(res.get("bytes_transferred", 0))
        ctx.progress(total_added, target, f"seed {seed} done ({total_added}/{target})")

    elapsed = max(1e-3, time.time() - t0)
    return {
        "added": total_added,
        "target": target,
        "target_reached": total_added >= target,
        "shortfall": max(0, target - total_added),
        "duplicates_skipped": total_dup,
        "seeds_used": seeds_used,
        "episodes_seen": total_episodes,
        "mode": mode,
        "workers": workers,
        "bytes_transferred": total_bytes,
        "bytes_per_sample": round(total_bytes / total_added, 1) if total_added else 0.0,
        "sec_per_sample": round(elapsed / total_added, 3) if total_added else 0.0,
        "elapsed_sec": round(elapsed, 2),
    }
