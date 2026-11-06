from __future__ import annotations

from ..jobs import Ctx, register
from . import explore, hf_cache


@register("corpus_import_hf", kind="io",
          defaults={"dataset": None, "limit": 100, "seed": 0, "min_dur": 6.0, "max_dur": 25.0})
def import_hf(ctx: Ctx, cfg: dict) -> dict:
    return hf_cache.import_from_cache(
        dataset=cfg.get("dataset"), limit=int(cfg["limit"]), seed=int(cfg["seed"]),
        min_dur=float(cfg["min_dur"]), max_dur=float(cfg["max_dur"]),
        log=ctx.log, should_stop=ctx.should_stop, progress=ctx.progress,
    )


@register("corpus_fetch_explore", kind="io", defaults={"seed": 234608111, "limit": 40})
def fetch_explore(ctx: Ctx, cfg: dict) -> dict:
    return explore.fetch_batch(int(cfg["seed"]), int(cfg["limit"]), log=ctx.log,
                               should_stop=ctx.should_stop, progress=ctx.progress)
