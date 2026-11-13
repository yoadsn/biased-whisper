"""Scriptable entry point: every UI action is a job type runnable here too.

  uv run bw serve                      # start the UI/API
  uv run bw corpus-import --limit 100  # import digit-containing utterances from the local HF cache
  uv run bw run phase1 '{"limit": 20}'
  uv run bw jobs
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bw")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    c = sub.add_parser("corpus-import")
    c.add_argument("--dataset", default=None)
    c.add_argument("--limit", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--content", choices=["digits", "latin"], default="digits")

    r = sub.add_parser("run", help="run a job synchronously")
    r.add_argument("type")
    r.add_argument("config", nargs="?", default="{}", help="JSON config (overrides defaults)")

    sub.add_parser("jobs")
    sub.add_parser("job-types")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        uvicorn.run("bwhisper.api:app", host=a.host, port=a.port, log_level="info")
        return 0

    from . import api  # noqa: F401  (registers all job types)
    from .jobs import REGISTRY, manager

    if a.cmd == "corpus-import":
        j = manager().run_sync("corpus_import_hf", {"dataset": a.dataset, "limit": a.limit, "seed": a.seed, "content": a.content})
    elif a.cmd == "run":
        j = manager().run_sync(a.type, json.loads(a.config))
    elif a.cmd == "jobs":
        for d in manager().list():
            print(d["id"], d["status"], d["progress"], d.get("message", ""))
        return 0
    elif a.cmd == "job-types":
        for k, v in REGISTRY.items():
            print(k, json.dumps(v["defaults"], ensure_ascii=False))
        return 0
    print(json.dumps(j.to_dict(), ensure_ascii=False, indent=1))
    return 0 if j.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
