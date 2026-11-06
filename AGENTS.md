# AGENTS.md

Project: **biased-whisper** — can we steer a Hebrew Whisper model (ivrit.ai checkpoints) toward
**digits** or **spelled-out cardinal numbers** by nudging a small set of weights?

**Start here → [`knowledge/index.md`](knowledge/index.md).** The `knowledge/` folder is the project's
memory: theory, plan, reviews, hyperparameters, architecture and experiment findings, as an
[OKF v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format) bundle (markdown + YAML
frontmatter, `type` required). It is version-controlled and meant to grow.

## Rules for agents working here

1. **Read before acting**: `knowledge/index.md`, then the concept docs relevant to your task.
2. **Write back what you learn**: new findings → `knowledge/findings/<date>-<slug>.md`; decisions or
   design changes → update the relevant concept doc. Append a line to `knowledge/log.md` (newest first,
   `YYYY-MM-DD` headings) and keep `knowledge/index.md` listing every concept.
   Run `uv run python scripts/okf_lint.py` before committing knowledge changes.
3. **Frontmatter**: every concept file needs `type`; also give `title`, `description`, `tags`,
   `generated: {by: <agent>/<model>, at: <ISO8601Z>}`, and `status: draft|stable|deprecated`.
   Use absolute bundle links (`/reference/hyperparameters.md`). Never use `index.md`/`log.md` as concept names.
4. **Never commit** data, model weights, audio, tensors, cookies/secrets. `data/` is gitignored. Anything
   under `data/secrets/` (explore.ivrit.ai cookie) must stay local.
5. **Everything is a job** (`src/bwhisper/jobs.py`). The UI and CLI call the same job registry; results live in
   `data/runs/<run-id>/`. Prefer adding a job type + UI form over one-off scripts, so state stays visible in the UI.
6. Hebrew text is RTL: in the UI use the `.rtl` class; never reverse strings in code.

## Commands

```bash
uv sync --python 3.13                       # env (torch, transformers, fastapi, ...)
uv run bw serve --port 8765                 # UI + API  -> http://localhost:8765
uv run bw corpus-import --limit 100         # offline corpus from the local HF cache
uv run bw run phase1 '{"limit": 20, "model": "yoad/whisper-tiny"}'
uv run bw run phase2 '{"phase1_runs": ["<phase1-run-id>"]}'
uv run bw run phase3 '{"phase2_run": "<phase2-run-id>"}'
uv run bw job-types                         # every job type with its default config
uv run pytest                               # tests (decoder parity, aggregation statistics)
```

Models: `yoad/whisper-tiny` (smoke) and `yoad/whisper-small` (local inference); servers use larger models.
Layout: `src/bwhisper/` (python), `web/` (Alpine.js UI, CDN only, no bundler), `knowledge/` (OKF), `tests/`, `data/` (ignored).
