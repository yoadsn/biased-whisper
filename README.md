# biased-whisper

Can we steer a Hebrew Whisper model (ivrit.ai checkpoints) toward **digits** (`20`) or **spelled-out cardinal numbers** (`עשרים`) by nudging a small set of weights or neurons—without touching the decoder loop or logits processor?

**Start here → [`knowledge/index.md`](knowledge/index.md)** for the full hypothesis, methodology (`Approach v2`), architecture, and experimental findings (OKF v0.2 bundle).

## Key findings (`ivrit-ai/whisper-v2-d3-e3`, `whisper-small`)

* **Existence proof (Phase 1):** On the same audio clip, the model assigns substantial probability mass to both digit and cardinal surface forms at the style-decision token.
* **Style patterns (Phase 1.5):** Hebrew prefix letters (`ב/ל/מ/כ/ו/ה`) attach directly to digits (`ב20`) or via hyphen (`ב-20`) alongside bare digits and cardinal words (`בעשרים`).
* **Where the decision lives (Phase 2N & Phase 3):** A compact set of upper-decoder MLP neurons (`layers 9–11 fc2`) and final layer-norm weights governs the digit-vs-cardinal choice. Editing **64 units** (`0.02%` of parameters) shifts digit output from **35% → 69%** (digit steer) or **35% → 4%** (cardinal steer) on held-out audio while keeping non-number tokens intact (~88% unchanged).
* **Cross-domain transfer:** Independent fits on Knesset speech (`hf_cache`) and multi-source podcasts/interviews (`explore.ivrit.ai`) converge on the same units ($r = 0.87$, **71 / 100** top units shared) and transfer across domains in both directions.

## Quickstart

```bash
uv sync --python 3.13                       # install dependencies
uv run bw serve --port 8765                 # launch Alpine.js UI + FastAPI server -> http://localhost:8765
uv run bw job-types                         # list available job types and default configs
uv run pytest                               # run test suite
uv run python scripts/okf_lint.py           # validate knowledge/ OKF bundle
```

### CLI workflow

**A40 server setup (data/preflight only):** see
[`knowledge/reference/server-setup.md`](knowledge/reference/server-setup.md).

```bash
uv run bw run corpus_import_pockettts '{"limit":10000,"workers":8}'
# Explore requires your authenticated local configuration first:
uv run bw run corpus_fetch_explore '{"target":10000,"per_seed":500,"workers":8}'
uv run bw run setup_scale '{}'
```

Setup writes frozen recording-disjoint cohorts and an eight-device Phase 1 config;
it **does not start any experimental phase**. Wait for owner approval before the
workflow below. Original-text digit availability may be below the requested local
target; inventory files report the shortfall without using normalized variants.

```bash
# 1. Import corpus clips containing numbers
uv run bw corpus-import --limit 100

# 2. Phase 1: teacher-forced existence proof & style-span marginals
uv run bw run phase1 '{"limit": 100, "model": "yoad/whisper-small"}'

# 3. Phase 1.5: mine prefix / hyphen / unit surface patterns around numbers
uv run bw run phase1_5 '{"phase1_runs": ["<phase1-run-id>"]}'

# 4. Phase 2 / 2N: discover parameter or neuron/head edits
uv run bw run phase2n '{"phase1_runs": ["<phase1-run-id>"], "top_n": 64}'

# 5. Phase 3: evaluate weight-edited checkpoints on held-out audio
uv run bw run phase3 '{"phase2_run": "<phase2n-run-id>"}'
```

## Repository layout

* `src/bwhisper/` — core library, FastAPI backend, and `bw` CLI (`corpus/`, `phase1.py`, `phase1_5.py`, `phase2.py`, `phase2n.py`, `phase3.py`, `budget.py`, `hebrew_numbers.py`)
* `web/` — zero-build Alpine.js UI for corpus browsing, job execution, and side-by-side transcripts
* `knowledge/` — Open Knowledge Format (OKF v0.2) documentation and experiment findings
* `tests/` — unit and parity tests
* `data/` — local runtime corpus, audio clips, and experiment runs (gitignored)
