<p align="center">
  <img src="docs/assets/aforge-flow-logo.png" alt="aforge-flow logo" width="220">
</p>

<h1 align="center">aforge-flow</h1>

<p align="center">
  <b>GPU-accelerated artwork pipeline: category-aware background removal, optimization,
  meaningful names and zero-shot classification.</b><br>
  Built for tabletop RPG / VTT art libraries, and useful for any batch of web assets.
</p>

<p align="center">
  <a href="https://github.com/brunohaf/aforge-flow/actions/workflows/ci.yml"><img src="https://github.com/brunohaf/aforge-flow/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <img src="https://img.shields.io/badge/CUDA-optional-76B900.svg?logo=nvidia&logoColor=white" alt="CUDA optional">
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: alpha">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#profiles">Tabletop profile</a> •
  <a href="#make-targets">Make targets</a> •
  <a href="#background-removal-models">Models</a> •
  <a href="#classification">Classification</a> •
  <a href="#web-ui-work-in-progress">Web UI</a> •
  <a href="#development">Development</a>
</p>

## Features

Background removal uses a different segmentation model per asset category:
BRIA RMBG-2.0 for figures, BiRefNet-DIS for single objects and BiRefNet-Portrait
for faces and hair. It runs on rembg and ONNX Runtime and writes clean alpha PNGs.
When the GPU runs out of memory, it rebuilds the session, then falls back to CPU.

Media optimization converts images to WebP, PNG or JPEG, and video to WebM
(AV1 NVENC on a capable GPU, VP9 otherwise), MP4 or MKV. PNG output is always
pixel-exact, transparency included.

Renaming keeps a file's real name as a slug, so `Brain in a Jar.png` becomes
`brain-jar-1.png`. Only junk names, such as numeric IDs and hashes, go to a
captioning model (Qwen2.5-VL or BLIP). Near-duplicate pictures share one slug.

Zero-shot classification tags artwork with the 14 D&D creature types from the
image content alone, using a CLIP-family encoder (SigLIP 2 by default). Embeddings
are cached, so re-runs do no GPU work. An optional VLM looks only at the
low-confidence cases.

For large batches, aforge mirrors the input folder structure, skips duplicates and
existing outputs, and writes structured logs. `aforge recommend` and `--auto-tune`
pick `--workers` and GPU or CPU for your machine.

## Installation

### Prerequisites

| Tool                                                          | Needed for                                                                            |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Required. Python 3.11, virtualenv and dependencies                                    |
| GNU make                                                      | Optional. Shortcut targets (on Windows: `scoop install make` or `choco install make`) |
| [FFmpeg](https://ffmpeg.org/download.html)                    | Video only                                                                            |
| NVIDIA driver                                                 | Optional. GPU acceleration; everything also runs on CPU                               |
| [direnv](https://direnv.net)                                  | Optional. Activates `.venv` automatically when you `cd` into the repo                 |

### Install

```bash
git clone https://github.com/brunohaf/aforge-flow.git
cd aforge-flow
make deps-install        # = uv sync --frozen --all-extras
make doctor              # checks uv / ffmpeg / GPU driver and the torch build
```

On Windows and Linux, uv installs CUDA 12.6 PyTorch wheels from the
PyTorch index (see `[tool.uv.sources]` in `pyproject.toml`). PyPI only ships CPU
builds for Windows.

`make deps-install-core` installs only the core and dev dependencies. To pick
extras yourself, run `uv sync --extra web --extra classify`.

| Extra      | Adds                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------- |
| `classify` | Creature classification and Qwen captioning (`transformers`, `accelerate`, `bitsandbytes` on Linux) |
| `web`      | Local web UI (`fastapi`, `uvicorn`)                                                                 |
| `video`    | `ffmpeg-python`, `av`                                                                               |
| `dev`      | pytest, ruff, black, mypy                                                                           |

### Activate the environment

| Option              | Command                                                            |
| ------------------- | ------------------------------------------------------------------ |
| direnv (automatic)  | `make direnv` once; after that `cd aforge-flow` activates `.venv`  |
| Manual (bash/zsh)   | `source .venv/bin/activate`                                        |
| Manual (PowerShell) | `.\.venv\Scripts\Activate.ps1`                                     |
| No activation       | `uv run aforge ...` or the `make` targets                          |

pip still works if you don't use uv: `python -m venv .venv && pip install -e ".[classify,web]"`.
It installs PyPI's torch, which is CPU-only on Windows.

## Quick start

Put your files in `assets/` (the default input). Results go to `output/`.

```bash
make dirs                                  # creates assets/ and output/
make optimize-lossless                     # pixel-exact PNG recompression, keeps transparency
make remove-bg CATEGORY=item               # transparent PNGs using the item model
make process                               # background removal + WebP
make tabletop                              # + creature classification + meaningful names
```

The same commands without make:

```bash
aforge optimize -i assets -o output --image-format png --compression lossless
aforge remove-background -i assets -o output --category item
aforge process -i assets -o output --image-format webp
aforge process -i assets -o output --profile tabletop
aforge rename -i assets -o output            # meaningful names only
aforge classify -i assets --vlm-fallback     # writes <image>.classification.json
aforge recommend -i assets                   # suggested --workers / --gpu for this machine
```

Run `aforge -h` or `aforge <command> -h` to see every option. `go` and `pipe` are aliases
for `process`, `optimize-media` for `optimize`, and `rename` for `caption-rename`.

### Output naming

| Step                | Output                                                         |
| ------------------- | -------------------------------------------------------------- |
| `optimize`          | `output/<subdir>/<name>_optimized.<ext>`                       |
| `remove-background` | `output/<subdir>/<name>_nobg.png`                              |
| `caption-rename`    | `output/<subdir>/<slug>-<n>.<ext>` (e.g. `balhannoth-1.png`)   |
| `classify`          | `<input>.classification.json` sidecar, next to the input image |

### Compression levels

| `--compression`                            | PNG                                       | WebP / JPEG                                                    |
| ------------------------------------------ | ----------------------------------------- | -------------------------------------------------------------- |
| `lossless`                                 | Lossless (strongest deflate)              | Quality 95, still lossy (see [Known issues](#known-issues))    |
| `high_quality` / `balanced` / `aggressive` | Lossless (Pillow ignores quality for PNG) | Lower quality, smaller files                                   |

## Profiles

A profile is a named set of option defaults for one use case. Flags you pass
explicitly still win, wherever they appear on the command line.

| Profile    | What it turns on                                                                                                       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| `default`  | Background removal + media optimization                                                                                |
| `tabletop` | Adds D&D creature classification (`--classify`) and creature-aware naming (`--caption-rename`) for RPG / VTT libraries |

```bash
aforge process --profile tabletop                 # classify + rename on
aforge process --profile tabletop --no-classify   # profile, minus classification
```

Profiles live in [`src/aforge_flow/profiles.py`](src/aforge_flow/profiles.py).
To add one, add an entry to `PROFILES`.

## Make targets

Run `make` (or `make help`) to list every target. Variables can be overridden on
the command line, for example `make process IN=art OUT=web IMAGE_FORMAT=png DEVICE=--cpu`.

| Group    | Targets                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Setup    | `help` · `doctor` · `deps-install` · `deps-install-core` · `deps-update` · `direnv` · `protect-main` · `dirs`                        |
| Pipeline | `optimize` · `optimize-lossless` · `remove-bg` · `process` · `tabletop` · `rename` · `classify` · `recommend` · `info` · `benchmark` |
| Web UI   | `ui` · `web` · `web-ui-install` · `web-ui-build` · `web-ui-dev`                                                                      |
| Quality  | `tests` · `tests-gpu` · `coverage` · `lint` · `format` · `format-check` · `typecheck` · `check`                                      |
| Cleanup  | `clean` · `clean-output` · `distclean`                                                                                               |

| Variable                        | Default             | Meaning                                              |
| ------------------------------- | ------------------- | ---------------------------------------------------- |
| `IN` / `OUT`                    | `assets` / `output` | Input and output folders                             |
| `IMAGE_FORMAT` / `VIDEO_FORMAT` | `webp` / `webm`     | Target formats                                       |
| `COMPRESSION`                   | `balanced`          | `lossless`, `high_quality`, `balanced`, `aggressive` |
| `CATEGORY`                      | `general`           | Model routing, see below                             |
| `DEVICE`                        | `--gpu`             | `--gpu` or `--cpu`                                   |
| `ARGS`                          | (empty)             | Extra flags passed through to `aforge` or `pytest`   |
| `EXTRAS`                        | `--all-extras`      | What `deps-install` installs                         |

Recipes are plain commands, so they work under both `sh` and `cmd.exe`.
Filesystem helpers live in [`scripts/make_helpers.py`](scripts/make_helpers.py).

## Background removal models

`--category` picks the model ([`MODEL_BY_CATEGORY`](src/aforge_flow/config.py)):

| Category   | Model               | Typical content                                |
| ---------- | ------------------- | ---------------------------------------------- |
| `humanoid` | `bria-rmbg`         | Humans, elves, goblins, humanoid creatures     |
| `creature` | `bria-rmbg`         | Dragons, beasts, fiends, aberrations           |
| `spirit`   | `bria-rmbg`         | Ghosts, translucent and spectral entities      |
| `item`     | `birefnet-dis`      | Weapons, armor, potions, props: single objects |
| `portrait` | `birefnet-portrait` | Character portraits, faces, hair               |
| `token`    | `bria-rmbg`         | VTT tokens                                     |
| `general`  | `bria-rmbg`         | Mixed or unknown content                       |

Override the routing with `--model <name>`. Other models in `RembgModel` (validated against rembg 2.0.76):
`birefnet-general`, `birefnet-general-lite`, `birefnet-hrsod`, `birefnet-cod`,
`birefnet-massive`, `isnet-general-use`, `isnet-anime`, `u2net`, `u2netp`,
`u2net_human_seg`, `u2net_cloth_seg`, `silueta`, `sam`.

BiRefNet and BRIA run at a fixed 1024² internally. On out-of-memory the GPU session is rebuilt, and
if that still fails the image is processed on CPU. Output quality is the same; it is only slower.

## Classification

Requires the `classify` extra. The classifier uses only the image content.
File names, folders and metadata are never consulted. Each asset gets two labels:

- Asset type: `Portrait` or `Token`
- Creature category, one of the 14 D&D types: Aberration, Beast, Celestial,
  Construct, Dragon, Elemental, Fey, Fiend, Giant, Humanoid, Monstrosity, Ooze, Plant, Undead

Each image is embedded once and matched against prompt-engineered text prototypes
([`taxonomy.py`](src/aforge_flow/taxonomy.py)). The result is written as a sidecar:

```json
{
  "asset_type": "Token",
  "asset_type_raw": "...",
  "asset_type_confidence": 0.91,
  "category": "Undead",
  "confidence": 0.93,
  "category_margin": 0.41,
  "asset_type_margin": 0.37,
  "source": "clip"
}
```

| `--clip-model` | VRAM    | Notes                                               |
| -------------- | ------- | --------------------------------------------------- |
| `siglip2`      | ~1.5 GB | Default. Recommended for fantasy art                |
| `siglip`       | ~1.3 GB | Use when SigLIP 2 weights are unavailable           |
| `clip`         | ~1.2 GB | OpenAI CLIP ViT-L/14 baseline                       |
| `metaclip2`    | 2–4 GB  | More accuracy when you have VRAM to spare           |

A result is accepted when it scores at or above the threshold (default `0.85`) and
beats the runner-up by a clear margin (top-1 minus top-2). Otherwise, `--vlm-fallback`
sends that image to Qwen2.5-VL (`source: "vlm"`).

A batched pre-pass embeds every image once into
`.cache/clip_embeddings.sqlite`. Benchmark with `make benchmark IN=assets` or
`python benchmarks/benchmark_classifier.py eval --labels val.csv --threshold 0.85`.

## Hardware-aware tuning

```bash
aforge recommend -i assets --video-format webm   # inspect + suggest (add --json, --full-pipeline)
aforge optimize -i assets -o output --auto-tune  # apply the suggestion automatically
```

What the recommender takes into account:

- GPU video encoding goes through one NVENC block, so it keeps the worker count low.
- CPU encoders are already multithreaded, so it prefers fewer workers with more threads each.
- Worker counts are capped by free RAM, and by free VRAM when running the full pipeline.

## Web UI (work in progress)

The local browser UI lets you browse and upload assets, set options, watch progress live and
download results. **It is under active development and will be refactored**, so expect changes.

```bash
make ui                  # npm ci + build + serve on http://127.0.0.1:8000
make web                 # API only (serves web/dist if it was built)
make web-ui-dev          # Vite hot-reload dev server (run `make web` alongside)
```

The server binds to `127.0.0.1` by default and has no authentication. It only accepts loopback
`Host`/`Origin` headers, and it serves only media files from inside `--root`.
Details: [`web/README.md`](web/README.md).

## Development

```bash
make deps-install   # full env
make tests          # CPU-only suite: models are faked, nothing is downloaded
make check          # lint + format-check + tests with the 85% coverage gate (what CI runs)
make tests-gpu      # real-model tests (needs CUDA; downloads weights)
```

The test suite drives every CLI workflow end to end (`optimize`, `remove-background`,
`process` + profiles, `caption-rename`, `classify`, `recommend`, `info`, the web API).
Fakes replace the heavy models at the inference boundary (see
[`tests/conftest.py`](tests/conftest.py)).

### CI and releases

| Workflow | What it enforces |
|----------|------------------|
| `ci` | `uv lock --check`, ruff, black and tests with coverage ≥ 85% |
| `security` | [bandit](https://github.com/brunohaf/action-bandit), pip-audit, gitleaks, dependency review |
| `codeql` | CodeQL for Python, TypeScript and workflow files |
| `reviewdog` | ruff and black comments on the PR, plus actionlint and misspell |
| `release` | Tags `vX.Y.Z` when a PR labeled `bump:major`, `bump:minor` or `bump:patch` is merged |

The package version comes from that tag (setuptools-scm). `make protect-main` applies the branch
ruleset, which blocks a merge until every review thread is resolved and every check is green.

### Project structure

```text
aforge-flow/
├── src/aforge_flow/
│   ├── cli.py               # `aforge` entry point (Click)
│   ├── config.py            # Config model, enums, category → model routing
│   ├── profiles.py          # --profile presets
│   ├── pipeline.py          # orchestration, threading, summaries
│   ├── processors/          # background removal, optimization, naming, classification
│   ├── utils/               # hashing, caching, hardware detection, validation
│   └── web/                 # FastAPI backend for the web UI
├── web/                     # React + Vite frontend (WIP)
├── tests/                   # pytest suite (CPU-only by default)
├── benchmarks/              # classifier throughput / accuracy benchmark
├── scripts/                 # Makefile helpers
├── docs/assets/             # logo and images
├── Makefile
├── pyproject.toml / uv.lock
└── .envrc                   # direnv: auto-activate .venv
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved and [SECURITY.md](SECURITY.md)
to report vulnerabilities.

## Known issues

- WebP and JPEG at `--compression lossless` are still lossy (quality 95). For pixel-exact output, use PNG.

## License

[MIT](LICENSE) © Bruno Fonseca
