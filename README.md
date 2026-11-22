<h1 align="center">
  <br>
  <b>aforge-flow</b>
  <br>
</h1>

<h4 align="center">
GPU-accelerated toolkit for asset segmentation, optimization, conversion, and preparation for modern web and digital-media workflows.
</h4>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/cuda-optional-green.svg" alt="CUDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/packaging-uv-purple.svg" alt="uv"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey.svg" alt="Platform"></a>
</p>

> AForge, transform, and optimize digital assets for virtual tabletops, web publishing, content libraries, and creative workflows — from a CLI **or** a local web UI.

---

## What's in this repo

A **uv monorepo** with three independent projects:

| Project | What it is | Run it with |
|---------|------------|-------------|
| [`aforge_flow/`](aforge_flow/README.md) | The **pipeline + CLI** — background removal, media optimization, creature classification, caption-rename. Standalone; no web stack required. | `aforge …` |
| [`aforge_flow_api/`](aforge_flow_api/README.md) | A **FastAPI** service wrapping the pipeline: browse/upload, run jobs, stream live progress (SSE), download results. | `aforge-api` |
| [`aforge_flow_ui/`](aforge_flow_ui/README.md) | A **React + Vite** single-page app (dark/light, pt-BR / en-US) that drives the API. | `npm run dev` |

```
┌──────────────────┐   HTTP/SSE   ┌──────────────────┐   imports   ┌────────────────────┐
│  aforge_flow_ui  │ ───────────▶ │  aforge_flow_api │ ──────────▶ │     aforge_flow     │
│   (React/Vite)   │ ◀─────────── │    (FastAPI)     │             │ (pipeline + `aforge`)│
└──────────────────┘              └──────────────────┘             └────────────────────┘
```

The dependency arrow only points one way: the API depends on the pipeline, never the reverse — so you can run **just the CLI** with none of the web stack installed.

---

## The pipeline, in one breath

For each asset the pipeline can, in order: optionally **caption-rename** → optionally **classify** (creature taxonomy) → **remove background** → **optimize/transcode** (images → WebP, video → WebM). Background removal and optimization are on by default in `process`; classification and caption-rename are opt-in. See [`aforge_flow/README.md`](aforge_flow/README.md) for the full tour.

---

## Quick start (uv)

This repo uses **[uv](https://docs.astral.sh/uv/)** instead of conda/pip. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Just the CLI (no web stack)

```bash
uv sync --package aforge-flow        # installs only the pipeline + its deps into .venv
uv run aforge recommend -i assets    # inspect hardware + suggest settings
uv run aforge process -i assets -o output --gpu
```

### Background removal at scale (the big batch)

```bash
# 1) Confirm the GPU is actually active before committing to a long run.
#    Prints the *active* ONNXRuntime providers + ms/image + projected 20k runtime.
uv run aforge remove-background -i assets --benchmark 50

# 2) Run it. Re-running resumes — finished *_nobg.png outputs are skipped (do NOT
#    pass --no-skip-existing unless you intend to redo completed work).
uv run aforge remove-background -i assets -o output --workers 3 --png-compress-level 1
```

Useful flags (`remove-background` and `process`): `--provider auto|tensorrt|cuda|cpu`
(default `auto` → TensorRT→CUDA→CPU), `--fp16/--no-fp16`, `--png-compress-level 0-9`
(zlib speed/size only — always lossless), `--workers N`, `--no-dedup`,
`--ignore-dir NAME` (repeatable, matched at any depth), and `--ignore-extension EXT`
(repeatable; `remove-background` excludes video extensions by default). Run
`uv run aforge <command> -h` for the full list.

### The web UI (API + frontend)

```bash
uv sync                              # whole workspace (pipeline + API)

# 1) build the SPA, served by the API at /
cd aforge_flow_ui && npm install && npm run build && cd ..

# 2) launch the server (sandboxed to --root)
uv run aforge-api --root .           # open http://127.0.0.1:8000
```

For **hot-reload development**, run the API and the Vite dev server side by side:

```bash
uv run aforge-api --root .           # terminal 1 → :8000
cd aforge_flow_ui && npm run dev           # terminal 2 → :5173 (proxies /api → :8000)
```

> **GPU / CUDA:** on Linux the default PyPI `torch` wheel already bundles CUDA, so `uv sync` gets GPU support out of the box. To pin a specific CUDA/CPU build, see the commented `[tool.uv.index]` block in [`aforge_flow/pyproject.toml`](aforge_flow/pyproject.toml).
>
> **Background removal acceleration (rembg / ONNXRuntime):** rembg runs on
> `onnxruntime-gpu`. The default `--provider auto` prepends the **TensorRT** EP
> (FP16, engine cached under `<cache_dir>/trt_engines`) and falls back to CUDA, then
> CPU — so it works without TensorRT installed and gets faster once it is. For the
> TensorRT fast path under **WSL2**: install a `tensorrt` build matching your
> `onnxruntime-gpu` (e.g. ORT 1.27 → TensorRT 10.x), make sure the bundled CUDA libs
> are on `LD_LIBRARY_PATH` (the `nvidia/*/lib` dirs inside the venv), and verify with
> `python -c "import onnxruntime as o; print(o.get_available_providers())"`. If a run
> is unexpectedly slow, the startup log line `session_initialized … active=[…]` tells
> you which provider actually loaded — if it shows only `CPUExecutionProvider`, fix
> the environment before anything else.

---

## Pluggable taxonomy (not just D&D)

The classifier and the caption-rename name hints are driven by a **taxonomy profile** — a small JSON file under [`aforge_flow/src/aforge_flow/core/profiles/`](aforge_flow/src/aforge_flow/core/profiles/).

**Why are there D&D monster names in there?** The `dnd` profile ships a curated *name bank* (`mind-flayer`, `beholder`, …). It's fed to the Qwen caption backend as a **soft hint** so generated filenames prefer canonical monster-manual names instead of inventing vague ones (e.g. `mind-flayer` rather than `tentacle-face-guy`). It only biases naming — Qwen may still coin a short name when nothing fits.

**Switching domains** is one flag — D&D is just the default, not a hard-coded assumption:

```bash
uv run aforge classify -i assets --taxonomy mundane           # everyday photos
uv run aforge rename   -i assets -o out --taxonomy call_of_cthulhu
uv run aforge process  -i assets -o out --classify --taxonomy ./my_domain.json
```

Built-ins: `dnd` (default), `mundane` (people/animals/vehicles/landscapes…), `call_of_cthulhu` (Mythos). Add your own by dropping a `<name>.json` (optionally a `<name>.names.json`) in the profiles folder, or point `--taxonomy` at any JSON path.

---

## Project docs

- **[aforge_flow/README.md](aforge_flow/README.md)** — pipeline & `aforge` CLI: every command, all flags, models, classification, taxonomy profiles, examples.
- **[aforge_flow_api/README.md](aforge_flow_api/README.md)** — FastAPI service: endpoints, SSE, the `--root` sandbox, `AFORGE_FRONTEND_DIST`, curl recipes.
- **[aforge_flow_ui/README.md](aforge_flow_ui/README.md)** — React/Vite app: dev vs prod, the Vite proxy, i18n and theming.

---

## Development

```bash
uv sync                  # full workspace + dev tools (pytest, ruff, black, mypy)
uv run pytest            # runs aforge_flow/tests + aforge_flow_api/tests (CPU only)
uv run ruff check .
uv run black .
```

**uv is the only supported path.** Non-Python system deps are not an env-manager
concern: install `ffmpeg` (media optimization) and the CUDA driver / TensorRT
runtime via your OS — under WSL2 that's `apt` plus the matching `tensorrt` wheel.

---

## License

MIT License — Copyright (c) 2026 Bruno Fonseca. Full text in [LICENSE](LICENSE); reproduced below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

Built for modern asset libraries, virtual tabletops, and digital publishing workflows.
