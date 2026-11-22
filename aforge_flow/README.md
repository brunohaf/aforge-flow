# aforge-flow — pipeline & CLI

GPU-accelerated batch pipeline for tabletop/VTT artwork: strip backgrounds,
transcode images and video to web-friendly formats, classify creatures into a
pluggable taxonomy, and rename assets from AI captions. Ships the `aforge` CLI.

This is the **standalone** project — it has no dependency on the API or
frontend. For the web UI see [`../aforge_flow_api`](../aforge_flow_api/README.md)
and [`../aforge_flow_ui`](../aforge_flow_ui/README.md); for the monorepo overview see the
[root README](../README.md).

## Install / run (uv)

```bash
uv sync --package aforge-flow      # just this project into .venv (no FastAPI)
uv run aforge --help
uv run aforge info                 # version, PyTorch, CUDA/GPU details
```

Optional extras:

```bash
# Creature classification (CLIP-family + optional VLM fallback)
uv pip install -e ".[classify]"    # transformers>=4.49 (SigLIP 2), accelerate, bitsandbytes
# Video transcoding helpers
uv pip install -e ".[video]"
```

> **GPU/CUDA:** the default PyPI `torch` wheel bundles CUDA on Linux. To pin a
> specific build, uncomment the `[tool.uv.index]` block in `pyproject.toml`.
> FFmpeg must be on `PATH` for media optimization.

## Commands

| Command | Purpose | Aliases |
|---------|---------|---------|
| `process` | Full pipeline: background removal + media optimization (`--classify` / `--caption-rename` opt-in) | `go`, `pipe` |
| `remove-background` | Strip image backgrounds with rembg | |
| `optimize` | Transcode images → WebP, video → WebM | `optimize-media` |
| `caption-rename` | Rename images from AI captions; **copy videos through as-is** | `rename` |
| `classify` | Tag images with a creature taxonomy (CLIP/VLM) | |
| `recommend` | Suggest `--workers` / `--gpu` for this machine + workload | |
| `info` | Version, PyTorch, CUDA/GPU details | |

Run `uv run aforge <command> -h` for every flag.

### Examples

```bash
# Background removal only, tuned model per category
uv run aforge process -i assets/portraits -o output \
  --background-removal --no-media-optimization --category portrait --gpu

# Media optimization, hardware-auto-tuned
uv run aforge optimize -i assets -o output \
  --image-format webp --video-format webm --compression balanced --auto-tune

# Classify into the creature taxonomy (image content only)
uv run aforge classify -i assets --clip-model siglip2 --confidence-threshold 0.85 --vlm-fallback

# Full pipeline with classification
uv run aforge go -i assets -o output --classify --background-removal --media-optimization --gpu
```

## Hardware-aware tuning

`recommend` inspects CPU/RAM/GPU/VRAM + NVENC, scans the input (image vs video
mix and sizes), and prints a safe suggested command. `--auto-tune` applies the
same logic at run time, overriding `--workers` and `--gpu/--cpu`:

```bash
uv run aforge recommend -i assets --video-format webm        # inspect + suggest (--json for machine-readable)
uv run aforge optimize  -i assets -o output --auto-tune      # apply automatically
```

GPU video encoding shares a single NVENC block (more workers ≠ more throughput),
CPU VP9/x264 is already multithreaded, and worker counts are capped against
available RAM/VRAM to avoid OOM.

## Pluggable taxonomy (`--taxonomy`)

Classification labels/prompts **and** the caption name hints come from a
**profile** (JSON under [`src/aforge_flow/core/profiles/`](src/aforge_flow/core/profiles/)).
The module [`core/taxonomy.py`](src/aforge_flow/core/taxonomy.py) is a loader/registry over that data.

| Profile | Domain |
|---------|--------|
| `dnd` (default) | Dungeons & Dragons — 14 creature types + monster name bank |
| `mundane` | Everyday photos — person, animal, vehicle, building, landscape… |
| `call_of_cthulhu` | Lovecraftian Mythos — deep-one, cultist, shoggoth, mi-go… |

```bash
uv run aforge classify -i assets --taxonomy mundane
uv run aforge process  -i assets -o out --classify --taxonomy ./my_domain.json   # external file
```

**A profile file** carries category labels + CLIP prompts, the asset-type
formats + normalization, and (optionally) a `caption_instruction` (Qwen),
`caption_prefix` (BLIP), and a `name_bank`. The `dnd` profile reproduces the
original hard-coded taxonomy exactly, so the default behavior is unchanged.

### Why D&D monster names?

The `dnd` name bank (`dnd.names.json`) is a soft hint to the Qwen caption
backend so renamed files prefer canonical names (`mind-flayer`) over invented
phrasings. Profiles without a name bank (e.g. `mundane`) caption freely. The
default caption instruction is tuned to produce **specific, evocative** names
and to avoid generic descriptors like `armored-fighter`.

## Creature classification

Zero-shot, from **image content only** (file names/folders are ignored). Two
levels per asset — asset type (`Portrait`/`Token`) and category (e.g. one of the
14 D&D types). Each image is embedded once by a CLIP-family encoder and matched
against prompt-engineered prototypes; results land in `ProcessorResult.metadata`
and a `<image>.classification.json` sidecar.

| `--clip-model` | VRAM | Notes |
|----------------|------|-------|
| `siglip2` | ~1.5 GB | **Default.** Best zero-shot accuracy for fantasy art |
| `siglip`  | ~1.3 GB | Strong alternative |
| `clip`    | ~1.2 GB | OpenAI CLIP ViT-L/14 baseline |
| `metaclip2` | 2–4 GB | Accuracy headroom with more VRAM |

`confidence ≥ threshold` (default `0.85`) with a sufficient top1−top2 margin →
the CLIP result is auto-accepted; otherwise `--vlm-fallback` consults Qwen2.5-VL
on that single image. Embeddings are cached in `.cache/clip_embeddings.sqlite`,
so re-runs do zero GPU work.

## Background removal models

| Model | Best for |
|-------|----------|
| `bria-rmbg` | **Default** — humanoids, creatures, tokens, complex edges |
| `birefnet-general` | Creatures, fur, hair, wings, transparency |
| `birefnet-portrait` | Portraits, faces, hair |
| `birefnet-dis` | Items, products, isolated objects |
| `isnet-general-use`, `u2net`, `u2netp`, `silueta`, … | See `aforge remove-background -h` |

The model is auto-selected per `--category` (see `MODEL_BY_CATEGORY` in
[`core/config.py`](src/aforge_flow/core/config.py)); override with `--model`.

### Performance & execution provider

Background removal is GPU-bound and serialized (one inference at a time), so speed
comes from the runtime, not from parallelism. The same model and edges are kept.

| Flag | Default | Effect |
|------|---------|--------|
| `--provider` | `auto` | ONNXRuntime EP: `auto` tries TensorRT→CUDA→CPU; or force `tensorrt`/`cuda`/`cpu` |
| `--fp16/--no-fp16` | `--fp16` | FP16 in the TensorRT EP — ~1.5–2× faster, half the VRAM, visually identical on an alpha mask. `--no-fp16` for bit-stable masks |
| `--png-compress-level` | `6` | PNG zlib level (0–9). Lower = faster + larger, **always lossless**. Use `1` for big batches |
| `--workers` | `1` | Overlaps image decode / PNG encode with GPU inference; `3` is a good batch value |
| `--no-dedup` | dedup on | Skip the perceptual-hash dedup pass (saves CPU on large batches) |
| `--benchmark N` | off | Time N images, print the **active** providers + ms/image + projected 20k runtime, then exit |

`--provider auto` works even without TensorRT installed (the EP is skipped). The
startup log line `session_initialized … active=[…]` reports which provider actually
loaded — the reliable way to catch a silent CPU fallback. With TensorRT the engine
is built once and cached under `<cache_dir>/trt_engines`; the first run does a
one-time warmup build. See the root README for the WSL2 TensorRT setup.

> **Resolution & losslessness:** output is always full-resolution (the alpha is
> composited onto the original RGB, which is copied verbatim) and the PNG is lossless
> at any compress level. `optimize` is the only lossy step, dialed by `--compression`
> — and `--compression lossless` is now genuinely lossless (WebP `lossless=True`,
> video `-lossless 1` / `-qp 0`; lossless WebM falls back to CPU VP9 since `av1_nvenc`
> has no lossless mode).

### Discovery filters

Both `process` and `remove-background` accept `--ignore-dir NAME` (repeatable;
skips any directory of that name at any depth) and `--ignore-extension EXT`
(repeatable). `remove-background` defaults `--ignore-extension` to the video
extensions (it only handles images); `process` keeps videos by default since it
transcodes them.

## Project layout

```
aforge_flow/
├── pyproject.toml            # name = aforge-flow; script aforge = aforge_flow.cli:main
├── src/aforge_flow/
│   ├── cli/                  # command group + one module per command
│   │   ├── main.py  _shared.py
│   │   └── commands/{process,remove_background,optimize,classify,caption_rename,recommend,info}.py
│   ├── core/                 # config.py · pipeline.py · taxonomy.py · profiles/*.json
│   ├── processors/           # background_removal · media_optimization · creature_classification · caption_naming
│   ├── utils/                # validators · recommend · deduplication · embedding_cache · logger
│   └── scripts/benchmark_classifier.py
└── tests/                    # CPU-only unit tests (pytest -m "not gpu")
```

## Tests

```bash
uv run pytest aforge_flow/tests -m "not gpu"
```
