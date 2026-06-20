<h1 align="center">
  <br>
  <b>AForge-Flow</b>
  <br>
</h1>

<h4 align="center">
GPU-accelerated toolkit for asset segmentation, optimization, conversion, and preparation for modern web and digital media workflows.
</h4>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/cuda-optional-green.svg" alt="CUDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey.svg" alt="Platform"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#hardware-aware-tuning">Auto-Tuning</a> •
  <a href="#creature-classification">Creature Classification</a> •
  <a href="#asset-categories">Asset Categories</a> •
  <a href="#background-removal-models">Models</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#development">Development</a>
</p>

> AForge, transform, and optimize digital assets for virtual tabletops, web publishing, content libraries, and creative workflows.

---

## Key Features

### AI-Powered Background Removal

- GPU-accelerated image segmentation using Rembg and ONNX Runtime
- Support for BiRefNet, ISNet, U²-Net, and specialized segmentation models
- Automatic model selection based on asset category
- High-quality alpha extraction for transparent PNG generation
- Optimized for fantasy artwork, portraits, creatures, and RPG assets

### Zero-Shot Creature Classification

- Classifies fantasy artwork into the 14 D&D creature types **from image content alone** — file names, folders, and metadata are ignored
- Detects asset type (`Portrait` / `Token`) in the same pass
- Built on a CLIP-family vision encoder (SigLIP 2 by default) with prompt-engineered text prototypes — no expensive VLM inference on the common case
- Persistent embedding cache: every picture is embedded once, ever; re-runs do zero GPU work
- Optional VLM fallback (Qwen2.5-VL) for the low-confidence tail only

### Asset Optimization

- WebP conversion and optimization
- Image transcoding and compression
- Directory structure preservation
- Duplicate detection and skip-existing support
- Integrity verification

### Batch Processing

- Large-scale asset library processing
- Progress tracking and reporting
- Structured logging
- Hardware-aware tuning: `aforge recommend` inspects the machine + workload, and `--auto-tune` applies safe `--workers`/`--gpu` settings automatically
- Resumable processing

---

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg
- NVIDIA GPU + CUDA drivers (optional — enables GPU acceleration; CPU works too)

Run `aforge recommend` after install to see what your machine supports and the suggested settings.

### Conda Installation (Recommended)

Install Miniconda:

```bash
wget -qO ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash ~/miniconda.sh -b -p "$HOME/miniconda3"
```

Configure Conda:

```bash
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda init bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

Create environment:

```bash
git clone <repository-url>
cd aforge-flow

conda env create -f environment.yml
conda activate aforge-flow

pip install -e .
```

To enable **creature classification** (CLIP-family + optional VLM fallback),
install the extra into the **conda env** (not your global Python):

```bash
conda activate aforge-flow
pip install -e ".[classify]"
```

This pulls `transformers>=4.49` (required for SigLIP 2), `accelerate`, and
`bitsandbytes` (Linux, for the optional 4-bit VLM fallback). The SigLIP 2
weights (~1.5 GB) download to `.cache/` on first run.

### Manual Installation

```bash
git clone <repository-url>
cd aforge-flow

python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install "rembg[gpu]"
```

---

## Quick Start

### Background Removal

```bash
aforge process \
  --input assets/rembg/portraits \
  --output output/portraits \
  --background-removal \
  --no-media-optimization \
  --category portrait \
  --gpu
```

### Media Optimization

```bash
aforge optimize-media \
  --input assets/webmp \
  --output output/webmp \
  --image-format webp \
  --video-format webm \
  --compression balanced
```

WebM video is encoded with GPU AV1 (NVENC) when a capable GPU is available, and
falls back to CPU VP9 otherwise. Audio in WebM is encoded as Opus.

### Hardware-Aware Tuning

Let `aforge` pick safe, performant settings instead of guessing at `--workers`
and `--gpu/--cpu`. The `recommend` command inspects your CPU, RAM, GPU/VRAM, and
NVENC support, scans the input directory (image vs video mix and sizes), and
prints a suggested command — without risking out-of-memory or encoder-session
exhaustion:

```bash
aforge recommend --input assets --video-format webm
# add --full-pipeline to tune for `process` (background removal + classification)
# add --json for machine-readable output
```

To apply the same logic automatically at run time, pass `--auto-tune` to
`optimize-media` or `process`. It overrides `--workers` and `--gpu/--cpu` with
the recommended values and logs what it chose:

```bash
aforge optimize-media --input assets --output output --video-format webm --auto-tune
```

### Creature Classification

Classify a directory of images into the D&D creature taxonomy from image
content alone (requires the `[classify]` extra):

```bash
aforge classify \
  --input assets \
  --output output \
  --clip-model siglip2 \
  --confidence-threshold 0.85 \
  --gpu
```

Add `--vlm-fallback` to resolve low-confidence cases with Qwen2.5-VL. Or run
it as a stage of the full pipeline with `--classification`:

```bash
aforge process \
  --input assets --output output \
  --classification --clip-model siglip2 \
  --background-removal --media-optimization --gpu
```

### Full Processing Pipeline

```bash
aforge process \
  --input assets \
  --output output \
  --background-removal \
  --media-optimization \
  --category general \
  --image-format webp \
  --video-format webm \
  --compression balanced \
  --gpu
```

### System Information

aforge can automatically select processing strategies based on asset type.

| Category | Description |
|-----------|-------------|
| `humanoid` | Humans, elves, dwarves, goblins, orcs, zombies, and humanoid creatures |
| `creature` | Dragons, beasts, fiends, aberrations, fey, monsters, and non-humanoid entities |
| `spirit` | Ghosts, spectral creatures, translucent entities, magical manifestations |
| `item` | Weapons, armor, scrolls, potions, props, treasure, and equipment |
| `portrait` | Character portraits, NPC artwork, and journal illustrations |
| `token` | Virtual tabletop tokens and token-ready artwork |
| `general` | Mixed or unknown content |

---

## Creature Classification

Zero-shot classification from **image content only** (file names, folders, and
metadata are treated as unreliable and never consulted). Two levels per asset:

- **Asset type** — normalized to `Portrait` or `Token`
- **Creature category** — one of the 14 official D&D types: Aberration, Beast,
  Celestial, Construct, Dragon, Elemental, Fey, Fiend, Giant, Humanoid,
  Monstrosity, Ooze, Plant, Undead

Each image is embedded once by a CLIP-family encoder and matched against
prompt-engineered text prototypes (rich descriptions, not bare labels — see
`src/aforge_flow/taxonomy.py`). Results are returned in the
`ProcessorResult` metadata and written as a `<image>.classification.json`
sidecar:

```json
{ "asset_type": "Undead", "category": "Undead", "confidence": 0.93, "model": "siglip2", "source": "clip" }
```

### Encoders (`--clip-model`)

| Model | VRAM | Notes |
|-------|------|-------|
| `siglip2` | ~1.5 GB | **Default.** Best zero-shot accuracy for fantasy art |
| `siglip`  | ~1.3 GB | Strong; use if SigLIP 2 weights are unavailable |
| `clip`    | ~1.2 GB | OpenAI CLIP ViT-L/14 baseline |
| `metaclip2` | 2–4 GB | Accuracy headroom when more VRAM is available |

### Confidence workflow

`confidence ≥ threshold` (default `0.85`) and a sufficient top1−top2 margin →
the CLIP result is auto-accepted (`source: "clip"`). Otherwise, if
`--vlm-fallback` is set, Qwen2.5-VL classifies that single image
(`source: "vlm"`). The VLM only ever runs on the uncertain tail, so its cost
is amortized away.

### Throughput

A one-time batched embedding pre-pass runs before the per-file stage, then all
embeddings are cached in `.cache/clip_embeddings.sqlite` (float16, ~13 MB for
6,000 assets). Re-runs — including after tweaking thresholds or prompts — do
**zero** GPU work. Benchmark with:

```bash
python scripts/benchmark_classifier.py throughput -i assets --batch-size 64
python scripts/benchmark_classifier.py eval --labels val.csv --threshold 0.85
```

---

## Background Removal Models

### Recommended

| Model | Best For |
|---------|---------|
| `birefnet-general` | Humanoids, creatures, spirits, tokens, general-purpose extraction |
| `birefnet-portrait` | Portraits, faces, hair, headshots |
| `birefnet-dis` | Items, products, props, isolated objects |

### Additional Models

| Model | Description |
|---------|---------|
| `isnet-general-use` | High-quality general-purpose segmentation |
| `isnet-anime` | Anime and illustrated artwork |
| `u2net` | Legacy high-quality segmentation |
| `u2netp` | Lightweight U²-Net |
| `u2net_human_seg` | Human body segmentation |
| `u2net_anime` | Anime-focused U²-Net |
| `u2net_cloth_seg` | Clothing segmentation |
| `silueta` | Lightweight human silhouette extraction |
| `salient_object_amodal` | Salient object extraction with occlusion recovery |

### Suggested Configuration

```python
MODEL_BY_CATEGORY = {
    AssetCategory.HUMANOID: RembgModel.BIREFNET_GENERAL,
    AssetCategory.CREATURE: RembgModel.BIREFNET_GENERAL,
    AssetCategory.SPIRIT: RembgModel.BIREFNET_GENERAL,
    AssetCategory.ITEM: RembgModel.BIREFNET_DIS,
    AssetCategory.PORTRAIT: RembgModel.BIREFNET_PORTRAIT,
    AssetCategory.TOKEN: RembgModel.BIREFNET_GENERAL,
    AssetCategory.GENERAL: RembgModel.BIREFNET_GENERAL,
}
```

---

## Project Structure

```text
aforge-flow/
├── assets
│
├── output/
│
├── scripts/
│
├── src/
│   └── aforge_flow/
│       ├── processors/
│       ├── utils/
│       ├── cli.py
│       ├── config.py
│       └── pipeline.py
│
├── tests/
│
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## Performance Notes

### Auto-tuning

Rather than hand-tuning `--workers` and `--gpu/--cpu`, let the tool size them for
your hardware and workload:

```bash
aforge recommend --input assets --video-format webm   # inspect + suggest
aforge optimize-media --input assets --output output --auto-tune   # apply automatically
```

The recommender accounts for the key constraints:

- **GPU video encoding** uses a single shared NVENC block, so throughput does not
  scale with more workers — it keeps the worker count low to avoid contention.
- **CPU video encoding** (VP9/libx264) is already multithreaded, so it favors
  fewer workers with more threads each over many contending workers.
- **Memory** — worker counts are capped against available system RAM (and VRAM for
  the full pipeline) to avoid out-of-memory failures.

### Background Removal

BiRefNet models offer the highest extraction quality but require significantly more memory than ISNet or U²-Net.

For large images:

- Resize extremely large images before processing
- Use `--auto-tune` (or `aforge recommend`) to size workers to available memory

---

## License
MIT License

Copyright (c) 2026 Bruno Fonseca

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Built for modern asset libraries, virtual tabletops, and digital publishing workflows.
