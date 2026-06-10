<h1 align="center">
  <br>
  <b>AssetFoundry</b>
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
  <a href="#asset-categories">Asset Categories</a> •
  <a href="#background-removal-models">Models</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#development">Development</a>
</p>

> Forge, transform, and optimize digital assets for virtual tabletops, web publishing, content libraries, and creative workflows.

---

## Key Features

### AI-Powered Background Removal

- GPU-accelerated image segmentation using Rembg and ONNX Runtime
- Support for BiRefNet, ISNet, U²-Net, and specialized segmentation models
- Automatic model selection based on asset category
- High-quality alpha extraction for transparent PNG generation
- Optimized for fantasy artwork, portraits, creatures, and RPG assets

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
- Automatic GPU / CPU execution
- Resumable processing

---

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg
- NVIDIA GPU recommended
- CUDA-compatible drivers (optional)

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
cd assetfoundry

conda env create -f environment.yml
conda activate assetfoundry

pip install -e .
```

### Manual Installation

```bash
git clone <repository-url>
cd assetfoundry

python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install "rembg[gpu]"
```

---

## Quick Start

### Background Removal

```bash
assetfoundry process \
  --input assets/rembg/portraits \
  --output output/portraits \
  --background-removal \
  --no-media-optimization \
  --category portrait \
  --gpu
```

### Media Optimization

```bash
assetfoundry optimize-media \
  --input assets/webmp \
  --output output/webmp \
  --output-format webp \
  --compression balanced
```

### Full Processing Pipeline

```bash
assetfoundry process \
  --input assets \
  --output output \
  --background-removal \
  --media-optimization \
  --category general \
  --output-format webp \
  --compression balanced \
  --gpu
```

### System Information

AssetFoundry can automatically select processing strategies based on asset type.

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
assetfoundry/
├── assets
│
├── output/
│
├── scripts/
│
├── src/
│   └── assetfoundry/
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

### GPU

Recommended:

- RTX 3060+
- RTX 4070 ideal
- CUDA-enabled ONNX Runtime

### Background Removal

BiRefNet models offer the highest extraction quality but require significantly more memory than ISNet or U²-Net.

For large images:

- Limit worker count
- Resize extremely large images before processing
- Monitor VRAM using:

```bash
nvidia-smi
```

### Recommended Worker Counts

| GPU | Workers |
|------|---------|
| RTX 3060 | 1 |
| RTX 4070 | 1-2 |
| CPU Only | 1 |

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