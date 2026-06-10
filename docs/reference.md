# Foundry_Asset

GPU-accelerated asset optimization tool for batch background removal, image enhancement, and media transcoding. Designed for fantasy art, game assets, and digital content creation workflows.

## Features

- GPU-Accelerated Background Removal using rembg with NVIDIA CUDA support
- Specialized Models for humanoids, creatures, and items (fantasy-optimized)
- Media Optimization with WebM/WebP transcoding and compression
- Batch Processing with parallel execution and progress tracking
- Duplicate Detection to prevent reprocessing identical files
- High-Quality Output with integrity verification and lossless options
- Optimized for NVIDIA 4070 and modern GPUs
- Multi-threaded batch processing
- Conditional GPU/CPU fallback
- Configurable compression levels (lossless to aggressive)
- Structured logging and error handling
- Directory structure preservation in output
- Comprehensive statistics and reporting
- Skip existing files to resume interrupted runs
- Support for multiple asset categories

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 4070 or equivalent)
- FFmpeg installed system-wide

### Setup with Conda (Recommended)

#### Install Conda

```bash
wget -qO ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash ~/miniconda.sh -b -p "$HOME/miniconda3" && \
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)" && \
conda init bash && \
conda config --add channels conda-forge && \
conda config --set channel_priority strict && \
conda install -n base -y mamba && \
rm ~/miniconda.sh
source ~/.bashrc
```

#### Create the environment
Create isolated environment with GPU support:

```bash
# Clone or navigate to project
cd asset-optimizer

# Create conda environment
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml

# Activate environment
conda activate asset-optimizer

# Install package in development mode
pip install -e .
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install "rembg[gpu]>=0.0.52"

# Install package
pip install -e .
```

## Quick Start

### 1. Remove Backgrounds

Process images to remove backgrounds (independent of media optimization):

```bash
asset-optimizer remove-background \
  --input assets/rembg/portraits/humanoids \
  --output output/humanoids_nobg \
  --category humanoid \
  --model u2net_human_seg \
  --gpu
```

### 2. Optimize Media

Convert and compress images/videos to WebP/WebM:

```bash
asset-optimizer optimize-media \
  --input assets/webmp/portraits \
  --output output/portraits_optimized \
  --output-format webp \
  --compression balanced
```

### 3. Full Pipeline

Run complete optimization (background removal + media optimization):

```bash
asset-optimizer process \
  --input assets/ \
  --output output/ \
  --background-removal \
  --media-optimization \
  --category general \
  --output-format webp \
  --compression balanced \
  --workers 1 \
  --gpu
```

### 4. System Information

Check GPU availability and configuration:

```bash
asset-optimizer info
```

## Command Reference

### `process` - Full Pipeline

Process assets through complete pipeline:

```bash
asset-optimizer process [OPTIONS]

Options:
  -i, --input PATH              Input directory [default: assets]
  -o, --output PATH             Output directory [default: output]
  --background-removal          Enable background removal [default: True]
  --no-background-removal       Disable background removal
  --media-optimization          Enable media optimization [default: True]
  --no-media-optimization       Disable media optimization
  --category [humanoid|creature|item|general]  Asset category [default: general]
  --model [u2net|u2net_human_seg|isnet-general-use|siluette]  Model choice [default: isnet-general-use]
  --output-format [png|jpeg|webp|webm|mp4|mkv]  Output format [default: webp]
  --compression [lossless|high_quality|balanced|aggressive]  Compression level [default: balanced]
  --gpu / --cpu                Use GPU acceleration [default: gpu]
  --workers INTEGER             Parallel workers [default: 1]
  --skip-existing               Skip processed files [default: True]
  --no-skip-existing            Reprocess all files
  --log-level [DEBUG|INFO|WARNING|ERROR]  Log level [default: INFO]
```

### `remove-background` - Background Removal Only

Remove backgrounds independently (rembg only):

```bash
asset-optimizer remove-background [OPTIONS]

Options:
  -i, --input PATH              Input directory [default: assets]
  -o, --output PATH             Output directory [default: output]
  --category [humanoid|creature|item|general]  Category for model selection
  --model [u2net|u2net_human_seg|isnet-general-use|siluette]  Model choice
  --gpu / --cpu                 Use GPU acceleration [default: gpu]
  --workers INTEGER             Parallel workers [default: 1]
  --skip-existing               Skip processed files [default: True]
  --log-level [DEBUG|INFO|WARNING|ERROR]  Log level [default: INFO]
```

### `optimize-media` - Media Optimization Only

Optimize media independently (transcoding only):

```bash
asset-optimizer optimize-media [OPTIONS]

Options:
  -i, --input PATH              Input directory [default: assets]
  -o, --output PATH             Output directory [default: output]
  --output-format [png|jpeg|webp|webm|mp4|mkv]  Output format [default: webp]
  --compression [lossless|high_quality|balanced|aggressive]  Compression level
  --workers INTEGER             Parallel workers [default: 1]
  --skip-existing               Skip processed files [default: True]
  --log-level [DEBUG|INFO|WARNING|ERROR]  Log level [default: INFO]
```

## Configuration

### Asset Categories

Models are automatically selected based on asset category:

| Category | Model | Use Case |
|----------|-------|----------|
| `humanoid` | u2net_human_seg | Human figures, character portraits |
| `creature` | isnet-general-use | Fantasy creatures, monsters, animals |
| `item` | isnet-general-use | Equipment, props, magical items |
| `general` | isnet-general-use | Default for mixed content |

### Compression Levels

| Level | Quality | Use Case |
|-------|---------|----------|
| `lossless` | 95 | Archival, no loss tolerance |
| `high_quality` | 85 | Professional output, minimal loss |
| `balanced` | 75 | Standard use, good quality/size trade-off |
| `aggressive` | 60 | Space-critical, acceptable loss |

### Environment Variables

Create `.env` for configuration:

```bash
# GPU Device
CUDA_DEVICE=cuda:0

# Processing
MAX_WORKERS=1
BATCH_SIZE=1

# Logging
LOG_LEVEL=INFO
```

## Project Structure

```
asset-optimizer/
├── src/assetfoundry/          # Main package
│   ├── __init__.py               # Package entry point
│   ├── config.py                 # Configuration management
│   ├── cli.py                    # CLI interface
│   ├── pipeline.py               # Processing pipeline
│   ├── processors/               # Asset processors
│   │   ├── base.py               # Abstract processor
│   │   ├── background_removal.py # rembg integration
│   │   └── media_optimization.py # Media transcoding
│   └── utils/                    # Utilities
│       ├── logger.py             # Structured logging
│       ├── deduplication.py      # Duplicate detection
│       └── validators.py         # Input validation
├── tests/                        # Test suite
├── assets/                       # Input assets (preserved)
│   ├── rembg/                    # Raw images for background removal
│   │   ├── CoS/
│   │   │   ├── humanoids/
│   │   │   ├── creatures/
│   │   │   └── items/
│   │   └── portraits/
│   └── webmp/                    # Media for transcoding
├── output/                       # Output directory (auto-created)
├── pyproject.toml               # Python packaging (modern)
├── environment.yml              # Conda environment specification
├── README.md                    # This file
└── .gitignore                   # Git configuration
```

## Advanced Usage

### Custom Processing Script

```python
from pathlib import Path
from assetfoundry import Config, AssetCategory
from assetfoundry.pipeline import AssetPipeline

# Configure
config = Config(
    input_dir=Path("assets/creatures"),
    output_dir=Path("output/creatures"),
    rembg_category=AssetCategory.CREATURE,
    use_gpu=True
)

# Run
pipeline = AssetPipeline(config)
pipeline.setup()
pipeline.process_directory()
pipeline.print_summary()
```

### Monitoring Processing

Logs are structured with built-in metrics:

```bash
# Run with debug logging
asset-optimizer process --input assets --log-level DEBUG

# Monitor specific processor
asset-optimizer process --input assets --log-level INFO 2>&1 | grep "background_removal"
```

## Performance Tips

### GPU Optimization
1. NVIDIA 4070: Use batch_size=2-4 for optimal throughput
2. Memory: Monitor VRAM with `nvidia-smi` during processing
3. Driver: Ensure NVIDIA drivers are up-to-date
4. CUDA: Verify CUDA 11.8+ compatibility

### Processing Speed
- Parallel Workers: Set to 1.5-2x number of CPU cores
- Skip Existing: Reduce redundant reprocessing
- Format Selection: WebP faster than WebM, lossless slower

### Storage
- Output Location: Use SSD for better write performance
- Directory Structure: Preserve structure for organization
- Compression: Aggressive compression significantly reduces storage

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
asset-optimizer info

# Force CPU mode
asset-optimizer process --input assets --cpu --workers 1
```

### Out of Memory (OOM)
```bash
# Reduce batch size and workers
asset-optimizer process --input assets --workers 1

# Check GPU memory
nvidia-smi
```

### Processing Errors
```bash
# Enable debug logging
asset-optimizer process --input assets --log-level DEBUG

# Verify input files
ls -la assets/ | head
```

### Rembg Model Download
Models auto-download on first use. To pre-download:
```bash
python -c "from rembg import remove; remove(Image.new('RGB', (100, 100)))"
```

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src/assetfoundry
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Contributing
1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Format and lint: `black .` && `ruff check --fix .`
4. Push and create PR

## Performance Benchmarks

*On NVIDIA RTX 4070 with batch_size=1*

| Task | Format | Compression | Time | Output Size |
|------|--------|-------------|------|-------------|
| Background Removal | PNG | N/A | ~500ms | ~2-5 MB |
| Image Optimization | JPEG→WebP | Balanced | ~200ms | 40-60% smaller |
| Video Optimization | MP4→WebM | Balanced | ~30s/min | 50-70% smaller |

## License

MIT License - See LICENSE file for details

## Support

For issues, feature requests, or questions:
- Create an issue on GitHub
- Check existing issues and discussions
- Review logs with `--log-level DEBUG`

## Credits

Built with:
- [rembg](https://github.com/danielgatis/rembg) - Background removal
- [PyTorch](https://pytorch.org/) - Deep learning
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Pillow](https://python-pillow.org/) - Image processing

---

Asset Optimizer - Optimize your creative assets efficiently with GPU power.
