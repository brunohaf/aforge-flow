"""VTT Asset Forge - GPU-accelerated tabletop/VTT asset pipeline (``goforge`` CLI)."""

__version__ = "0.1.0"
__author__ = "VTT Asset Forge Contributors"
__license__ = "MIT"

from vtt_asset_forge.config import (
    AssetCategory,
    CompressionLevel,
    Config,
    MediaFormat,
    RembgModel,
)

__all__ = [
    "Config",
    "AssetCategory",
    "RembgModel",
    "MediaFormat",
    "CompressionLevel",
]
