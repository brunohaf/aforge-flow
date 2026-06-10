"""Asset Optimizer - Professional GPU-accelerated asset optimization tool."""

__version__ = "0.1.0"
__author__ = "Asset Optimizer Contributors"
__license__ = "MIT"

from asset_optimizer.config import (
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
