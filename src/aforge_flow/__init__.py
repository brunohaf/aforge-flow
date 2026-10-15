"""aforge-flow: GPU-accelerated artwork pipeline (``aforge`` CLI)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aforge_flow")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__author__ = "Bruno Fonseca"
__license__ = "MIT"

from aforge_flow.config import (
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
