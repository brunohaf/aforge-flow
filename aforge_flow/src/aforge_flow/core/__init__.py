"""Core domain layer: configuration, pipeline orchestrator, and taxonomy.

Re-exports the most-used symbols so callers can do ``from aforge_flow.core
import Config, AssetPipeline`` without reaching into submodules.
"""

from aforge_flow.core.config import (
    MODEL_BY_CATEGORY,
    AssetCategory,
    CaptionBackend,
    ClipModelChoice,
    CompressionLevel,
    Config,
    MediaFormat,
    RembgModel,
)
from aforge_flow.core.pipeline import AssetPipeline, result_event

__all__ = [
    "Config",
    "AssetCategory",
    "CaptionBackend",
    "ClipModelChoice",
    "CompressionLevel",
    "MediaFormat",
    "RembgModel",
    "MODEL_BY_CATEGORY",
    "AssetPipeline",
    "result_event",
]
