"""Processor modules for asset optimization."""

from asset_optimizer.processors.background_removal import BackgroundRemovalProcessor
from asset_optimizer.processors.base import BaseProcessor, ProcessorResult
from asset_optimizer.processors.caption_naming import CaptionRenameProcessor
from asset_optimizer.processors.media_optimization import MediaOptimizationProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorResult",
    "BackgroundRemovalProcessor",
    "CaptionRenameProcessor",
    "MediaOptimizationProcessor",
]
