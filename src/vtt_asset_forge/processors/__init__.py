"""Processor modules for asset optimization."""

from vtt_asset_forge.processors.background_removal import BackgroundRemovalProcessor
from vtt_asset_forge.processors.base import BaseProcessor, ProcessorResult
from vtt_asset_forge.processors.caption_naming import CaptionRenameProcessor
from vtt_asset_forge.processors.creature_classification import (
    CreatureClassificationProcessor,
)
from vtt_asset_forge.processors.media_optimization import MediaOptimizationProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorResult",
    "BackgroundRemovalProcessor",
    "CaptionRenameProcessor",
    "CreatureClassificationProcessor",
    "MediaOptimizationProcessor",
]
