"""Processor modules for asset optimization."""

from aforge_flow.processors.background_removal import BackgroundRemovalProcessor
from aforge_flow.processors.base import BaseProcessor, ProcessorResult
from aforge_flow.processors.caption_naming import CaptionRenameProcessor
from aforge_flow.processors.creature_classification import (
    CreatureClassificationProcessor,
)
from aforge_flow.processors.media_optimization import MediaOptimizationProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorResult",
    "BackgroundRemovalProcessor",
    "CaptionRenameProcessor",
    "CreatureClassificationProcessor",
    "MediaOptimizationProcessor",
]
