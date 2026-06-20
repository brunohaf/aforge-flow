"""Utility modules for asset optimizer."""

from vtt_asset_forge.utils.deduplication import DuplicateDetector
from vtt_asset_forge.utils.logger import configure_logging, get_logger
from vtt_asset_forge.utils.validators import (
    get_files_recursively,
    is_image,
    is_supported_format,
    is_video,
    validate_file,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "DuplicateDetector",
    "validate_file",
    "is_supported_format",
    "is_image",
    "is_video",
    "get_files_recursively",
]
