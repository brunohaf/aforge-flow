"""Utility modules for asset optimizer."""

from asset_optimizer.utils.deduplication import DuplicateDetector
from asset_optimizer.utils.logger import configure_logging, get_logger
from asset_optimizer.utils.validators import (
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
