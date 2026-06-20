"""Validators for input files and output integrity."""

from pathlib import Path
from typing import List, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv"}

# All supported formats
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_VIDEO_FORMATS


def is_supported_format(file_path: Path) -> bool:
    """
    Check if file format is supported.

    Args:
        file_path: Path to file

    Returns:
        True if format is supported, False otherwise
    """
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_FORMATS


def is_image(file_path: Path) -> bool:
    """
    Check if file is image.

    Args:
        file_path: Path to file

    Returns:
        True if file is image, False otherwise
    """
    return file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS


def is_video(file_path: Path) -> bool:
    """
    Check if file is video.

    Args:
        file_path: Path to file

    Returns:
        True if file is video, False otherwise
    """
    return file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS


def validate_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate file for processing.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    if not is_supported_format(file_path):
        return False, f"Unsupported format: {file_path.suffix}"

    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"
        if file_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            return False, "File is too large (>2GB)"
    except OSError as e:
        return False, f"Cannot access file: {e}"

    return True, ""


def get_files_recursively(
    directory: Path,
    supported_only: bool = True,
) -> List[Path]:
    """
    Get all files recursively from directory.

    Args:
        directory: Root directory
        supported_only: Only return supported formats

    Returns:
        List of file paths
    """
    files = []

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip temporary and cache files
        if file_path.name.startswith(".") or file_path.suffix in {".tmp", ".bak"}:
            continue

        if supported_only and not is_supported_format(file_path):
            continue

        files.append(file_path)

    return sorted(files)
