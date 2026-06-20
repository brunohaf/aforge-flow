"""Utility modules for asset optimizer."""

from aforge_flow.utils.deduplication import DuplicateDetector
from aforge_flow.utils.logger import configure_logging, get_logger
from aforge_flow.utils.recommend import (
    HardwareInfo,
    Recommendation,
    WorkloadInfo,
    as_dict,
    build_command_line,
    detect_hardware,
    recommend_settings,
    scan_workload,
)
from aforge_flow.utils.validators import (
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
    "HardwareInfo",
    "WorkloadInfo",
    "Recommendation",
    "detect_hardware",
    "scan_workload",
    "recommend_settings",
    "build_command_line",
    "as_dict",
]
