"""Base processor class for asset optimization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from asset_optimizer.config import Config

logger = structlog.get_logger(__name__)


class ProcessorResult:
    """Result of processing an asset."""

    def __init__(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        success: bool = False,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize processor result.

        Args:
            input_path: Input file path
            output_path: Output file path
            success: Whether processing succeeded
            error: Error message if failed
            metadata: Additional metadata
        """
        self.input_path = input_path
        self.output_path = output_path
        self.success = success
        self.error = error
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        """String representation."""
        status = "✓" if self.success else "✗"
        return (
            f"{status} {self.input_path.name} "
            f"→ {self.output_path.name if self.output_path else 'N/A'}"
        )


class BaseProcessor(ABC):
    """Abstract base class for asset processors."""

    def __init__(self, config: Config):
        """
        Initialize processor.

        Args:
            config: Asset optimizer configuration
        """
        self.config = config
        self.logger = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    def process(self, input_path: Path) -> ProcessorResult:
        """
        Process asset.

        Args:
            input_path: Input file path

        Returns:
            ProcessorResult instance
        """
        pass

    @abstractmethod
    def can_process(self, input_path: Path) -> bool:
        """
        Check if processor can handle file.

        Args:
            input_path: Input file path

        Returns:
            True if processor can handle file
        """
        pass

    def get_output_path(self, input_path: Path, suffix: Optional[str] = None) -> Path:
        """
        Get output path for input file.

        Args:
            input_path: Input file path
            suffix: Custom suffix for output file

        Returns:
            Output file path
        """
        if self.config.preserve_directory_structure:
            # Preserve directory structure
            rel_path = input_path.relative_to(self.config.input_dir)
            output_dir = self.config.output_dir / rel_path.parent
        else:
            output_dir = self.config.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename
        if suffix is None:
            suffix = self.config.output_suffix or ""

        stem = input_path.stem
        new_name = f"{stem}{suffix}{input_path.suffix}"
        return output_dir / new_name

    def log_result(self, result: ProcessorResult) -> None:
        """Log processing result."""
        if result.success:
            self.logger.info(
                "processing_success",
                input=str(result.input_path),
                output=str(result.output_path),
                **result.metadata,
            )
        else:
            self.logger.error(
                "processing_failed",
                input=str(result.input_path),
                error=result.error,
            )
