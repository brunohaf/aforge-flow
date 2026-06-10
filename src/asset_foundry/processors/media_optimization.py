"""Media optimization processor for video and image transcoding."""

import subprocess
from pathlib import Path
from typing import List, Optional

import structlog
from PIL import Image

from asset_optimizer.config import CompressionLevel, Config, MediaFormat
from asset_optimizer.processors.base import BaseProcessor, ProcessorResult
from asset_optimizer.utils import is_image, is_video, validate_file

logger = structlog.get_logger(__name__)

# FFmpeg quality presets
QUALITY_PRESETS = {
    CompressionLevel.LOSSLESS: {
        "image": {"quality": 95, "method": 6},
        "video": {"crf": 18, "preset": "slower"},
    },
    CompressionLevel.HIGH_QUALITY: {
        "image": {"quality": 85, "method": 6},
        "video": {"crf": 22, "preset": "slow"},
    },
    CompressionLevel.BALANCED: {
        "image": {"quality": 75, "method": 5},
        "video": {"crf": 28, "preset": "medium"},
    },
    CompressionLevel.AGGRESSIVE: {
        "image": {"quality": 60, "method": 4},
        "video": {"crf": 35, "preset": "fast"},
    },
}


class MediaOptimizationProcessor(BaseProcessor):
    """Processor for media optimization and transcoding."""

    def __init__(self, config: Config, output_format: Optional[MediaFormat] = None):
        """
        Initialize media optimization processor.

        Args:
            config: Asset optimizer configuration
            output_format: Output format override
        """
        super().__init__(config)
        self.output_format = output_format or config.output_format
        self.compression_level = config.compression_level

        self.logger.info(
            "initializing_processor",
            output_format=self.output_format.value,
            compression_level=self.compression_level.value,
        )

    def can_process(self, input_path: Path) -> bool:
        """Check if file can be optimized."""
        return is_image(input_path) or is_video(input_path)

    def process(self, input_path: Path) -> ProcessorResult:
        """
        Optimize media file.

        Args:
            input_path: Input file path

        Returns:
            ProcessorResult with optimized file
        """
        # Validate input
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)

        if is_image(input_path):
            return self._process_image(input_path)
        elif is_video(input_path):
            return self._process_video(input_path)
        else:
            return ProcessorResult(
                input_path,
                success=False,
                error="Unsupported media type",
            )

    def _process_image(self, input_path: Path) -> ProcessorResult:
        """Process image optimization."""
        output_path = self.get_output_path(input_path, suffix=self.config.output_suffix)
        output_path = output_path.with_suffix(f".{self.output_format.value}")

        try:
            # Skip if output exists
            if output_path.exists() and self.config.skip_existing:
                self.logger.info(
                    "skipping_existing",
                    input=str(input_path),
                    output=str(output_path),
                )
                return ProcessorResult(
                    input_path,
                    output_path=output_path,
                    success=True,
                    metadata={"skipped": True},
                )

            self.logger.info(
                "processing_image",
                input=str(input_path),
                output_format=self.output_format.value,
            )

            # Open and optimize image
            try:
                image = Image.open(input_path)

                # Convert RGBA to RGB if saving as JPEG
                if self.output_format in {MediaFormat.JPEG, MediaFormat.JPG} and image.mode == "RGBA":
                    rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3] if len(image.split()) == 4 else None)
                    image = rgb_image

                # Get quality settings
                quality_preset = QUALITY_PRESETS[self.compression_level]["image"]

                # Save optimized image
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(
                    output_path,
                    quality=quality_preset["quality"],
                    optimize=True,
                )

                # Verify output
                if self.config.verify_integrity and not output_path.exists():
                    raise OSError("Output file was not created")

                input_size = input_path.stat().st_size
                output_size = output_path.stat().st_size
                compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0

                return ProcessorResult(
                    input_path,
                    output_path=output_path,
                    success=True,
                    metadata={
                        "input_size": input_size,
                        "output_size": output_size,
                        "compression_ratio": f"{compression_ratio:.1f}%",
                    },
                )

            except Exception as e:
                return ProcessorResult(
                    input_path,
                    success=False,
                    error=f"Image processing failed: {e}",
                )

        except Exception as e:
            return ProcessorResult(
                input_path,
                success=False,
                error=f"Unexpected error: {e}",
            )

    def _process_video(self, input_path: Path) -> ProcessorResult:
        """Process video optimization."""
        output_path = self.get_output_path(input_path, suffix=self.config.output_suffix)
        output_path = output_path.with_suffix(f".{self.output_format.value}")

        try:
            # Skip if output exists
            if output_path.exists() and self.config.skip_existing:
                self.logger.info(
                    "skipping_existing",
                    input=str(input_path),
                    output=str(output_path),
                )
                return ProcessorResult(
                    input_path,
                    output_path=output_path,
                    success=True,
                    metadata={"skipped": True},
                )

            self.logger.info(
                "processing_video",
                input=str(input_path),
                output_format=self.output_format.value,
            )

            # Build FFmpeg command
            quality_preset = QUALITY_PRESETS[self.compression_level]["video"]
            cmd = self._build_ffmpeg_command(input_path, output_path, quality_preset)

            # Run FFmpeg
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )

                # Verify output
                if self.config.verify_integrity and not output_path.exists():
                    raise OSError("Output file was not created")

                input_size = input_path.stat().st_size
                output_size = output_path.stat().st_size
                compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0

                return ProcessorResult(
                    input_path,
                    output_path=output_path,
                    success=True,
                    metadata={
                        "input_size": input_size,
                        "output_size": output_size,
                        "compression_ratio": f"{compression_ratio:.1f}%",
                    },
                )

            except subprocess.TimeoutExpired:
                return ProcessorResult(
                    input_path,
                    success=False,
                    error="FFmpeg processing timeout",
                )
            except subprocess.CalledProcessError as e:
                return ProcessorResult(
                    input_path,
                    success=False,
                    error=f"FFmpeg error: {e.stderr}",
                )

        except Exception as e:
            return ProcessorResult(
                input_path,
                success=False,
                error=f"Unexpected error: {e}",
            )

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        quality_preset: dict,
    ) -> List[str]:
        """Build FFmpeg command for video transcoding."""
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-c:v",
            "libvpx-vp9" if self.output_format == MediaFormat.WEBM else "libx264",
            "-crf",
            str(quality_preset["crf"]),
            "-preset",
            quality_preset["preset"],
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-y",  # Overwrite output
            str(output_path),
        ]
        return cmd
