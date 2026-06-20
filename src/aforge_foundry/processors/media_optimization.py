"""Media optimization processor for video and image transcoding."""

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import structlog
from aforge_flow.config import CompressionLevel, Config, MediaFormat
from aforge_flow.processors.base import BaseProcessor, ProcessorResult
from aforge_flow.utils import is_image, is_video, validate_file
from PIL import Image

logger = structlog.get_logger(__name__)

# FFmpeg quality presets.
#
# Video params are codec-specific because each encoder uses a different quality
# control: ``crf``/``preset`` for libx264 (mp4), ``crf``/``cpu_used`` for
# libvpx-vp9 (CPU WebM), and ``cq`` for av1_nvenc (GPU WebM). The cq/cpu_used
# columns are added alongside the existing crf/preset so all three codecs map
# from the same compression level.
QUALITY_PRESETS = {
    CompressionLevel.LOSSLESS: {
        "image": {"quality": 95, "method": 6},
        "video": {"crf": 18, "preset": "slower", "cq": 24, "cpu_used": 1},
    },
    CompressionLevel.HIGH_QUALITY: {
        "image": {"quality": 85, "method": 6},
        "video": {"crf": 22, "preset": "slow", "cq": 30, "cpu_used": 2},
    },
    CompressionLevel.BALANCED: {
        "image": {"quality": 75, "method": 5},
        "video": {"crf": 28, "preset": "medium", "cq": 36, "cpu_used": 4},
    },
    CompressionLevel.AGGRESSIVE: {
        "image": {"quality": 60, "method": 4},
        "video": {"crf": 35, "preset": "fast", "cq": 44, "cpu_used": 5},
    },
}


@lru_cache(maxsize=None)
def _encoder_available(name: str) -> bool:
    """Return True if the given ffmpeg encoder (e.g. ``av1_nvenc``) is present.

    Probes ``ffmpeg -encoders`` once and caches the result for the process.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return name in out.stdout
    except Exception:
        return False


class MediaOptimizationProcessor(BaseProcessor):
    """Processor for media optimization and transcoding."""

    def __init__(
        self,
        config: Config,
        image_format: Optional[MediaFormat] = None,
        video_format: Optional[MediaFormat] = None,
    ):
        """
        Initialize media optimization processor.

        Args:
            config: Asset optimizer configuration
            image_format: Output format override for image inputs
            video_format: Output format override for video inputs
        """
        super().__init__(config)
        self.image_format = image_format or config.image_format
        self.video_format = video_format or config.video_format
        self.compression_level = config.compression_level

        self.logger.info(
            "initializing_processor",
            image_format=self.image_format.value,
            video_format=self.video_format.value,
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
        output_path = output_path.with_suffix(f".{self.image_format.value}")

        try:
            # Skip if a valid (non-empty) output already exists. The size check
            # ensures previously-failed 0-byte outputs are re-encoded, not skipped.
            if (
                output_path.exists()
                and output_path.stat().st_size > 0
                and self.config.skip_existing
            ):
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
                output_format=self.image_format.value,
            )

            # Open and optimize image
            try:
                image = Image.open(input_path)

                # Convert RGBA to RGB if saving as JPEG
                if self.image_format in {MediaFormat.JPEG, MediaFormat.JPG} and image.mode == "RGBA":
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
        output_path = output_path.with_suffix(f".{self.video_format.value}")

        try:
            # Skip if a valid (non-empty) output already exists. The size check
            # ensures previously-failed 0-byte outputs are re-encoded, not skipped.
            if (
                output_path.exists()
                and output_path.stat().st_size > 0
                and self.config.skip_existing
            ):
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

            # Build FFmpeg command
            video_encoder = self._select_video_encoder()
            self.logger.info(
                "processing_video",
                input=str(input_path),
                output_format=self.video_format.value,
                encoder=video_encoder,
            )

            quality_preset = QUALITY_PRESETS[self.compression_level]["video"]
            cmd = self._build_ffmpeg_command(
                input_path, output_path, quality_preset, video_encoder
            )

            # Run FFmpeg
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
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

    def _select_video_encoder(self) -> str:
        """Pick the video encoder for the configured output format.

        WebM prefers GPU AV1 (``av1_nvenc``) when GPU use is enabled and NVENC is
        available, falling back to CPU ``libvpx-vp9``. Other formats use
        ``libx264``. Note there is no ``vp9_nvenc`` on NVIDIA, so VP9 is always CPU.
        """
        if self.video_format == MediaFormat.WEBM:
            if self.config.use_gpu and _encoder_available("av1_nvenc"):
                return "av1_nvenc"
            return "libvpx-vp9"
        return "libx264"

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        quality_preset: dict,
        video_encoder: str,
    ) -> List[str]:
        """Build FFmpeg command for video transcoding.

        Codec, audio codec, and quality args are all selected per-encoder: WebM
        containers require Opus audio (AAC is invalid), av1_nvenc uses NVENC-style
        ``-cq``/``-preset pN`` controls, and libvpx-vp9 uses ``-crf`` with
        ``-row-mt``/``-cpu-used`` for speed (it ignores ``-preset``).
        """
        cmd: List[str] = ["ffmpeg", "-i", str(input_path), "-c:v", video_encoder]

        if video_encoder == "av1_nvenc":
            cmd += [
                "-preset", "p5",
                "-rc", "vbr",
                "-cq", str(quality_preset["cq"]),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ]
        elif video_encoder == "libvpx-vp9":
            cmd += [
                "-crf", str(quality_preset["crf"]),
                "-b:v", "0",
                "-row-mt", "1",
                "-deadline", "good",
                "-cpu-used", str(quality_preset["cpu_used"]),
            ]
        else:  # libx264
            cmd += [
                "-crf", str(quality_preset["crf"]),
                "-preset", quality_preset["preset"],
            ]

        # WebM only supports Opus/Vorbis audio; everything else uses AAC.
        if self.video_format == MediaFormat.WEBM:
            cmd += ["-c:a", "libopus", "-b:a", "128k"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "128k"]

        cmd += ["-y", str(output_path)]  # -y overwrites output
        return cmd
