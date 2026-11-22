"""Media optimization processor: images -> WebP, video -> WebM.

The robustness model mirrors :mod:`aforge_flow.processors.background_removal`:

* **Capability is probed, never assumed.** ``ffmpeg -encoders`` lists what the
  binary was *compiled* with, not what this machine can *run*: nvenc encoders are
  compiled into essentially every ffmpeg build, and they fail at runtime on a
  machine with no NVIDIA card, on a GPU whose silicon lacks that codec (AV1
  encode needs Ada / RTX 40xx), or on a driver too old for the build's NVENC API.
  So each encoder is validated with a real one-frame trial encode, exactly as
  rembg checks ``get_providers()`` for what ORT *actually activated*.
* **Every GPU path has a CPU fallback.** A failed NVENC session degrades that one
  file to a CPU encode instead of failing it.
* **Outputs are written atomically.** ffmpeg leaves a truncated file behind when
  it is killed or times out. Such a file is non-empty, so ``skip_existing`` would
  mistake it for finished work on the next run and the corruption would become
  permanent. Encodes go to a temp sibling and are renamed only once verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional

import structlog
from PIL import Image, ImageOps

from aforge_flow.core.config import CompressionLevel, Config, MediaFormat, WebmCodec
from aforge_flow.processors.base import BaseProcessor, ProcessorResult
from aforge_flow.utils import is_image, is_video, validate_file

logger = structlog.get_logger(__name__)


WEBP_MAX_DIM = 16383
"""libwebp cannot encode an image whose width or height exceeds this."""

MAX_FILENAME = 255
"""Per-component filename cap on NTFS and ext4.

Distinct from MAX_PATH: the ``\\\\?\\`` prefix lifts the 260-char limit on the *whole
path*, but no prefix lifts the 255-char limit on a single *name*. Foundry asset
names breach it on their own, before ``_optimized`` is even appended.
"""

LOSSY_SOURCE_EXTS = frozenset({".jpg", ".jpeg"})
"""Sources that are already lossy, so a lossless target cannot improve on them."""

VISUALLY_LOSSLESS_QUALITY = 92
"""WebP quality used for LOSSLESS when the source is already lossy.

Encoding a JPEG into *lossless* WebP does not recover anything — it faithfully
preserves the JPEG's own compression artifacts, at roughly 4-5x the size. That is
the same trap as bit-exact video from an h264 master (see CompressionLevel), and
it is resolved the same way: keep the pixels visually indistinguishable and let
the file get smaller. q92 is well above the source's effective quality.
"""


def fit_filename(path: Path) -> Path:
    """Shorten *path*'s final component to what the filesystem will accept.

    Truncation alone would be dangerous here — these names share very long
    prefixes (``900297781783772527_Male Monk Wizard Book Tome - ...``), so cutting
    them at 255 would silently collide distinct assets onto one output. A digest of
    the full stem is appended to keep them distinct.
    """
    if len(path.name) <= MAX_FILENAME:
        return path

    suffix = path.suffix
    digest = hashlib.sha1(path.stem.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    keep = MAX_FILENAME - len(suffix) - len(digest) - 1
    shortened = path.with_name(f"{path.stem[:keep]}-{digest}{suffix}")
    logger.warning(
        "filename_truncated",
        original_len=len(path.name),
        output=shortened.name,
        detail="name exceeded the 255-char filesystem limit; a hash keeps it unique",
    )
    return shortened

ALPHA_PIX_FMT_PREFIXES = ("yuva", "ya")
ALPHA_PIX_FMTS = frozenset(
    {"rgba", "bgra", "argb", "abgr", "rgba64le", "rgba64be", "bgra64le", "bgra64be"}
)
"""Pixel formats that carry an alpha channel.

Note this is only half of alpha detection for WebM: VP9 stores alpha in a
*separate* plane, so a transparent .webm probes as plain ``yuv420p`` and is only
identifiable by the container tag ``alpha_mode=1``. See :func:`probe_media`.
"""


class FFmpegNotFound(RuntimeError):
    """Raised when no usable ffmpeg/ffprobe binary can be located."""


def _pil_format(target: MediaFormat) -> str:
    """Pillow's registered name for *target* — note it knows JPEG, never 'JPG'."""
    if target in {MediaFormat.JPEG, MediaFormat.JPG}:
        return "JPEG"
    return target.name


@dataclass(frozen=True)
class ImagePreset:
    quality: int
    method: int


@dataclass(frozen=True)
class VideoPreset:
    """Per-codec quality settings for one compression level.

    Each codec needs its own number because the scales are *not* interchangeable:
    VP9 and SVT-AV1 use CRF 0-63, x264 uses CRF 0-51, and NVENC uses CQ 0-51.
    Feeding one codec's CRF to another (as this module previously did for
    VP9/x264) silently mis-targets quality by a wide margin.
    """

    vp9_crf: int  # 0-63
    vp9_cpu_used: int  # 0-8, lower = slower/better
    av1_crf: int  # 0-63
    av1_preset: int  # SVT-AV1 0-13, lower = slower/better
    nvenc_cq: int  # 0-51
    nvenc_preset: str  # p1 (fast) .. p7 (best)
    x264_crf: int  # 0-51
    x264_preset: str
    audio_kbps: int


# LOSSLESS is truly lossless for images, and *visually* lossless for video — see
# CompressionLevel's docstring for why bit-exact video is the wrong target here.
QUALITY_PRESETS: dict[CompressionLevel, dict] = {
    CompressionLevel.LOSSLESS: {
        "image": ImagePreset(quality=100, method=6),
        "video": VideoPreset(
            vp9_crf=18, vp9_cpu_used=0,
            av1_crf=18, av1_preset=4,
            nvenc_cq=20, nvenc_preset="p7",
            x264_crf=14, x264_preset="slower",
            audio_kbps=192,
        ),
    },
    CompressionLevel.HIGH_QUALITY: {
        "image": ImagePreset(quality=90, method=6),
        "video": VideoPreset(
            vp9_crf=26, vp9_cpu_used=1,
            av1_crf=26, av1_preset=6,
            nvenc_cq=28, nvenc_preset="p6",
            x264_crf=20, x264_preset="slow",
            audio_kbps=160,
        ),
    },
    CompressionLevel.BALANCED: {
        "image": ImagePreset(quality=80, method=5),
        "video": VideoPreset(
            vp9_crf=33, vp9_cpu_used=2,
            av1_crf=32, av1_preset=8,
            nvenc_cq=36, nvenc_preset="p5",
            x264_crf=26, x264_preset="medium",
            audio_kbps=128,
        ),
    },
    CompressionLevel.AGGRESSIVE: {
        "image": ImagePreset(quality=65, method=4),
        "video": VideoPreset(
            vp9_crf=42, vp9_cpu_used=4,
            av1_crf=42, av1_preset=10,
            nvenc_cq=46, nvenc_preset="p4",
            x264_crf=32, x264_preset="fast",
            audio_kbps=96,
        ),
    },
}


@lru_cache(maxsize=None)
def _tool(name: str) -> str:
    """Locate the ``ffmpeg``/``ffprobe`` binary, or raise :class:`FFmpegNotFound`.

    ``AFORGE_FFMPEG_DIR`` overrides the search, which matters on Windows: winget
    installs ffmpeg but the PATH change only reaches shells started afterwards,
    so a long-lived session cannot see it.
    """
    override = os.environ.get("AFORGE_FFMPEG_DIR")
    if override:
        candidate = Path(override) / f"{name}.exe" if os.name == "nt" else Path(override) / name
        if candidate.is_file():
            return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    raise FFmpegNotFound(
        f"{name} not found on PATH. Install it (Windows: `winget install Gyan.FFmpeg`, "
        f"macOS: `brew install ffmpeg`, Debian: `apt install ffmpeg`), or point "
        f"AFORGE_FFMPEG_DIR at the directory containing it."
    )


def ffmpeg_available() -> bool:
    """Return True when both ffmpeg and ffprobe can be located."""
    try:
        _tool("ffmpeg"), _tool("ffprobe")
    except FFmpegNotFound:
        return False
    return True


@lru_cache(maxsize=None)
def encoder_works(name: str) -> bool:
    """Return True only if *name* can actually encode a frame on this machine.

    Trial-encodes one frame to the null muxer. This is deliberately stronger than
    grepping ``ffmpeg -encoders``: that only proves the encoder was compiled in.
    A build listing ``av1_nvenc`` still fails at runtime on a pre-Ada GPU, with no
    NVIDIA card at all, or — observed on this project's own dev box — when the
    driver's NVENC API is older than the one the build was compiled against.
    """
    try:
        ffmpeg = _tool("ffmpeg")
    except FFmpegNotFound:
        return False

    try:
        proc = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                "-c:v", name, "-frames:v", "1", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("encoder_probe_errored", encoder=name, reason=str(exc))
        return False

    ok = proc.returncode == 0
    if not ok:
        # The reason matters: a stale driver is fixable, missing silicon is not.
        logger.info(
            "encoder_unusable",
            encoder=name,
            reason=(proc.stderr or "").strip().splitlines()[:2],
        )
    return ok


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe reports about a source file."""

    has_video: bool
    has_audio: bool
    has_alpha: bool
    pix_fmt: str


def _pix_fmt_has_alpha(pix_fmt: str) -> bool:
    return pix_fmt.startswith(ALPHA_PIX_FMT_PREFIXES) or pix_fmt in ALPHA_PIX_FMTS


def probe_media(path: Path) -> MediaInfo:
    """Inspect *path* with ffprobe.

    Alpha needs two checks, not one. A ProRes 4444 .mov or a PNG sequence exposes
    alpha in its ``pix_fmt`` (``yuva444p10le``, ``rgba``), but a transparent WebM
    does not: VP9 keeps alpha in a separate plane and the base ``pix_fmt`` still
    reads ``yuv420p``. Only the container tag ``alpha_mode=1`` reveals it. Testing
    ``pix_fmt`` alone would silently classify every transparent WebM as opaque and
    flatten it to black.
    """
    proc = subprocess.run(
        [
            _tool("ffprobe"), "-v", "error",
            "-show_entries", "stream=codec_type,pix_fmt:stream_tags=alpha_mode",
            "-of", "json", os_path(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    streams = json.loads(proc.stdout or "{}").get("streams", [])

    has_video = has_audio = has_alpha = False
    pix_fmt = ""

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            has_audio = True
        elif codec_type == "video":
            has_video = True
            pix_fmt = pix_fmt or stream.get("pix_fmt", "")
            tags = stream.get("tags") or {}
            if _pix_fmt_has_alpha(stream.get("pix_fmt", "")) or tags.get("alpha_mode") == "1":
                has_alpha = True

    return MediaInfo(has_video=has_video, has_audio=has_audio, has_alpha=has_alpha, pix_fmt=pix_fmt)


def os_path(path: Path) -> str:
    """Render *path* for an OS call, escaping Windows' 260-char MAX_PATH if needed.

    Foundry asset trees nest deeply and carry very long descriptive filenames, so a
    minority of outputs land past MAX_PATH and fail with WinError 3/123. The ``\\\\?\\``
    prefix lifts the limit to 32767, but it demands a fully-qualified, normalised
    path — hence the ``resolve()``.
    """
    if os.name != "nt":
        return str(path)
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\") or len(resolved) < 250:
        return resolved
    return f"\\\\?\\{resolved}"


@contextmanager
def atomic_output(final: Path) -> Iterator[Path]:
    """Yield a temp path to write to, renamed onto *final* only on clean exit.

    Guards against the trap that a crashed/timed-out ffmpeg leaves a truncated
    but non-empty file at the destination, which ``skip_existing`` then treats as
    completed work on every subsequent run — silently baking in the corruption.

    The temp name is kept short (``~`` + 8 hex) because it has to fit inside the
    same path budget as the final name.
    """
    Path(os_path(final.parent)).mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"~{uuid.uuid4().hex[:8]}.part")
    try:
        yield tmp
    except BaseException:
        Path(os_path(tmp)).unlink(missing_ok=True)
        raise
    os.replace(os_path(tmp), os_path(final))


class MediaOptimizationProcessor(BaseProcessor):
    """Transcode images to WebP and video to WebM (or other configured targets)."""

    _nvenc_slots: Optional[threading.Semaphore] = None
    """Caps concurrent NVENC sessions across all worker threads; see config.nvenc_max_sessions."""

    def __init__(
        self,
        config: Config,
        image_format: Optional[MediaFormat] = None,
        video_format: Optional[MediaFormat] = None,
    ):
        super().__init__(config)
        self.image_format = image_format or config.image_format
        self.video_format = video_format or config.video_format
        self.compression_level = config.compression_level

        if MediaOptimizationProcessor._nvenc_slots is None:
            MediaOptimizationProcessor._nvenc_slots = threading.Semaphore(
                config.nvenc_max_sessions
            )

        # Surface a missing ffmpeg at startup rather than as a cryptic WinError 2
        # on the first video, but do not hard-fail: an image-only batch never
        # needs ffmpeg.
        self._ffmpeg_ok = ffmpeg_available()
        if not self._ffmpeg_ok:
            self.logger.warning(
                "ffmpeg_missing",
                detail="video files will fail; images are unaffected",
                fix="winget install Gyan.FFmpeg (or set AFORGE_FFMPEG_DIR)",
            )

        self.logger.info(
            "initializing_processor",
            image_format=self.image_format.value,
            video_format=self.video_format.value,
            compression_level=self.compression_level.name,
            webm_codec=config.webm_codec.value,
            ffmpeg=self._ffmpeg_ok,
        )

    def can_process(self, input_path: Path) -> bool:
        return is_image(input_path) or is_video(input_path)

    def process(self, input_path: Path) -> ProcessorResult:
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)

        target = self.image_format if is_image(input_path) else self.video_format
        output_path = fit_filename(
            self.get_output_path(
                input_path, suffix=self.config.output_suffix
            ).with_suffix(f".{target.value}")
        )

        if self._skip_existing(input_path, output_path):
            return ProcessorResult(
                input_path, output_path=output_path, success=True, metadata={"skipped": True}
            )

        if self._is_already_target_format(input_path):
            return self._copy_through(input_path, output_path)

        if is_image(input_path):
            result = self._process_image(input_path, output_path)
        elif is_video(input_path):
            result = self._process_video(input_path, output_path)
        else:
            return ProcessorResult(input_path, success=False, error="Unsupported media type")

        return self._reject_if_bigger(input_path, result)

    def _reject_if_bigger(self, input_path: Path, result: ProcessorResult) -> ProcessorResult:
        """Discard a transcode that came out larger than its source, keeping the original.

        Some sources cannot be beaten: a well-encoded h264 mp4 re-encoded to AV1 at a
        quality-preserving CRF roughly *doubles* in size, because the only way to get
        smaller is to throw away quality the source still has. Emitting that file would
        defeat the entire point of the run, so we keep the original bytes instead — the
        library can then never grow, whatever it is fed.
        """
        if not (self.config.keep_smallest_output and result.success):
            return result
        if result.metadata.get("passthrough"):
            return result

        input_size = result.metadata.get("input_size", 0)
        output_size = result.metadata.get("output_size", 0)
        if not input_size or output_size <= input_size:
            return result

        original = result.output_path
        # Keep the source's own extension — this is a copy, not a transcode.
        kept = fit_filename(
            self.get_output_path(input_path, suffix=self.config.output_suffix)
        )
        self.logger.warning(
            "transcode_rejected_kept_original",
            input=str(input_path),
            input_size=input_size,
            transcoded_size=output_size,
            detail="re-encoding this source only makes it bigger; copying it verbatim",
        )

        try:
            if original is not None:
                Path(os_path(original)).unlink(missing_ok=True)
            with atomic_output(kept) as tmp:
                shutil.copy2(os_path(input_path), os_path(tmp))
        except Exception as exc:
            return ProcessorResult(
                input_path, success=False, error=f"Failed to keep original: {exc}"
            )

        return ProcessorResult(
            input_path,
            output_path=kept,
            success=True,
            metadata={
                "input_size": input_size,
                "output_size": input_size,
                "compression_ratio": "0.0%",
                "kept_original": True,
                "rejected_transcode_size": output_size,
            },
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _skip_existing(self, input_path: Path, output_path: Path) -> bool:
        existing = Path(os_path(output_path))
        if not (self.config.skip_existing and existing.exists()):
            return False
        if existing.stat().st_size == 0:
            return False  # a 0-byte leftover is a previous failure, not a result
        self.logger.info("skipping_existing", input=str(input_path), output=str(output_path))
        return True

    def _is_already_target_format(self, input_path: Path) -> bool:
        """True when *input_path* is already in its target container.

        Re-encoding a .webp to .webp (or .webm to .webm) buys nothing and, at any
        lossy level, costs a full generation of quality.
        """
        if not self.config.passthrough_matching_format:
            return False

        suffix = input_path.suffix.lower().lstrip(".")
        target = (self.image_format if is_image(input_path) else self.video_format).value

        jpeg_aliases = {"jpg", "jpeg"}
        if suffix in jpeg_aliases and target in jpeg_aliases:
            return True
        return suffix == target

    def _copy_through(self, input_path: Path, output_path: Path) -> ProcessorResult:
        """Copy an already-optimal file verbatim to the output tree."""
        self.logger.info("passthrough_copy", input=str(input_path), output=str(output_path))
        try:
            with atomic_output(output_path) as tmp:
                shutil.copy2(os_path(input_path), os_path(tmp))
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"Copy failed: {exc}")

        size = Path(os_path(output_path)).stat().st_size
        return ProcessorResult(
            input_path,
            output_path=output_path,
            success=True,
            metadata={
                "input_size": size,
                "output_size": size,
                "compression_ratio": "0.0%",
                "passthrough": True,
            },
        )

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _process_image(self, input_path: Path, output_path: Path) -> ProcessorResult:
        preset: ImagePreset = QUALITY_PRESETS[self.compression_level]["image"]
        is_lossless = self.compression_level == CompressionLevel.LOSSLESS
        source_is_lossy = input_path.suffix.lower() in LOSSY_SOURCE_EXTS
        target = self.image_format

        if is_lossless and target in {MediaFormat.JPEG, MediaFormat.JPG}:
            return ProcessorResult(
                input_path,
                success=False,
                error="LOSSLESS is incompatible with JPEG output; use --image-format webp or png",
            )

        try:
            with Image.open(os_path(input_path)) as opened:
                if getattr(opened, "n_frames", 1) > 1:
                    self.logger.warning(
                        "multiframe_input_flattened",
                        input=str(input_path),
                        frames=opened.n_frames,
                        detail="only the first frame is kept",
                    )
                image = self._prepare_image(opened, target)

                # libwebp rejects anything past 16383px outright. Failing the file
                # would drop the asset entirely, so degrade the container to PNG
                # (also lossless) and keep the pixels.
                if target == MediaFormat.WEBP and max(image.size) > WEBP_MAX_DIM:
                    output_path = output_path.with_suffix(".png")
                    target = MediaFormat.PNG
                    self.logger.warning(
                        "webp_dimension_limit_exceeded",
                        input=str(input_path),
                        size=image.size,
                        limit=WEBP_MAX_DIM,
                        detail="falling back to PNG for this file",
                    )

                self.logger.info(
                    "processing_image",
                    input=str(input_path),
                    output_format=target.value,
                    lossless=is_lossless and not source_is_lossy,
                    source_is_lossy=source_is_lossy,
                )

                with atomic_output(output_path) as tmp:
                    image.save(
                        os_path(tmp),
                        format=_pil_format(target),
                        **self._image_save_kwargs(
                            preset, is_lossless, target, source_is_lossy
                        ),
                    )
                    self._verify_image(tmp)
        except Exception as exc:
            return ProcessorResult(
                input_path, success=False, error=f"Image processing failed: {exc}"
            )

        return self._result(input_path, output_path, extra={"format": target.value})

    def _prepare_image(self, image: Image.Image, target: MediaFormat) -> Image.Image:
        """Normalise orientation and colour mode for *target*.

        Two things the previous version got wrong. EXIF orientation was ignored,
        so camera/phone JPEGs came out rotated. And only the RGBA->JPEG case was
        handled, so a CMYK TIFF or 16-bit PNG raised inside ``save``.
        """
        image = ImageOps.exif_transpose(image)

        has_alpha = "A" in image.getbands() or (
            image.mode == "P" and "transparency" in image.info
        )

        if target in {MediaFormat.JPEG, MediaFormat.JPG}:
            if not has_alpha:
                return image.convert("RGB")
            # JPEG has no alpha: composite onto white rather than dropping the
            # channel, which would expose whatever RGB sat under transparent pixels.
            rgba = image.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat

        return image.convert("RGBA" if has_alpha else "RGB")

    def _image_save_kwargs(
        self,
        preset: ImagePreset,
        is_lossless: bool,
        target: MediaFormat,
        source_is_lossy: bool = False,
    ) -> dict:
        if target == MediaFormat.WEBP:
            if is_lossless and source_is_lossy:
                # "Lossless as far as possible" for a JPEG means visually lossless,
                # not bit-exact — see VISUALLY_LOSSLESS_QUALITY.
                return {"quality": VISUALLY_LOSSLESS_QUALITY, "method": preset.method}
            if is_lossless:
                # In lossless mode libwebp reinterprets `quality` as compression
                # *effort* (higher = smaller/slower), so 100 + method 6 is the
                # maximum-compression setting, not a quality claim.
                return {"lossless": True, "quality": 100, "method": preset.method}
            return {"quality": preset.quality, "method": preset.method}

        if target == MediaFormat.PNG:
            return {"optimize": True}

        return {"quality": preset.quality, "optimize": True, "subsampling": "4:4:4"}

    def _verify_image(self, path: Path) -> None:
        if not self.config.verify_integrity:
            return
        if Path(os_path(path)).stat().st_size == 0:
            raise OSError("Encoder produced an empty file")
        with Image.open(os_path(path)) as written:
            written.verify()  # decodes the container; catches truncated writes

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def _process_video(self, input_path: Path, output_path: Path) -> ProcessorResult:
        if not self._ffmpeg_ok:
            return ProcessorResult(
                input_path,
                success=False,
                error="ffmpeg not found — install it (winget install Gyan.FFmpeg) "
                "or set AFORGE_FFMPEG_DIR",
            )

        try:
            info = probe_media(input_path)
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"ffprobe failed: {exc}")

        if not info.has_video:
            return ProcessorResult(input_path, success=False, error="No video stream found")

        encoder = self._select_video_encoder(info.has_alpha)
        preset: VideoPreset = QUALITY_PRESETS[self.compression_level]["video"]

        self.logger.info(
            "processing_video",
            input=str(input_path),
            output_format=self.video_format.value,
            encoder=encoder,
            has_alpha=info.has_alpha,
            has_audio=info.has_audio,
        )

        result = self._encode(input_path, output_path, info, preset, encoder)
        if result.success or not self.config.cpu_fallback_on_oom:
            return result

        # An NVENC session can fail for reasons that have nothing to do with this
        # file (driver-level session cap, another process holding the encoder).
        # Degrade this one file to CPU instead of losing it — the CPU encoder is
        # also the higher-quality-per-bit one, so nothing is given up but time.
        cpu_encoder = self._cpu_encoder_for(info.has_alpha)
        if encoder == cpu_encoder:
            return result

        self.logger.warning(
            "gpu_encode_failed_cpu_fallback",
            input=str(input_path),
            gpu_encoder=encoder,
            cpu_encoder=cpu_encoder,
            error=result.error,
        )
        fallback = self._encode(input_path, output_path, info, preset, cpu_encoder)
        if fallback.success:
            fallback.metadata["device"] = "cpu_fallback"
        return fallback

    def _encode(
        self,
        input_path: Path,
        output_path: Path,
        info: MediaInfo,
        preset: VideoPreset,
        encoder: str,
    ) -> ProcessorResult:
        cmd_tail = self._build_ffmpeg_command(input_path, info, preset, encoder)
        is_nvenc = encoder.endswith("_nvenc")

        try:
            with atomic_output(output_path) as tmp:
                # ffmpeg infers the container from the extension, which the .part
                # suffix would hide — so state it explicitly.
                cmd = cmd_tail + ["-f", self._muxer(), "-y", os_path(tmp)]
                slots = MediaOptimizationProcessor._nvenc_slots
                if is_nvenc and slots is not None:
                    with slots:
                        self._run_ffmpeg(cmd)
                else:
                    self._run_ffmpeg(cmd)
                self._verify_video(tmp)
        except subprocess.TimeoutExpired:
            return ProcessorResult(
                input_path,
                success=False,
                error=f"ffmpeg timed out after {self.config.ffmpeg_timeout_seconds}s",
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()
            return ProcessorResult(
                input_path,
                success=False,
                error=f"ffmpeg error ({encoder}): {' | '.join(detail[-3:]) or 'unknown'}",
            )
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"Encode failed: {exc}")

        return self._result(input_path, output_path, extra={"encoder": encoder})

    def _run_ffmpeg(self, cmd: List[str]) -> None:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.config.ffmpeg_timeout_seconds,
        )

    def _muxer(self) -> str:
        return "webm" if self.video_format == MediaFormat.WEBM else self.video_format.value

    def _cpu_encoder_for(self, has_alpha: bool) -> str:
        """The CPU encoder for the configured container.

        WebM alpha is VP9-only: SVT-AV1 fed a ``yuva420p`` stream emits a corrupt
        file rather than erroring, so alpha sources must never reach it.
        """
        if self.video_format != MediaFormat.WEBM:
            return "libx264"
        if has_alpha:
            return "libvpx-vp9"

        codec = self.config.webm_codec
        if codec == WebmCodec.VP9:
            return "libvpx-vp9"
        if codec in (WebmCodec.AV1, WebmCodec.AUTO) and encoder_works("libsvtav1"):
            return "libsvtav1"
        return "libvpx-vp9"

    def _select_video_encoder(self, has_alpha: bool) -> str:
        """Pick the encoder for this file.

        GPU is only a *speed* option here, never a quality one — NVENC trades
        compression efficiency for throughput, so the CPU encoders produce smaller
        files at equal quality. It is therefore skipped entirely for alpha (which
        no NVENC encoder supports) and is only used when the trial encode proves
        the driver and silicon can actually run it.
        """
        if has_alpha or self.video_format != MediaFormat.WEBM:
            if not has_alpha and self.config.use_gpu and self.video_format != MediaFormat.WEBM:
                if encoder_works("h264_nvenc"):
                    return "h264_nvenc"
            return self._cpu_encoder_for(has_alpha)

        if self.config.use_gpu and self.config.webm_codec != WebmCodec.VP9:
            if encoder_works("av1_nvenc"):
                return "av1_nvenc"
            self.logger.info(
                "nvenc_unavailable_using_cpu",
                detail="av1_nvenc failed its trial encode; CPU encoding is higher quality anyway",
            )

        return self._cpu_encoder_for(has_alpha)

    def _build_ffmpeg_command(
        self, input_path: Path, info: MediaInfo, preset: VideoPreset, encoder: str
    ) -> List[str]:
        """Assemble everything up to the output path.

        ``-nostdin`` matters once ``max_workers > 1``: without it every concurrent
        ffmpeg inherits the terminal's stdin and they fight over keystrokes.
        """
        cmd: List[str] = [
            _tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", os_path(input_path), "-c:v", encoder,
        ]

        if encoder == "libvpx-vp9":
            cmd += [
                "-crf", str(preset.vp9_crf),
                "-b:v", "0",  # constant-quality mode; without this CRF is ignored
                "-row-mt", "1",
                "-deadline", "good",
                "-cpu-used", str(preset.vp9_cpu_used),
                # VP9 is the only WebM encoder that carries alpha, and it needs to
                # be told to keep it: the default yuv420p would flatten it away.
                "-pix_fmt", "yuva420p" if info.has_alpha else "yuv420p",
            ]
        elif encoder == "libsvtav1":
            cmd += [
                "-crf", str(preset.av1_crf),
                "-preset", str(preset.av1_preset),
                "-svtav1-params", "tune=0",  # 0 = subjective quality, 1 = PSNR
                "-pix_fmt", "yuv420p",
            ]
        elif encoder.endswith("_nvenc"):
            cmd += [
                "-preset", preset.nvenc_preset,
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", str(preset.nvenc_cq),
                "-b:v", "0",
                "-spatial-aq", "1",
                "-pix_fmt", "yuv420p",
            ]
        else:  # libx264
            cmd += [
                "-crf", str(preset.x264_crf),
                "-preset", preset.x264_preset,
                # Pin 4:2:0 8-bit: a 4:4:4 or 10-bit source would otherwise yield a
                # High 4:4:4 profile file that many players cannot decode.
                "-pix_fmt", "yuv420p",
            ]

        threads = self._thread_budget()
        if threads:
            cmd += ["-threads", str(threads)]

        if not info.has_audio:
            cmd += ["-an"]
        elif self.video_format == MediaFormat.WEBM:
            # WebM accepts only Opus/Vorbis; AAC in a .webm is an invalid file.
            cmd += ["-c:a", "libopus", "-b:a", f"{preset.audio_kbps}k", "-vbr", "on"]
        else:
            cmd += ["-c:a", "aac", "-b:a", f"{preset.audio_kbps}k"]

        return cmd

    def _thread_budget(self) -> int:
        """Cores per ffmpeg job, so N workers do not each grab every core."""
        if self.config.max_workers <= 1:
            return 0  # 0 = let ffmpeg decide
        cores = os.cpu_count() or 1
        return max(1, cores // self.config.max_workers)

    def _verify_video(self, path: Path) -> None:
        """Confirm the encode really produced a decodable video stream.

        Not redundant with ffmpeg's exit code: SVT-AV1 handed an alpha input was
        observed to exit cleanly while writing a file whose EBML header will not
        parse. The old ``output_path.exists()`` check could never have caught that.
        """
        if not self.config.verify_integrity:
            return
        if Path(os_path(path)).stat().st_size == 0:
            raise OSError("Encoder produced an empty file")
        if not probe_media(path).has_video:
            raise OSError("Encoded file has no decodable video stream")

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _result(self, input_path: Path, output_path: Path, extra: dict) -> ProcessorResult:
        input_size = Path(os_path(input_path)).stat().st_size
        output_size = Path(os_path(output_path)).stat().st_size
        ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0.0

        if output_size > input_size:
            # Routine and legitimate when transcoding a lossy JPEG/h264 source to a
            # lossless/near-lossless target, but worth surfacing on a run whose whole
            # purpose is to shrink the library.
            self.logger.warning(
                "output_larger_than_input",
                input=str(input_path),
                input_size=input_size,
                output_size=output_size,
            )

        return ProcessorResult(
            input_path,
            output_path=output_path,
            success=True,
            metadata={
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": f"{ratio:.1f}%",
                **extra,
            },
        )
