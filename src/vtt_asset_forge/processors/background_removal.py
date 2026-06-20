"""Background removal processor using rembg."""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog
from vtt_asset_forge.config import MODEL_BY_CATEGORY, AssetCategory, Config
from vtt_asset_forge.processors.base import BaseProcessor, ProcessorResult
from vtt_asset_forge.utils import is_image, validate_file
from PIL import Image

if TYPE_CHECKING:
    from rembg import BaseSession

logger = structlog.get_logger(__name__)

MAX_INFERENCE_DIM: int = 1024
"""Cap used only by the optional alpha-matting pass to bound its RAM use.

Note: it does NOT reduce segmentation VRAM — BiRefNet/BRIA hard-code a
1024x1024 input inside rembg, so downscaling the source before inference is
upscaled straight back and saves nothing."""

OOM_MARKERS = frozenset(
    {
        "out of memory",
        "failed to allocate",
        "cuda failure",
        "cudamalloc",
        "cudnn_status",
        "cudamemcpyasync",
        "cudastreamsynchronize",
    }
)
"""Lowercased substrings that identify GPU OOM / corrupted-CUDA-context errors
in ONNX/CUDA tracebacks (e.g. ``CUDA failure 2: out of memory``,
``CUDNN_STATUS_INTERNAL_ERROR``). Matched case-insensitively via
:func:`_is_oom_error`."""


def _is_oom_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* looks like a GPU OOM or corrupted-context error."""
    msg = str(exc).lower()
    return any(marker in msg for marker in OOM_MARKERS)


def resize_for_inference(
    image: Image.Image,
    max_dim: int = MAX_INFERENCE_DIM,
) -> tuple[Image.Image, tuple[int, int]]:
    """Downscale *image* so its longest side is at most *max_dim*.

    If the image already fits, it is returned unchanged.

    Args:
        image: Source image in any PIL mode.
        max_dim: Maximum allowed size for the longest side, in pixels.

    Returns:
        A ``(inference_image, original_size)`` tuple where *original_size*
        is ``(width, height)`` before any resizing.
    """
    original_size: tuple[int, int] = image.size
    longest = max(original_size)

    if longest <= max_dim:
        return image, original_size

    scale = max_dim / longest
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

    logger.debug("resized_for_inference", original=original_size, inference=new_size)
    return image.resize(new_size, Image.LANCZOS), original_size


def composite_mask(original: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply *mask*'s alpha channel onto full-resolution *original*.

    When the segmentation was run on a downscaled copy, *mask* will be
    smaller than *original*.  This function upscales the alpha channel back
    with ``LANCZOS`` before compositing so output sharpness is bounded by
    the original image, not the inference resolution.

    Args:
        original: Full-resolution source image (any mode).
        mask: RGBA image produced by rembg, possibly at a reduced size.

    Returns:
        Full-resolution RGBA image with background removed.
    """
    if mask.mode != "RGBA":
        mask = mask.convert("RGBA")

    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)

    r, g, b, _ = original.convert("RGBA").split()
    *_, alpha = mask.split()
    return Image.merge("RGBA", (r, g, b, alpha))

class BackgroundRemovalProcessor(BaseProcessor):
    """Remove image backgrounds using `rembg` and an ONNX segmentation model.

    GPU memory is managed defensively:

    * All inference calls are serialised through a class-level lock so that
      multiple processor instances (e.g. from a thread pool) never compete
      for VRAM simultaneously.
    * The CUDA execution provider is created with a memory-frugal config
      (heuristic cuDNN algo search, request-sized arena) so a single 1024²
      BiRefNet/BRIA inference fits on an 8 GB laptop GPU.
    * On OOM the corrupted session is destroyed and rebuilt; if VRAM still
      will not fit the image, it falls back to a cached CPU session
      (``config.cpu_fallback_on_oom``) so the batch never cascade-fails.
      Output quality is identical on CPU — only slower.

    Args:
        config: Global vtt_asset_forge configuration.
        category: Optional category override for model selection.  Falls back
            to ``config.rembg_category`` when omitted.
    """

    _gpu_lock: threading.Lock = threading.Lock()
    """Serialises GPU calls across all instances to prevent concurrent VRAM exhaustion."""

    def __init__(self, config: Config, category: Optional[AssetCategory] = None) -> None:
        super().__init__(config)

        self.category = category or config.rembg_category
        self.model_name: str = (
            MODEL_BY_CATEGORY[self.category].value
            if self.category in MODEL_BY_CATEGORY
            else config.rembg_model.value
        )
        self.use_gpu: bool = config.use_gpu

        # Lazily-built CPU session, reused as the OOM fallback for every image.
        self._cpu_session: Optional[BaseSession] = None

        logger.info("initializing_processor", model=self.model_name, category=self.category)
        self.session: BaseSession = self._create_session()

    def _build_providers(self, force_cpu: bool) -> list:
        """Return an ONNXRuntime ``providers`` list for :meth:`_create_session`.

        On GPU, the CUDA provider is configured to minimise VRAM: a heuristic
        cuDNN convolution-algorithm search (instead of the default EXHAUSTIVE
        search, which allocates a huge workspace) and a request-sized arena.
        ``config.gpu_mem_limit_gb`` optionally caps the arena. ``CPUExecutionProvider``
        is appended as a graceful fallback for unsupported nodes.
        """
        if force_cpu or not self.use_gpu:
            return ["CPUExecutionProvider"]

        cuda_opts: dict = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        if self.config.gpu_mem_limit_gb > 0:
            cuda_opts["gpu_mem_limit"] = int(self.config.gpu_mem_limit_gb * 1024**3)

        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

    def _create_session(self, force_cpu: bool = False) -> BaseSession:
        """Initialise an rembg inference session for :attr:`model_name`.

        Args:
            force_cpu: Build a CPU-only session, ignoring GPU configuration.

        Returns:
            A ready-to-use rembg ``BaseSession``.

        Raises:
            Exception: Propagates any error raised by ``rembg.new_session``
                after logging it.
        """
        from rembg import new_session

        providers = self._build_providers(force_cpu)

        try:
            session = new_session(self.model_name, providers=providers)
        except Exception:
            logger.exception("session_initialization_failed", model=self.model_name)
            raise

        logger.info("session_initialized", model=self.model_name, providers=providers)
        return session

    def _get_cpu_session(self) -> BaseSession:
        """Return a cached CPU-only session, building it on first use."""
        if self._cpu_session is None:
            logger.info("creating_cpu_fallback_session", model=self.model_name)
            self._cpu_session = self._create_session(force_cpu=True)
        return self._cpu_session

    def _flush_gpu_memory(self) -> None:
        """Destroy the GPU session to free its ONNXRuntime CUDA arena.

        ONNXRuntime owns its CUDA allocations independently of PyTorch, so
        ``torch.cuda.empty_cache()`` cannot free them — only dropping the
        ``InferenceSession`` and collecting it does. We therefore release
        :attr:`session` here; callers must rebuild it before the next GPU run.
        ``torch`` is flushed too as a best-effort for any shared context.
        """
        self.session = None  # type: ignore[assignment]
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        gc.collect()

    def can_process(self, input_path: Path) -> bool:
        """Return ``True`` when *input_path* is a supported image format.

        Args:
            input_path: Candidate file path.
        """
        return is_image(input_path)


    def _segment(self, source: Image.Image, remove_fn, session: BaseSession) -> Image.Image:
        """Run rembg segmentation and refine the mask onto the full-res *source*.

        rembg already segments at the model's fixed 1024² and LANCZOS-upscales
        the mask back to *source* size, so no manual downscaling is needed.
        """
        mask = remove_fn(source, session=session)

        if mask.size != source.size:
            mask = composite_mask(source, mask)

        return self._refine_edges(source, mask)

    def _remove_with_retry(
        self,
        input_path: Path,
        source: Image.Image,
        output_path: Path,
        remove_fn,
        max_retries: int = 2,
    ) -> ProcessorResult:
        """Run background removal with OOM recovery, then edge refinement.

        Because BiRefNet/BRIA run at a fixed 1024² inside rembg, shrinking the
        input cannot relieve VRAM pressure. Instead, on an OOM we destroy the
        (now cuDNN-corrupted) GPU session and rebuild a fresh one before
        retrying. If the GPU still cannot fit the image and
        ``config.cpu_fallback_on_oom`` is set, the image is processed on a
        cached CPU session — identical quality, just slower — so a single
        oversized image never cascade-fails the whole batch.

        Args:
            input_path: Original file path (used only for result metadata).
            source: Full-resolution source image.
            output_path: Destination path for the PNG output.
            remove_fn: Callable with the same signature as ``rembg.remove``.
            max_retries: GPU attempts (with session rebuild) before CPU fallback.

        Returns:
            A :class:`ProcessorResult` describing the outcome.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                with self._gpu_lock:
                    if self.session is None:
                        self.session = self._create_session()
                    mask = self._segment(source, remove_fn, self.session)
                return self._save(input_path, mask, output_path)

            except Exception as exc:
                last_exc = exc
                if not _is_oom_error(exc):
                    logger.exception("processing_failed", input=str(input_path))
                    return ProcessorResult(
                        input_path,
                        success=False,
                        error=f"Background removal failed: {exc}",
                    )

                logger.warning(
                    "oom_rebuilding_session",
                    input=str(input_path),
                    attempt=attempt,
                )
                with self._gpu_lock:
                    self._flush_gpu_memory()  # drops the corrupted GPU session
                time.sleep(0.5 * attempt)

        # GPU could not fit the image — fall back to CPU rather than fail it.
        if self.config.cpu_fallback_on_oom:
            logger.warning("oom_cpu_fallback", input=str(input_path))
            try:
                mask = self._segment(source, remove_fn, self._get_cpu_session())
                result = self._save(input_path, mask, output_path)
                result.metadata["device"] = "cpu_fallback"
                return result
            except Exception as exc:
                logger.exception("cpu_fallback_failed", input=str(input_path))
                last_exc = exc

        return ProcessorResult(
            input_path,
            success=False,
            error=f"Background removal failed after OOM recovery: {last_exc}",
        )

    def process(self, input_path: Path) -> ProcessorResult:
        """Remove the background from a single image file.

        The result is always saved as a PNG with an alpha channel.  If
        ``config.skip_existing`` is enabled and the output already exists,
        the file is left untouched and a successful result is returned
        immediately.

        Args:
            input_path: Path to the source image.

        Returns:
            A :class:`ProcessorResult` whose ``success`` flag indicates
            whether background removal and saving completed without error.
        """
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)

        output_path = self._mirror_output_path(input_path)

        if output_path.exists() and self.config.skip_existing:
            logger.info("skipping_existing", input=str(input_path), output=str(output_path))
            return ProcessorResult(
                input_path,
                output_path=output_path,
                success=True,
                metadata={"skipped": True},
            )

        try:
            from rembg import remove as rembg_remove
        except ImportError:
            return ProcessorResult(
                input_path,
                success=False,
                error="rembg is not installed — run: pip install rembg[gpu]",
            )

        try:
            source = Image.open(input_path)
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"Failed to open image: {exc}")

        logger.info("processing_start", input=str(input_path), model=self.model_name, size=source.size)

        return self._remove_with_retry(input_path, source, output_path, rembg_remove)

    def _mirror_output_path(self, input_path: Path) -> Path:
        """Derive an output path that mirrors *input_path*'s position within the input directory.

        Given ``input_dir/a/b/image.jpg`` and ``output_dir``, produces
        ``output_dir/a/b/image_nobg.png``, preserving the full subdirectory
        hierarchy rather than flattening everything into the output root.

        Args:
            input_path: Absolute or relative path to the source image.

        Returns:
            Destination path under :attr:`config.output_dir` with ``_nobg``
            appended to the stem and the extension replaced with ``.png``.
        """
        relative = input_path.relative_to(self.config.input_dir)
        return self.config.output_dir / relative.parent / f"{input_path.stem}_nobg.png"

    def _refine_edges(self, source: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply the configured edge-refinement stages to a segmented mask.

        For strong models (BiRefNet/BRIA) the mask is already clean, so the
        defaults keep this light to avoid the "smudged" look:

        * **Alpha edge sharpening** (default on) — a smoothstep contrast curve
        around 0.5 that crispens the alpha edge and shrinks the
        semi-transparent halo while preserving anti-aliasing.
        * **Alpha matting** (default off) — recovers smooth hair/fur edges via
        ``pymatting``; useful for weaker models but can soften strong masks.
        * **Edge decontamination** (default off, ``radius>0``) — corrects RGB
        bleed on fringe pixels. The per-pixel pass is slow; leave off unless a
        colour halo persists.
        * **Alpha erosion** (default off) — trims the outermost alpha layer as a
        last resort; can thin fine details such as hair.

        Args:
            source: Full-resolution source image (reference for alpha matting).
            mask: RGBA mask produced by rembg (already at full resolution).

        Returns:
            Refined RGBA image. With all stages disabled, *mask* is returned
            unchanged.
        """
        logger.debug(
            "refine_edges_start",
            alpha_matting=self.config.alpha_matting,
            alpha_edge_sharpening=self.config.alpha_edge_sharpening,
            erode_alpha=self.config.erode_alpha,
            edge_radius=self.config.edge_decontamination_radius,
        )

        if self.config.alpha_matting:
            try:
                refined = apply_alpha_matting(source, mask)
                if refined is not None:
                    mask = refined
            except Exception as exc:
                logger.warning("alpha_matting_failed", reason=str(exc))

        if self.config.alpha_edge_sharpening > 0:
            mask = sharpen_alpha(mask, strength=self.config.alpha_edge_sharpening)
            logger.debug("alpha_sharpened")

        if self.config.edge_decontamination_radius > 0:
            mask = decontaminate_edges(mask, radius=self.config.edge_decontamination_radius)
            logger.debug("decontamination_done")

        if self.config.erode_alpha:
            mask = erode_alpha(mask, pixels=1)
            logger.debug("erosion_done")

        return mask

    def _save(
        self,
        input_path: Path,
        image: Image.Image,
        output_path: Path,
    ) -> ProcessorResult:
        """Persist *image* to *output_path* as a PNG file.

        Args:
            input_path: Original source path (used for metadata and logging).
            image: Processed RGBA image ready for saving.
            output_path: Destination ``.png`` path.

        Returns:
            A successful :class:`ProcessorResult` with size metadata, or a
            failed one if writing raised an exception.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="PNG")

            if self.config.verify_integrity and not output_path.exists():
                raise OSError("Output file was not created")

        except Exception as exc:
            logger.exception("save_failed", input=str(input_path))
            return ProcessorResult(input_path, success=False, error=f"Failed to save output: {exc}")

        return ProcessorResult(
            input_path,
            output_path=output_path,
            success=True,
            metadata={
                "input_size": input_path.stat().st_size,
                "output_size": output_path.stat().st_size,
                "category": self.category.value,
            },
        )

    """Edge decontamination and alpha matting utilities for rembg output."""


def sharpen_alpha(image: Image.Image, strength: float = 0.5) -> Image.Image:
    """Solidify the foreground and crispen the alpha edge of an RGBA mask.

    BiRefNet/BRIA min-max-normalise their mask, so the *interior* of the
    subject lands around alpha 200–254 (≈0.8–0.99) instead of a solid 255 —
    the whole subject ends up faintly translucent ("esfumaçado"), and the
    upscaled edge carries a soft halo.

    This applies a "levels" remap: a transition band centred on 0.5 with
    half-width ``w = 0.5 * (1 - strength)``. Alpha above ``0.5 + w`` snaps to
    fully opaque (solid subject), alpha below ``0.5 - w`` snaps to fully
    transparent (halo removed), and the band between is smoothstepped to keep a
    thin anti-aliased edge.

    Args:
        image: RGBA image produced by rembg.
        strength: ``[0, 1]``. ``0`` is a no-op; ``0.5`` (default) snaps
            everything ≥0.75 to opaque and ≤0.25 to transparent while keeping a
            soft mid band; ``1`` is a hard binary cut at 0.5.

    Returns:
        A new RGBA ``Image`` with a solid interior and a crisp edge.
    """
    strength = float(max(0.0, min(1.0, strength)))
    if strength == 0.0:
        return image

    rgba = np.array(image)
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    half_band = 0.5 * (1.0 - strength)
    low, high = 0.5 - half_band, 0.5 + half_band

    if high > low:
        t = np.clip((alpha - low) / (high - low), 0.0, 1.0)
    else:  # strength == 1.0 → hard threshold at 0.5
        t = (alpha >= 0.5).astype(np.float32)

    # Smoothstep (3t² - 2t³) for a soft, non-jagged transition within the band.
    remapped = t * t * (3.0 - 2.0 * t)

    rgba[..., 3] = np.clip(remapped * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def decontaminate_edges(image: Image.Image, radius: int = 1) -> Image.Image:
    """Remove white/light color bleed from semi-transparent edge pixels.

    When a segmentation model is trained on white-background images, pixels
    near the subject boundary tend to have their RGB channels "contaminated"
    by the background color.  This function corrects those pixels by
    replacing each semi-transparent pixel's color with a weighted average
    of its fully-opaque neighbors, effectively pushing the background color
    out of the fringe.

    Args:
        image: RGBA image produced by rembg.
        radius: Neighbourhood radius (px) used when sampling opaque neighbors.
            ``1`` is sufficient for most halos; increase to ``2`` for thicker
            fringes at the cost of a small runtime hit.

    Returns:
        A new RGBA ``Image`` with edge colors corrected.
    """
    rgba = np.array(image, dtype=np.float32)          # (H, W, 4)
    rgb, alpha = rgba[..., :3], rgba[..., 3]

    opaque_mask = alpha > 240                          # fully-opaque pixels
    fringe_mask = (alpha > 0) & ~opaque_mask          # semi-transparent edge

    if not fringe_mask.any():
        return image                                   # nothing to fix

    corrected = rgb.copy()

    # For every fringe pixel, collect the mean color of opaque neighbors
    # within *radius* and use that as the "true" foreground color.
    ys, xs = np.where(fringe_mask)
    h, w = alpha.shape

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)

        neighbor_opaque = opaque_mask[y0:y1, x0:x1]
        if not neighbor_opaque.any():
            continue

        neighbor_rgb = rgb[y0:y1, x0:x1]
        corrected[y, x] = neighbor_rgb[neighbor_opaque].mean(axis=0)

    result = rgba.copy()
    result[..., :3] = corrected
    return Image.fromarray(result.astype(np.uint8), mode="RGBA")


def decontaminate_edges_wip(image: Image.Image, radius: int = 1) -> Image.Image:
    """Remove white/light color bleed from semi-transparent edge pixels.

    When a segmentation model is trained on white-background images, pixels
    near the subject boundary tend to have their RGB channels "contaminated"
    by the background color.  This function corrects those pixels by
    replacing each semi-transparent pixel's color with a weighted average
    of its fully-opaque neighbors, effectively pushing the background color
    out of the fringe.

    Args:
        image: RGBA image produced by rembg.
        radius: Neighbourhood radius (px) used when sampling opaque neighbors.
            ``1`` is sufficient for most halos; increase to ``2`` for thicker
            fringes at the cost of a small runtime hit.

    Returns:
        A new RGBA ``Image`` with edge colors corrected.
    """
    rgba  = np.array(image, dtype=np.float32)
    rgb   = rgba[..., :3]
    alpha = rgba[..., 3]

    opaque_mask = alpha > 240

    # Near-white fully-opaque pixels touching the edge of the subject
    # are likely background bleed even though alpha=255.
    channel_max  = rgb.max(axis=-1)
    channel_min  = rgb.min(axis=-1)
    is_near_white = (channel_min > 230) & ((channel_max - channel_min) < 15)

    # Build a boundary mask: opaque pixels adjacent to transparent ones.
    from scipy.ndimage import binary_dilation
    transparent_mask = alpha < 10
    boundary = binary_dilation(transparent_mask, iterations=radius) & opaque_mask

    fringe_mask = ((alpha > 0) & ~opaque_mask) | (boundary & is_near_white)

    if not fringe_mask.any():
        return image

    corrected = rgb.copy()
    ys, xs = np.where(fringe_mask)
    h, w = alpha.shape

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)

        # Only sample from opaque non-white neighbors as color reference
        neighbor_opaque    = opaque_mask[y0:y1, x0:x1]
        neighbor_not_white = ~is_near_white[y0:y1, x0:x1]
        good_neighbors     = neighbor_opaque & neighbor_not_white

        if good_neighbors.any():
            corrected[y, x] = rgb[y0:y1, x0:x1][good_neighbors].mean(axis=0)
        else:
            # No good neighbor found — make transparent instead
            rgba[y, x, 3] = 0

    result = rgba.copy()
    result[..., :3] = corrected
    return Image.fromarray(result.astype(np.uint8), mode="RGBA")

def erode_alpha(image: Image.Image, pixels: int = 1) -> Image.Image:
    """Shrink the alpha channel by *pixels* to trim stray fringe pixels.

    A one-pixel erosion removes the outermost layer of semi-transparent
    pixels that the segmentation mask often misclassifies as foreground.
    Use this as a last resort when :func:`decontaminate_edges` alone is
    insufficient — it is a destructive operation that can thin fine details
    such as hair or fur.

    Args:
        image: RGBA image produced by rembg.
        pixels: Number of pixels to erode.  Defaults to ``1``.

    Returns:
        A new RGBA ``Image`` with a slightly tighter alpha channel.
    """
    from scipy.ndimage import binary_erosion

    rgba = np.array(image)
    alpha = rgba[..., 3]

    binary = alpha > 128
    eroded = binary_erosion(binary, iterations=pixels)
    rgba[..., 3] = np.where(eroded, alpha, 0)

    return Image.fromarray(rgba, mode="RGBA")


def apply_alpha_matting(
    original: Image.Image,
    mask: Image.Image,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_structure_size: int = 10,
) -> Image.Image:
    """Refine a coarse rembg mask using alpha matting.

    Builds a trimap from the coarse mask by eroding the foreground region
    to create a definite-foreground band and dilating to create a
    definite-background band, with the transition zone left as unknown.

    Args:
        original: Full-resolution source image (RGB or RGBA).
        mask: Coarse RGBA mask from rembg.
        foreground_threshold: Alpha values above this are treated as
            definite foreground (0–255).
        background_threshold: Alpha values below this are treated as
            definite background (0–255).
        erode_structure_size: Radius of the morphological structuring element
            used to separate foreground/background trimap regions. Larger
            values widen the unknown band and can recover more detail but
            increase compute time.

    Returns:
        Full-resolution RGBA image with a refined, smooth alpha channel.

    Raises:
        ImportError: If ``pymatting`` is not installed.
    """
    try:
        from pymatting import estimate_alpha_lkm
    except ImportError as exc:
        raise ImportError("pymatting is required for alpha matting") from exc

    from scipy.ndimage import binary_dilation, binary_erosion

    original_size = original.size
    needs_resize = max(original_size) > MAX_INFERENCE_DIM

    if needs_resize:
        scale = MAX_INFERENCE_DIM / max(original_size)
        small_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        original_small = original.convert("RGB").resize(small_size, Image.LANCZOS)
        mask_small = mask.resize(small_size, Image.LANCZOS)
    else:
        original_small = original.convert("RGB")
        mask_small = mask

    source_np = np.array(original_small, dtype=np.float64) / 255.0
    alpha_np  = np.array(mask_small.getchannel("A"), dtype=np.float64) / 255.0

    binary    = alpha_np > 0.5
    structure = np.ones((erode_structure_size, erode_structure_size), dtype=bool)
    definite_fg = binary_erosion(binary,  structure=structure)
    definite_bg = ~binary_dilation(binary, structure=structure)

    trimap = np.full(alpha_np.shape, 0.5)
    trimap[definite_fg] = 1.0
    trimap[definite_bg] = 0.0

    refined_alpha = estimate_alpha_lkm(source_np, trimap)

    # Always composite the refined ALPHA onto the ORIGINAL RGB. Using
    # estimate_foreground_ml's recoloured RGB shifts the subject's true colours
    # and is a source of the "smudged" look, so we only borrow its alpha.
    alpha_img = Image.fromarray((refined_alpha * 255).astype(np.uint8), mode="L")

    if needs_resize:
        alpha_img = alpha_img.resize(original_size, Image.LANCZOS)

    r, g, b, _ = original.convert("RGBA").split()
    return Image.merge("RGBA", (r, g, b, alpha_img))