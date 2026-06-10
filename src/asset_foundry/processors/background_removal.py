"""Background removal processor using rembg."""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog
from PIL import Image

from asset_optimizer.config import MODEL_BY_CATEGORY, AssetCategory, Config
from asset_optimizer.processors.base import BaseProcessor, ProcessorResult
from asset_optimizer.utils import is_image, validate_file

if TYPE_CHECKING:
    from rembg import BaseSession

logger = structlog.get_logger(__name__)

MAX_INFERENCE_DIM: int = 1024
"""Images larger than this are downscaled before alpha matting and the
refined alpha is upscaled back. Prevents RAM exhaustion on large inputs."""

OOM_MARKERS = frozenset({"Failed to allocate memory", "CUDA out of memory"})
"""Substrings that identify GPU out-of-memory errors in ONNX/CUDA tracebacks."""


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
    * On OOM the image is retried at 75 % of its previous inference
      resolution, up to *max_retries* times.  Because the mask is always
      composited back onto the full-resolution original, this does not
      degrade output quality.

    Args:
        config: Global asset-optimizer configuration.
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
        self.providers: list[str] = (
            ["CUDAExecutionProvider"] if config.use_gpu else ["CPUExecutionProvider"]
        )

        logger.info("initializing_processor", model=self.model_name, category=self.category)
        self.session: BaseSession = self._create_session()


    def _create_session(self) -> BaseSession:
        """Initialise an rembg inference session for :attr:`model_name`.

        Returns:
            A ready-to-use rembg ``BaseSession``.

        Raises:
            Exception: Propagates any error raised by ``rembg.new_session``
                after logging it.
        """
        from rembg import new_session

        try:
            session = new_session(self.model_name, providers=self.providers)
        except Exception:
            logger.exception("session_initialization_failed", model=self.model_name)
            raise

        logger.info("session_initialized", model=self.model_name, providers=self.providers)
        return session

    def _flush_gpu_memory(self) -> None:
        """Release cached GPU allocations and run the Python garbage collector.

        Attempts to call ``torch.cuda.empty_cache()`` when PyTorch is
        installed; silently skips the CUDA flush otherwise.  Always calls
        ``gc.collect()`` regardless.
        """
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


    def _remove_with_retry(
        self,
        input_path: Path,
        source: Image.Image,
        output_path: Path,
        remove_fn,
        max_retries: int = 3,
    ) -> ProcessorResult:
        """Run background removal with automatic OOM recovery and edge refinement.

        On each OOM failure the inference dimension is reduced by 25 % and
        GPU memory is flushed before the next attempt.  The mask returned
        by rembg is always composited back onto the full-resolution *source*,
        so output quality is independent of the inference dimension chosen.

        After a successful segmentation, a three-stage edge refinement pipeline
        is applied:

        1. **Alpha matting** (optional) — refines soft edges such as hair and fur
        using ``pymatting``.
        2. **Edge decontamination** — replaces white/light color bleed on
        semi-transparent fringe pixels with the mean color of their opaque
        neighbours.
        3. **Alpha erosion** (optional) — trims one pixel from the alpha boundary
        to remove any remaining stray opaque pixels.

        Args:
            input_path: Original file path (used only for result metadata).
            source: Full-resolution source image.
            output_path: Destination path for the PNG output.
            remove_fn: Callable with the same signature as ``rembg.remove``.
            max_retries: Maximum number of attempts before giving up.

        Returns:
            A :class:`ProcessorResult` describing the outcome.
        """
        inference_dim = MAX_INFERENCE_DIM

        for attempt in range(1, max_retries + 1):
            inference_image, _ = resize_for_inference(source, inference_dim)

            try:
                with self._gpu_lock:
                    mask = remove_fn(inference_image, session=self.session)

                if inference_image.size != source.size:
                    mask = composite_mask(source, mask)

                mask = self._refine_edges(source, mask)

                return self._save(input_path, mask, output_path)

            except Exception as exc:
                is_oom = any(marker in str(exc) for marker in OOM_MARKERS)

                if is_oom and attempt < max_retries:
                    inference_dim = int(inference_dim * 0.75)
                    logger.warning(
                        "oom_retrying",
                        input=str(input_path),
                        attempt=attempt,
                        next_inference_dim=inference_dim,
                    )
                    self._flush_gpu_memory()
                    time.sleep(0.5 * attempt)
                    continue

                logger.exception("processing_failed", input=str(input_path))
                return ProcessorResult(
                    input_path,
                    success=False,
                    error=f"Background removal failed: {exc}",
                )

        return ProcessorResult(input_path, success=False, error="Max retries exceeded")

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
        """Apply the edge refinement pipeline to a freshly segmented mask.

        Runs up to three stages depending on configuration:

        * **Alpha matting** — recovers smooth, detail-preserving edges on hair,
        fur, and translucent objects by estimating true foreground/background
        colors in the transition zone.  Requires ``pymatting``.
        * **Edge decontamination** — corrects residual RGB color bleed on
        semi-transparent fringe pixels by replacing their color with the mean
        of their fully-opaque neighbours.
        * **Alpha erosion** — trims the outermost pixel layer from the alpha
        channel.  Useful as a last resort when a halo persists after the
        two stages above, but can thin fine details such as hair.

        Args:
            source: Full-resolution source image, used as the reference for
                alpha matting.
            mask: Coarse RGBA mask produced by rembg (already at full
                resolution after any composite step).

        Returns:
            Refined RGBA image.  When all stages are disabled in config, the
            input *mask* is returned unchanged.
        """
        logger.debug("refine_edges_start", alpha_matting=self.config.alpha_matting, erode_alpha=self.config.erode_alpha, edge_radius=self.config.edge_decontamination_radius)

        if self.config.alpha_matting:
            try:
                refined = apply_alpha_matting(source, mask)
                if refined is not None:
                    mask = refined
            except Exception as exc:
                logger.warning("alpha_matting_failed", reason=str(exc))

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
        from pymatting import estimate_alpha_lkm, estimate_foreground_ml, stack_images
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
    foreground    = estimate_foreground_ml(source_np, refined_alpha)
    result_np     = stack_images(foreground, refined_alpha)
    result        = Image.fromarray((result_np * 255).astype(np.uint8), mode="RGBA")

    if needs_resize:
        alpha_refined = result.getchannel("A").resize(original_size, Image.LANCZOS)
        r, g, b, _ = original.convert("RGBA").split()
        return Image.merge("RGBA", (r, g, b, alpha_refined))

    return result