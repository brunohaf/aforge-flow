"""Image renaming processor using BLIP image captioning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import structlog
from PIL import Image

from asset_optimizer.config import Config
from asset_optimizer.processors.base import BaseProcessor, ProcessorResult
from asset_optimizer.utils import is_image, validate_file

if TYPE_CHECKING:
    from transformers import BlipForConditionalGeneration, BlipProcessor

logger = structlog.get_logger(__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
"""HuggingFace model ID. Switch to ``blip-image-captioning-large`` for
more descriptive captions at the cost of ~1 GB extra VRAM."""


def caption_to_slug(caption: str) -> str:
    """Convert a free-form caption string into a filesystem-safe slug.

    Args:
        caption: Raw caption text from the model.

    Returns:
        Lowercase hyphen-separated slug, e.g.
        ``"armored-knight-standing-with-sword-and-shield"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", caption.lower()).strip("-")


class CaptionRenameProcessor(BaseProcessor):
    """Rename images using AI-generated captions from BLIP.

    Loads a BLIP image-captioning model on initialisation and generates
    a descriptive slug for each image, renaming (or copying) it
    accordingly.  The original file is never deleted — the output is
    always a new file under ``config.output_dir``.

    Args:
        config: Global asset-optimizer configuration.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device = "cuda" if config.use_gpu else "cpu"
        self._model: Optional[BlipForConditionalGeneration] = None
        self._processor: Optional[BlipProcessor] = None
        logger.info("initializing_caption_processor", model=MODEL_ID, device=self.device)
        self._load_model()

    def _load_model(self) -> None:
        """Download (first run) and load the BLIP model onto :attr:`device`.

        The model is cached by HuggingFace in ``~/.cache/huggingface`` so
        subsequent runs are fully offline.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for caption renaming — "
                "install it with: pip install transformers"
            ) from exc

        self._processor = BlipProcessor.from_pretrained(MODEL_ID)
        self._model = (
            BlipForConditionalGeneration
            .from_pretrained(MODEL_ID)
            .to(self.device)
        )
        logger.info("caption_model_loaded", model=MODEL_ID, device=self.device)

    def can_process(self, input_path: Path) -> bool:
        """Return ``True`` when *input_path* is a supported image format.

        Args:
            input_path: Candidate file path.
        """
        return is_image(input_path)

    def process(self, input_path: Path) -> ProcessorResult:
        """Generate a caption for *input_path* and save a renamed copy.

        The source image is opened, captioned, and written to
        ``output_dir`` under a slug derived from the caption.  If two
        images produce the same slug a numeric suffix is appended to
        avoid collisions.

        Args:
            input_path: Path to the source image.

        Returns:
            A :class:`ProcessorResult` whose ``output_path`` is the
            renamed file and whose ``metadata`` contains the raw caption
            and generated slug.
        """
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)

        try:
            image = Image.open(input_path).convert("RGB")
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"Failed to open image: {exc}")

        try:
            caption = self._caption(image)
        except Exception as exc:
            logger.exception("captioning_failed", input=str(input_path))
            return ProcessorResult(input_path, success=False, error=f"Captioning failed: {exc}")

        slug = caption_to_slug(caption)
        output_path = self._unique_output_path(input_path, slug)

        if output_path.exists() and self.config.skip_existing:
            logger.info("skipping_existing", input=str(input_path), output=str(output_path))
            return ProcessorResult(
                input_path,
                output_path=output_path,
                success=True,
                metadata={"skipped": True, "caption": caption, "slug": slug},
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        except Exception as exc:
            logger.exception("save_failed", input=str(input_path))
            return ProcessorResult(input_path, success=False, error=f"Failed to save: {exc}")

        logger.info(
            "captioned_and_renamed",
            input=str(input_path),
            output=str(output_path),
            caption=caption,
        )
        return ProcessorResult(
            input_path,
            output_path=output_path,
            success=True,
            metadata={
                "caption": caption,
                "slug": slug,
                "input_size": input_path.stat().st_size,
                "output_size": output_path.stat().st_size,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _caption(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        """Run BLIP inference on *image* and return the raw caption string.

        Args:
            image: RGB PIL image to caption.
            max_new_tokens: Maximum number of tokens the model may generate.
                Higher values allow longer, more descriptive captions.

        Returns:
            Raw caption string, e.g.
            ``"a armored knight standing with a sword and shield"``.
        """
        import torch

        inputs = self._processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.decode(output[0], skip_special_tokens=True)

    def _unique_output_path(self, input_path: Path, slug: str) -> Path:
        """Return a collision-free output path for *slug*.

        If ``output_dir/slug.png`` already exists, appends ``-2``, ``-3``,
        etc. until a free name is found.

        Args:
            input_path: Original source path, used to mirror directory
                structure under ``output_dir``.
            slug: Filename slug derived from the caption.

        Returns:
            A ``Path`` that does not yet exist on disk.
        """
        relative = input_path.relative_to(self.config.input_dir)
        base = self.config.output_dir / relative.parent / slug
        candidate = base.with_suffix(".png")

        counter = 2
        while candidate.exists():
            candidate = base.with_name(f"{slug}-{counter}").with_suffix(".png")
            counter += 1

        return candidate