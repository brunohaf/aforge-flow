"""Image renaming processor using BLIP image captioning.

1. **Replica consistency** — images that are the *same picture* (e.g. the
   same artwork at a different resolution, or with a slightly dirtier
   background) are detected with a perceptual hash and share a single
   base slug.  The caption is generated **once per distinct picture**,
   so background spots / resolution no longer change the name.
2. **Simple, fantasy-style captions** — captioning is conditioned on a
   fantasy prompt and the result is trimmed to a short, filler-free slug
   such as ``bearded-scholar-golden-robes``.
3. **Deterministic output** — beam search with sampling disabled and a
   fixed seed, so re-running on the same inputs reproduces the same names.

Replicas are numbered incrementally: ``slug-1.png``, ``slug-2.png``, ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import structlog
from vtt_asset_forge.config import Config
from vtt_asset_forge.processors.base import BaseProcessor, ProcessorResult
from vtt_asset_forge.utils import is_image, validate_file
from PIL import Image

if TYPE_CHECKING:
    from transformers import BlipForConditionalGeneration, BlipProcessor

logger = structlog.get_logger(__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
"""HuggingFace model ID. Switch to ``blip-image-captioning-large`` for
more descriptive captions at the cost of ~1 GB extra VRAM."""

FANTASY_PROMPT = "a fantasy character portrait of"
"""Default conditional prompt fed to BLIP. Biases captions toward short,
fantasy-flavoured descriptions. The prefix is stripped from the result
before slugifying. Override per-run via ``config.caption_prompt``."""

DEFAULT_HASH_THRESHOLD = 5
"""Maximum Hamming distance (out of 64 bits) between two perceptual
hashes for the images to be treated as the *same picture*. Override via
``config.replica_hash_threshold``. 0 = byte-identical only; 8-10 tolerates
heavier recompression/cleanup."""

DEFAULT_SLUG_MAX_WORDS = 5
"""Maximum number of words kept in a generated slug, for short names.
Override via ``config.slug_max_words``."""

# Words removed from captions so slugs stay short and "fantasy-like".
_FILLER_WORDS = frozenset({
    "a", "an", "the", "of", "with", "and", "is", "are", "this", "that",
    "his", "her", "their", "to", "in", "on", "at", "as", "it", "for",
    "person", "man", "woman", "character", "portrait", "image", "picture",
    "photo", "illustration", "artwork", "drawing", "render", "rendering",
    "there", "who", "wearing", "standing", "sitting", "some", "they",
})


def caption_to_slug(caption: str) -> str:
    """Convert a free-form caption string into a filesystem-safe slug.

    Kept byte-for-byte compatible with the original implementation so any
    external caller importing this function is unaffected.

    Args:
        caption: Raw caption text from the model.

    Returns:
        Lowercase hyphen-separated slug, e.g.
        ``"armored-knight-standing-with-sword-and-shield"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", caption.lower()).strip("-")


def simplify_caption(
    caption: str,
    *,
    max_words: int = DEFAULT_SLUG_MAX_WORDS,
    prompt_prefix: str = "",
) -> str:
    """Trim a raw caption into a short, filler-free slug.

    Strips the conditional prompt prefix (BLIP echoes it back), drops
    articles/filler words, keeps the first *max_words* meaningful tokens,
    then slugifies via :func:`caption_to_slug`.

    Args:
        caption: Raw caption string from the model.
        max_words: Maximum number of meaningful words to keep.
        prompt_prefix: Conditional prompt that may prefix the caption.

    Returns:
        A short slug, e.g. ``"bearded-scholar-golden-robes"``.
    """
    text = caption.lower().strip()

    prefix = prompt_prefix.lower().strip()
    if prefix and text.startswith(prefix):
        text = text[len(prefix):].strip()

    words: List[str] = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token in _FILLER_WORDS:
            continue
        words.append(token)
        if len(words) >= max_words:
            break

    # Fall back to the raw caption if filtering removed everything.
    slug = caption_to_slug("-".join(words)) if words else caption_to_slug(caption)
    return slug


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit dHash (difference hash) for *image*.

    Pure-Pillow implementation — no extra dependencies. The hash is
    invariant to resolution and robust to minor background noise, which is
    exactly what's needed to recognise replicas of the same artwork.

    Args:
        image: Any PIL image (converted internally to grayscale).
        hash_size: Hash side length; 8 yields a 64-bit hash.

    Returns:
        The hash as an integer.
    """
    resized = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(resized.getdata())
    row_stride = hash_size + 1

    value = 0
    for row in range(hash_size):
        base = row * row_stride
        for col in range(hash_size):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Return the number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


@dataclass
class _ReplicaGroup:
    """In-memory record of a distinct picture and its replicas."""

    phash: int
    base_slug: str
    caption: str
    count: int = 0
    members: List[Path] = field(default_factory=list)


class CaptionRenameProcessor(BaseProcessor):
    """Rename images using AI-generated captions from BLIP.

    Loads a BLIP image-captioning model on initialisation and generates a
    descriptive slug for each *distinct* image, renaming (or copying) it
    accordingly. Replicas of the same picture share the slug and are
    numbered incrementally. The original file is never deleted — the
    output is always a new file under ``config.output_dir``.

    The constructor signature, :meth:`can_process`, :meth:`process`, and
    the returned :class:`ProcessorResult` (including its existing metadata
    keys) are unchanged from the original processor.

    Args:
        config: Global vtt_asset_forge configuration. The following
            optional attributes are honoured when present (all fall back
            to sensible defaults, so existing configs keep working):
            ``caption_prompt``, ``replica_hash_threshold``,
            ``slug_max_words``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device = "cuda" if config.use_gpu else "cpu"
        self._model: Optional[BlipForConditionalGeneration] = None
        self._processor: Optional[BlipProcessor] = None

        # New, all optional — pulled via getattr for config retrocompat.
        self._caption_prompt: str = getattr(config, "caption_prompt", FANTASY_PROMPT)
        self._hash_threshold: int = getattr(
            config, "replica_hash_threshold", DEFAULT_HASH_THRESHOLD
        )
        self._slug_max_words: int = getattr(
            config, "slug_max_words", DEFAULT_SLUG_MAX_WORDS
        )

        # Replica bookkeeping (persists across process() calls).
        self._groups: List[_ReplicaGroup] = []
        self._used_base_slugs: set[str] = set()

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
        """Caption *input_path* (or reuse a replica's caption) and save it.

        The source image is opened and perceptually hashed. If it matches a
        picture already seen this run, the existing base slug is reused and
        the replica counter is incremented; otherwise a fresh caption is
        generated. The file is written to ``output_dir`` as
        ``<slug>-<n>.png``.

        Args:
            input_path: Path to the source image.

        Returns:
            A :class:`ProcessorResult` whose ``output_path`` is the renamed
            file and whose ``metadata`` contains the raw caption, generated
            slug, replica index, and perceptual hash (plus the original
            size keys).
        """
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)

        try:
            image = Image.open(input_path).convert("RGB")
        except Exception as exc:
            return ProcessorResult(input_path, success=False, error=f"Failed to open image: {exc}")

        # Identify (or create) the picture group this image belongs to.
        try:
            phash = perceptual_hash(image)
        except Exception as exc:
            logger.exception("hashing_failed", input=str(input_path))
            return ProcessorResult(input_path, success=False, error=f"Hashing failed: {exc}")

        group = self._match_group(phash)
        if group is None:
            try:
                caption = self._caption(image)
            except Exception as exc:
                logger.exception("captioning_failed", input=str(input_path))
                return ProcessorResult(input_path, success=False, error=f"Captioning failed: {exc}")

            slug = simplify_caption(
                caption,
                max_words=self._slug_max_words,
                prompt_prefix=self._caption_prompt,
            )
            base_slug = self._reserve_base_slug(slug)
            group = _ReplicaGroup(phash=phash, base_slug=base_slug, caption=caption)
            self._groups.append(group)
            logger.info("new_picture_group", input=str(input_path), slug=base_slug, caption=caption)
        else:
            logger.info(
                "replica_detected",
                input=str(input_path),
                slug=group.base_slug,
                distance=hamming_distance(phash, group.phash),
            )

        group.count += 1
        group.members.append(input_path)
        index = group.count

        output_path = self._build_output_path(input_path, group.base_slug, index)

        if output_path.exists() and self.config.skip_existing:
            logger.info("skipping_existing", input=str(input_path), output=str(output_path))
            return ProcessorResult(
                input_path,
                output_path=output_path,
                success=True,
                metadata={
                    "skipped": True,
                    "caption": group.caption,
                    "slug": group.base_slug,
                    "replica_index": index,
                    "phash": f"{phash:016x}",
                },
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
            caption=group.caption,
            replica_index=index,
        )
        return ProcessorResult(
            input_path,
            output_path=output_path,
            success=True,
            metadata={
                "caption": group.caption,
                "slug": group.base_slug,
                "replica_index": index,
                "phash": f"{phash:016x}",
                "input_size": input_path.stat().st_size,
                "output_size": output_path.stat().st_size,
            },
        )

    def _caption(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        """Run deterministic BLIP inference on *image*.

        Captioning is conditioned on :attr:`_caption_prompt` and uses beam
        search with sampling disabled plus a fixed seed, so the same image
        always yields the same caption.

        Args:
            image: RGB PIL image to caption.
            max_new_tokens: Maximum number of tokens the model may generate.

        Returns:
            Raw caption string (may still include the prompt prefix).
        """
        import torch

        if self._caption_prompt:
            inputs = self._processor(
                image, text=self._caption_prompt, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self._processor(image, return_tensors="pt").to(self.device)

        torch.manual_seed(0)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,        # deterministic beam search
                do_sample=False,    # no randomness
            )
        return self._processor.decode(output[0], skip_special_tokens=True)

    def _match_group(self, phash: int) -> Optional[_ReplicaGroup]:
        """Return the closest existing group within the hash threshold.

        Args:
            phash: Perceptual hash of the current image.

        Returns:
            The best-matching :class:`_ReplicaGroup`, or ``None`` if this is
            a previously unseen picture.
        """
        best: Optional[_ReplicaGroup] = None
        best_distance = self._hash_threshold + 1
        for group in self._groups:
            distance = hamming_distance(phash, group.phash)
            if distance <= self._hash_threshold and distance < best_distance:
                best, best_distance = group, distance
        return best

    def _reserve_base_slug(self, slug: str) -> str:
        """Reserve a base slug, disambiguating across *distinct* pictures.

        Two different pictures can produce the same caption; this keeps
        their base slugs unique (``slug``, ``slug-alt2``, ...) so their
        numbered replicas never collide. Empty captions fall back to
        ``image``.

        Args:
            slug: Candidate base slug from the caption.

        Returns:
            A base slug not yet used by any other group this run.
        """
        base = slug or "image"
        candidate = base
        counter = 2
        while candidate in self._used_base_slugs:
            candidate = f"{base}-alt{counter}"
            counter += 1
        self._used_base_slugs.add(candidate)
        return candidate

    def _build_output_path(self, input_path: Path, slug: str, index: int) -> Path:
        """Build the numbered output path, mirroring source directories.

        Produces ``output_dir/<rel_parent>/<slug>-<index>.png``. As a final
        safety net it bumps *index* if that exact file already exists on
        disk (e.g. left over from a previous run with ``skip_existing`` off).

        Args:
            input_path: Original source path, used to mirror directory
                structure under ``output_dir``.
            slug: Base slug shared by the picture's replicas.
            index: 1-based replica number.

        Returns:
            A concrete output ``Path``.
        """
        relative = input_path.relative_to(self.config.input_dir)
        parent = self.config.output_dir / relative.parent

        candidate = (parent / f"{slug}-{index}").with_suffix(".png")
        bump = index
        while candidate.exists() and not self.config.skip_existing:
            bump += 1
            candidate = (parent / f"{slug}-{bump}").with_suffix(".png")
        return candidate

    def _unique_output_path(self, input_path: Path, slug: str) -> Path:
        """Backward-compatible helper retained from the original processor.

        Returns a collision-free ``output_dir/slug.png`` (appending ``-2``,
        ``-3`` on collision). The main pipeline now uses
        :meth:`_build_output_path`; this remains only so existing callers
        of ``_unique_output_path`` keep working.

        Args:
            input_path: Original source path.
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