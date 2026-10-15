"""Image renaming processor using AI image captioning.

1. **Replica consistency** — images that are the *same picture* (e.g. the
   same artwork at a different resolution, or with a slightly dirtier
   background) are detected with a perceptual hash and share a single
   base slug.  The caption is generated **once per distinct picture**,
   so background spots / resolution no longer change the name.
2. **Simple, fantasy-style captions** — captioning is produced by a
   pluggable backend and trimmed to a short, filler-free slug such as
   ``bearded-scholar-golden-robes``:

   * ``blip`` — Salesforce BLIP conditioned on a fantasy text prefix
     (light, ~1 GB). Mundane words are remapped to fantasy creatures
     via :data:`_CREATURE_SYNONYMS`.
   * ``qwen`` — Qwen2.5-VL, an instruction-tuned VLM *told* to emit a
     short fantasy-creature name directly (e.g. ``gibbering-mouther``
     rather than ``dolphin``).
3. **Deterministic output** — sampling disabled with a fixed seed, so
   re-running on the same inputs reproduces the same names.

Replicas are numbered incrementally, keeping the source extension and
bytes: ``slug-1.png``, ``slug-2.png``, ... (or ``.jpg``, ``.webp``, ...).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from PIL import Image

from aforge_flow.config import CaptionBackend, Config, hf_revision
from aforge_flow.processors.base import BaseProcessor, ProcessorResult
from aforge_flow.utils import is_image, validate_file

logger = structlog.get_logger(__name__)

MODEL_ID = "Salesforce/blip-image-captioning-base"
"""HuggingFace model ID. Switch to ``blip-image-captioning-large`` for
more descriptive captions at the cost of ~1 GB extra VRAM."""


FANTASY_PROMPT = "ancient-red-dragon-breathing-fire"
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
_FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "with",
        "and",
        "is",
        "are",
        "this",
        "that",
        "his",
        "her",
        "their",
        "to",
        "in",
        "on",
        "at",
        "as",
        "it",
        "for",
        "person",
        "man",
        "woman",
        "character",
        "portrait",
        "image",
        "picture",
        "photo",
        "illustration",
        "artwork",
        "drawing",
        "render",
        "rendering",
        "there",
        "who",
        "wearing",
        "standing",
        "sitting",
        "some",
        "they",
    }
)

# Mundane caption words BLIP tends to emit, mapped to fantasy-creature
# equivalents so a "dolphin with tentacles" becomes an "aberration with
# tentacles" rather than a real-world animal. Values are already slug-safe
# (lowercase, hyphenated) and each counts as a single word toward the slug
# length limit. Extend via ``config`` is intentionally not supported — keep
# the vocabulary curated here so output stays consistent across runs.
_CREATURE_SYNONYMS = {
    "dolphin": "aberration",
    "whale": "aberration",
    "fish": "aberration",
    "shark": "aberration",
    "eel": "aberration",
    "octopus": "mind-flayer",
    "squid": "mind-flayer",
    "jellyfish": "gibbering-mouther",
    "slug": "gibbering-mouther",
    "worm": "gibbering-mouther",
    "lizard": "dragonkin",
    "crocodile": "dragonkin",
    "alligator": "dragonkin",
    "snake": "serpent",
    "turtle": "dragon-turtle",
    "frog": "slaad",
    "toad": "slaad",
    "dog": "hellhound",
    "wolf": "dire-wolf",
    "cat": "displacer-beast",
    "horse": "nightmare",
    "goat": "satyr",
    "bear": "owlbear",
    "rat": "wererat",
    "bat": "stirge",
    "ape": "ogre",
    "monkey": "imp",
    "bird": "harpy",
    "owl": "owlbear",
    "crow": "harpy",
    "raven": "harpy",
    "spider": "phase-spider",
    "scorpion": "phase-spider",
    "insect": "swarm",
    "soldier": "warrior",
    "knight": "paladin",
    "monster": "aberration",
    "creature": "aberration",
    "alien": "aberration",
}


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
    articles/filler words, translates mundane creature words into their
    fantasy equivalents (see :data:`_CREATURE_SYNONYMS`), keeps the first
    *max_words* meaningful tokens, then slugifies via
    :func:`caption_to_slug` — so ``"a dolphin with tentacles"`` becomes
    ``"aberration-tentacles"`` rather than ``"dolphin-tentacles"``.

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
        text = text[len(prefix) :].strip()

    words: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token in _FILLER_WORDS:
            continue
        words.append(_CREATURE_SYNONYMS.get(token, token))
        if len(words) >= max_words:
            break

    slug = caption_to_slug("-".join(words)) if words else caption_to_slug(caption)
    return slug


# Tokens the asset pipeline appends to filenames; carry no naming value and
# are stripped before deciding whether a source name is meaningful.
_PIPELINE_NOISE = frozenset(
    {
        "nobg",
        "no",
        "bg",
        "rembg",
        "cutout",
        "cutouts",
        "alpha",
        "transparent",
        "optimized",
        "optimised",
        "optim",
        "opt",
        "scaled",
        "upscaled",
        "resized",
        "final",
        "copy",
        "edit",
        "edited",
        "v1",
        "v2",
        "webp",
        "png",
        "jpg",
        "jpeg",
    }
)


def _looks_like_id(token: str) -> bool:
    """Return ``True`` if *token* is an opaque identifier rather than a word.

    Catches the two junk-name shapes seen in the asset library: long runs of
    digits (e.g. ``900297781783671446`` social-media IDs) and hex digests
    (e.g. ``af49b6a5f78957e448ce5b9d29a29369`` content hashes).

    Args:
        token: A single slug token (lowercase, no separators).

    Returns:
        ``True`` when the token looks machine-generated.
    """
    if len(token) >= 6 and token.isdigit():
        return True
    if len(token) >= 16 and all(c in "0123456789abcdef" for c in token):
        return True
    return False


def source_name_slug(
    stem: str,
    *,
    max_words: int = DEFAULT_SLUG_MAX_WORDS,
) -> str | None:
    """Derive a slug from an existing *meaningful* source filename.

    Slugifies *stem*, drops pipeline-noise suffixes (``_nobg``, ``_optimized``,
    …) and opaque identifiers (numeric IDs, hex hashes), then keeps the first
    *max_words* remaining tokens. Unlike :func:`simplify_caption`, the curated
    creature-synonym remapping is **not** applied — a hand-authored source name
    is taken at face value rather than second-guessed.

    Returns ``None`` when nothing word-like survives (the name was pure junk),
    signalling the caller to fall back to image captioning.

    Args:
        stem: The source filename without its extension.
        max_words: Maximum number of tokens to keep in the slug.

    Returns:
        A short slug such as ``"brain-in-a-jar"``, or ``None`` if the source
        name carries no meaningful words.
    """
    tokens = [t for t in caption_to_slug(stem).split("-") if t]
    kept = [t for t in tokens if t not in _PIPELINE_NOISE and not _looks_like_id(t)]

    # Require at least one genuine word so IDs like "900297781783671446" or
    # "af49b6a5..." (after noise stripping) fall through to captioning.
    if not any(any(c.isalpha() for c in t) and len(t) >= 3 for t in kept):
        return None

    words: list[str] = []
    for token in kept:
        if token in _FILLER_WORDS:
            continue
        words.append(token)
        if len(words) >= max_words:
            break

    slug = caption_to_slug("-".join(words))
    return slug or None


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
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
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


class _BlipBackend:
    """BLIP captioning conditioned on a fantasy text prefix.

    The conditional prompt is echoed back by BLIP, so it is also exposed as
    :attr:`slug_prefix` for the slug simplifier to strip.
    """

    name = "blip"

    def __init__(self, config: Config, device: str) -> None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for caption renaming — "
                "install it with: pip install transformers"
            ) from exc

        self.device = device
        self.slug_prefix: str = getattr(config, "caption_prompt", FANTASY_PROMPT)
        cache_dir = getattr(config, "cache_dir", None)
        kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
        revision = hf_revision(MODEL_ID)

        logger.info("loading_caption_model", backend=self.name, model=MODEL_ID, device=device)
        self._processor = BlipProcessor.from_pretrained(MODEL_ID, revision=revision, **kwargs)
        self._model = BlipForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=revision, **kwargs
        ).to(device)
        logger.info("caption_model_loaded", backend=self.name, model=MODEL_ID, device=device)

    def caption(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        """Run deterministic BLIP inference (beam search, fixed seed)."""
        import torch

        if self.slug_prefix:
            inputs = self._processor(image, text=self.slug_prefix, return_tensors="pt").to(
                self.device
            )
        else:
            inputs = self._processor(image, return_tensors="pt").to(self.device)

        torch.manual_seed(0)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        return self._processor.decode(output[0], skip_special_tokens=True)


def _append_namebank_hint(prompt: str) -> str:
    """Append the curated creature name bank to a Qwen prompt as a soft hint.

    The bank (see :mod:`aforge_flow.taxonomy`) is rendered as type-grouped
    lines and framed so Qwen *prefers* a canonical name but may still invent a
    short one when nothing fits. Falls back to the unmodified prompt if the
    name bank can't be loaded, so captioning never hard-fails on a bad file.

    Args:
        prompt: The base instruction from ``config.caption_vlm_prompt``.

    Returns:
        The prompt with the name-bank hint appended (or unchanged on error).
    """
    try:
        from aforge_flow.taxonomy import creature_name_prompt_block

        block = creature_name_prompt_block()
    except Exception:  # pragma: no cover - defensive: never break captioning
        logger.warning("namebank_unavailable_using_freeform_prompt")
        return prompt

    return (
        f"{prompt}\n\n"
        "Prefer the closest matching name from this reference list of known "
        "creatures (grouped by type). If the subject clearly matches one, reply "
        "with that exact name. Only if nothing fits, give a short 2-4 word name.\n"
        f"{block}"
    )


class _QwenBackend:
    """Qwen2.5-VL captioning — the VLM is *instructed* to emit a fantasy name.

    Reuses the same load pattern as the classifier's VLM fallback
    (``AutoModelForImageTextToText`` + 4-bit on CUDA when bitsandbytes is
    available). The instruction is not echoed back, so :attr:`slug_prefix`
    is empty.
    """

    name = "qwen"
    slug_prefix = ""

    def __init__(self, config: Config, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = device
        self._prompt: str = getattr(config, "caption_vlm_prompt", "").strip()
        if getattr(config, "caption_vlm_namebank", False):
            self._prompt = _append_namebank_hint(self._prompt)
        hf_id: str = getattr(config, "caption_vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct")
        cache_dir = getattr(config, "cache_dir", None)
        cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}

        kwargs: dict = dict(cache_kwargs)
        if device == "cuda":
            kwargs["dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            try:  # optional 4-bit so a 7B model fits a 12 GB GPU
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                logger.info("bitsandbytes_unavailable_loading_fp16")

        logger.info("loading_caption_model", backend=self.name, model=hf_id, device=device)
        revision = hf_revision(hf_id)
        self._processor = AutoProcessor.from_pretrained(hf_id, revision=revision, **cache_kwargs)
        self._model = AutoModelForImageTextToText.from_pretrained(
            hf_id, revision=revision, **kwargs
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()
        # Captioning is greedy (do_sample=False); drop the sampling defaults the
        # model ships with so transformers doesn't warn that they're ignored.
        for unused in ("temperature", "top_p", "top_k"):
            if hasattr(self._model.generation_config, unused):
                setattr(self._model.generation_config, unused, None)
        logger.info("caption_model_loaded", backend=self.name, model=hf_id, device=device)

    def caption(self, image: Image.Image, max_new_tokens: int = 24) -> str:
        """Ask Qwen2.5-VL for a short fantasy name (greedy, deterministic)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # Decode only the freshly generated tokens, not the echoed prompt.
        return self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()


def _make_backend(config: Config, device: str):
    """Construct the captioning backend selected by ``config.caption_backend``."""
    backend = getattr(config, "caption_backend", CaptionBackend.BLIP)
    if backend == CaptionBackend.QWEN:
        return _QwenBackend(config, device)
    return _BlipBackend(config, device)


@dataclass
class _ReplicaGroup:
    """In-memory record of a distinct picture and its replicas."""

    phash: int
    base_slug: str
    caption: str
    count: int = 0
    members: list[Path] = field(default_factory=list)


class CaptionRenameProcessor(BaseProcessor):
    """Rename images using AI-generated captions.

    Loads a captioning backend on initialisation — BLIP or Qwen2.5-VL,
    selected via ``config.caption_backend`` — and generates a descriptive
    slug for each *distinct* image, renaming (or copying) it accordingly.
    Replicas of the same picture share the slug and are numbered
    incrementally. The original file is never deleted, and the output keeps
    the source bytes and extension (no re-encoding).

    The constructor signature, :meth:`can_process`, :meth:`process`, and
    the returned :class:`ProcessorResult` (including its existing metadata
    keys) are unchanged from the original processor.

    Args:
        config: Global aforge_flow configuration. The following
            optional attributes are honoured when present (all fall back
            to sensible defaults, so existing configs keep working):
            ``caption_backend``, ``caption_prompt``, ``caption_vlm_model``,
            ``caption_vlm_prompt``, ``replica_hash_threshold``,
            ``slug_max_words``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device = "cuda" if config.use_gpu else "cpu"

        # Read via getattr so configs predating these fields keep working.
        self._hash_threshold: int = getattr(
            config, "replica_hash_threshold", DEFAULT_HASH_THRESHOLD
        )
        self._slug_max_words: int = getattr(config, "slug_max_words", DEFAULT_SLUG_MAX_WORDS)
        self._preserve_source_names: bool = getattr(config, "caption_preserve_source_names", True)

        self._groups: list[_ReplicaGroup] = []
        self._used_base_slugs: set[str] = set()

        # The captioning backend (BLIP/Qwen) is loaded lazily on first use, so a
        # run whose files all have meaningful source names never pays the VLM
        # download / VRAM cost.
        logger.info("initializing_caption_processor", device=self.device)
        self._backend = None  # type: ignore[assignment]

    def _get_backend(self):
        """Return the captioning backend, loading it on first use."""
        if self._backend is None:
            self._backend = _make_backend(self.config, self.device)
        return self._backend

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

        try:
            phash = perceptual_hash(image)
        except Exception as exc:
            logger.exception("hashing_failed", input=str(input_path))
            return ProcessorResult(input_path, success=False, error=f"Hashing failed: {exc}")

        group = self._match_group(phash)
        if group is None:
            # Prefer an already-meaningful source name over re-captioning the
            # pixels — a hand-authored "Balhannoth" beats a BLIP/Qwen guess.
            source_slug = (
                source_name_slug(input_path.stem, max_words=self._slug_max_words)
                if self._preserve_source_names
                else None
            )

            if source_slug is not None:
                caption = input_path.stem
                slug = source_slug
                name_source = "source"
            else:
                try:
                    caption = self._get_backend().caption(image)
                except Exception as exc:
                    logger.exception("captioning_failed", input=str(input_path))
                    return ProcessorResult(
                        input_path, success=False, error=f"Captioning failed: {exc}"
                    )

                slug = simplify_caption(
                    caption,
                    max_words=self._slug_max_words,
                    prompt_prefix=self._get_backend().slug_prefix,
                )
                name_source = "caption"

            base_slug = self._reserve_base_slug(slug)
            group = _ReplicaGroup(phash=phash, base_slug=base_slug, caption=caption)
            self._groups.append(group)
            logger.info(
                "new_picture_group",
                input=str(input_path),
                slug=base_slug,
                caption=caption,
                name_source=name_source,
            )
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
            # Copy the original bytes rather than re-encoding the in-memory
            # RGB image: preserves the source format, quality and metadata.
            shutil.copy2(input_path, output_path)
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

    def _match_group(self, phash: int) -> _ReplicaGroup | None:
        """Return the closest existing group within the hash threshold.

        Args:
            phash: Perceptual hash of the current image.

        Returns:
            The best-matching :class:`_ReplicaGroup`, or ``None`` if this is
            a previously unseen picture.
        """
        best: _ReplicaGroup | None = None
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

        Produces ``output_dir/<rel_parent>/<slug>-<index><ext>`` where
        ``<ext>`` is the *source* file's extension — renaming never changes
        the format. As a final safety net it bumps *index* if that exact
        file already exists on disk (e.g. left over from a previous run with
        ``skip_existing`` off).

        Args:
            input_path: Original source path, used to mirror directory
                structure under ``output_dir`` and to preserve the extension.
            slug: Base slug shared by the picture's replicas.
            index: 1-based replica number.

        Returns:
            A concrete output ``Path``.
        """
        relative = input_path.relative_to(self.config.input_dir)
        parent = self.config.output_dir / relative.parent
        suffix = input_path.suffix

        candidate = parent / f"{slug}-{index}{suffix}"
        bump = index
        while candidate.exists() and not self.config.skip_existing:
            bump += 1
            candidate = parent / f"{slug}-{bump}{suffix}"
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
