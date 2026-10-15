"""Configuration management for asset optimizer."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class AssetCategory(str, Enum):
    HUMANOID = "humanoid"
    CREATURE = "creature"
    SPIRIT = "spirit"
    ITEM = "item"
    PORTRAIT = "portrait"
    TOKEN = "token"
    GENERAL = "general"


class ClipModelChoice(str, Enum):
    """CLIP-family encoder used by the creature classifier.

    SigLIP 2 (``siglip2``) is the recommended default for fantasy artwork: it
    has the strongest zero-shot accuracy of the four and still fits a 12 GB
    consumer GPU comfortably. See the trade-off table in
    ``processors/creature_classification.py``.
    """

    SIGLIP2 = "siglip2"
    SIGLIP = "siglip"
    CLIP = "clip"
    METACLIP2 = "metaclip2"


class CaptionBackend(str, Enum):
    """Image-captioning backend used by the rename processor.

    ``blip`` is the lightweight default (Salesforce BLIP, ~1 GB) and biases
    captions via a conditional text prefix. ``qwen`` uses an instruction-tuned
    Qwen2.5-VL VLM that can be *told* to emit a short fantasy-creature name
    directly (e.g. ``gibbering-mouther`` instead of ``dolphin``), at the cost
    of a larger download and more VRAM/latency per image.
    """

    BLIP = "blip"
    QWEN = "qwen"


class RembgModel(str, Enum):
    """Available rembg models for background removal.

    Every value here is validated against ``rembg.sessions.sessions_class``
    in rembg 2.0.76 — do not add names the installed rembg cannot resolve,
    or ``new_session`` will silently fall back to ``u2net``.

    See more:
        https://github.com/danielgatis/rembg#available-models
    """

    # State-of-the-art general background removal by BRIA AI (RMBG-2.0).
    # Best default for fantasy artwork: humanoids, creatures, items, and
    # complex edges. Runs at a fixed 1024x1024 internally.
    BRIA_RMBG = "bria-rmbg"

    # Strong general segmentation. Good alternative to BRIA for creatures,
    # dragons, ghosts, fur, hair, wings, and transparency.
    BIREFNET_GENERAL = "birefnet-general"

    # Lighter BiRefNet variant — lower VRAM, slightly less edge fidelity.
    BIREFNET_GENERAL_LITE = "birefnet-general-lite"

    # Optimized for human portraits and facial subjects.
    # Excels at hair extraction and facial boundaries.
    BIREFNET_PORTRAIT = "birefnet-portrait"

    # Dichotomous Image Segmentation (DIS).
    # Optimized for a single salient object: items, products, weapons, armor.
    BIREFNET_DIS = "birefnet-dis"

    # High-resolution salient object detection BiRefNet variant.
    BIREFNET_HRSOD = "birefnet-hrsod"

    # Camouflaged object detection BiRefNet variant.
    BIREFNET_COD = "birefnet-cod"

    # BiRefNet trained on a massive dataset; broadest coverage, heaviest VRAM.
    BIREFNET_MASSIVE = "birefnet-massive"

    # Modern general-purpose segmentation. Lower VRAM than BiRefNet/BRIA;
    # good fallback when the heavier models preserve too much background.
    ISNET = "isnet-general-use"

    # Optimized for anime, manga, cel-shaded, and illustrated artwork.
    ISNET_ANIME = "isnet-anime"

    # Legacy high-quality general model. Stable but surpassed by ISNet/BiRefNet.
    U2NET = "u2net"

    # Lightweight U²-Net variant. Fastest/smallest; quality is secondary.
    U2NETP = "u2netp"

    # Human body segmentation, optimized for people and portraits.
    U2NET_HUMAN_SEG = "u2net_human_seg"

    # Clothing segmentation for fashion/apparel extraction and try-on.
    U2NET_CLOTH_SEG = "u2net_cloth_seg"

    # Lightweight human silhouette extraction. Fast and resource-efficient.
    SILUETA = "silueta"

    # Prompt-guided interactive segmentation. Best for assisted workflows.
    SAM = "sam"


class MediaFormat(str, Enum):
    """Supported media formats."""

    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"

    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"

    @property
    def is_image(self) -> bool:
        """True if this format is an image container."""
        return self in {MediaFormat.PNG, MediaFormat.JPEG, MediaFormat.JPG, MediaFormat.WEBP}

    @property
    def is_video(self) -> bool:
        """True if this format is a video container."""
        return self in {MediaFormat.MP4, MediaFormat.MKV, MediaFormat.WEBM}


class CompressionLevel(int, Enum):
    """Compression levels for media optimization."""

    LOSSLESS = 0
    HIGH_QUALITY = 1
    BALANCED = 2
    AGGRESSIVE = 3


class Config(BaseModel):
    """Main configuration for asset optimizer."""

    input_dir: Path = Field(default=Path("assets"), description="Input assets directory")
    output_dir: Path = Field(default=Path("output"), description="Output assets directory")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory for models")

    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    gpu_device: str = Field(default="cuda", description="GPU device identifier")
    batch_size: int = Field(default=1, description="Batch size for processing")

    caption_rename_enabled: bool = Field(default=False, description="Enable caption-based renaming")
    caption_backend: CaptionBackend = Field(
        default=CaptionBackend.QWEN,
        description="Captioning backend: 'blip' (light, prefix-biased) or 'qwen' "
        "(Qwen2.5-VL VLM that can name fantasy creatures directly). Defaults to "
        "'qwen' because BLIP mislabels fantasy creatures; it is only used as a "
        "fallback for files whose source name is not meaningful (see "
        "caption_preserve_source_names).",
    )
    caption_preserve_source_names: bool = Field(
        default=True,
        description="When true, files that already have a meaningful source name "
        "(e.g. 'Balhannoth', 'Brain in a Jar') keep that name (slugified) instead "
        "of being re-captioned. Only files with non-descriptive names — pure "
        "numeric IDs, hex hashes — are sent to the captioning backend. Set false "
        "to caption every file from image content regardless of its source name.",
    )
    caption_prompt: str = Field(
        default="heavy-armored-fighter-with-a-spear",
        description="Conditional prompt fed to BLIP; biases captions toward short "
        "fantasy descriptions. Stripped from the result before slugifying. "
        "Ignored when caption_backend='qwen' (use caption_vlm_prompt instead).",
    )
    caption_vlm_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="HuggingFace id of the Qwen2.5-VL model used when "
        "caption_backend='qwen'. The 3B model is a good speed/quality balance for "
        "per-image captioning; '7B-Instruct' is sharper but heavier.",
    )
    caption_vlm_prompt: str = Field(
        default=(
            "You are naming fantasy tabletop RPG artwork. In 2 to 4 words, give a "
            "short Dungeons & Dragons style name for the main subject. Prefer "
            "fantasy creatures (types of fantasy creatures: undead,fiend,celestial,construct,draconic,elemental"
            "aberration, fey, giant, human, dragonborn, elf, dwarf, tiefling, ooze, monstrosity)"
            "over real-world references. Reply with only the name as "
            "lowercase hyphen-separated words, no punctuation."
        ),
        description="Instruction sent to the Qwen2.5-VL captioner. Only used when "
        "caption_backend='qwen'.",
    )
    caption_vlm_namebank: bool = Field(
        default=True,
        description="When true (and caption_backend='qwen'), append the curated "
        "creature name bank (creature_names.json) to the prompt as a soft hint so "
        "Qwen prefers canonical monster-manual names. It may still invent a short "
        "name when nothing fits. Set false to let Qwen name freely.",
    )
    replica_hash_threshold: int = Field(
        default=5,
        ge=0,
        le=64,
        description="Max Hamming distance (out of 64) between perceptual hashes for "
        "two images to be treated as the same picture (replicas share a slug).",
    )
    slug_max_words: int = Field(
        default=5,
        ge=1,
        description="Maximum number of words kept in a generated caption slug.",
    )

    classification_enabled: bool = Field(
        default=False, description="Enable D&D creature classification"
    )
    clip_model: ClipModelChoice = Field(
        default=ClipModelChoice.SIGLIP2,
        description="CLIP-family encoder for zero-shot classification",
    )
    classification_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Accept the CLIP result outright at/above this confidence; "
        "below it, optionally consult the VLM fallback.",
    )
    classification_margin_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum top1-top2 probability gap; a close call is treated "
        "as uncertain even if top1 confidence is high.",
    )
    classification_temperature: float = Field(
        default=0.01,
        gt=0.0,
        description="Softmax temperature over candidate classes; ~0.01 mirrors "
        "CLIP's learned logit scale (~100). Lower = sharper distribution.",
    )
    classification_batch_size: int = Field(
        default=32, ge=1, description="Image batch size for the embedding pre-pass"
    )
    classification_sidecar: bool = Field(
        default=True,
        description="Write a <image>.classification.json sidecar per asset",
    )
    vlm_fallback_enabled: bool = Field(
        default=False,
        description="Use a VLM (Qwen2.5-VL) to resolve uncertain cases only",
    )
    vlm_model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="HuggingFace id of the VLM fallback model",
    )

    rembg_enabled: bool = Field(default=True, description="Enable background removal")
    rembg_model: RembgModel | None = Field(
        default=None, description="Explicit rembg model; None routes by rembg_category"
    )
    rembg_category: AssetCategory = Field(
        default=AssetCategory.GENERAL, description="Asset category for model selection"
    )

    cpu_fallback_on_oom: bool = Field(
        default=True,
        description="When the GPU runs out of VRAM (even after recreating the session), "
        "process that single image on CPU instead of failing it. Same quality, slower.",
    )
    gpu_mem_limit_gb: float = Field(
        default=0,
        description="Hard cap on the ONNXRuntime CUDA arena, in GiB. 0 disables the cap "
        "(recommended). Set e.g. 6 to leave headroom on an 8 GB laptop GPU.",
    )

    media_optimization_enabled: bool = Field(default=True, description="Enable media optimization")
    # Output formats are split by media class: image inputs are transcoded to
    # ``image_format`` and video inputs to ``video_format``. A single format
    # cannot serve both, since the processor routes by detected input type.
    image_format: MediaFormat = Field(
        default=MediaFormat.WEBP, description="Output format for image inputs"
    )
    video_format: MediaFormat = Field(
        default=MediaFormat.WEBM, description="Output format for video inputs"
    )
    compression_level: CompressionLevel = Field(
        default=CompressionLevel.BALANCED, description="Compression level"
    )

    recursive: bool = Field(default=True, description="Process directories recursively")
    skip_existing: bool = Field(default=True, description="Skip already processed files")
    verify_integrity: bool = Field(default=True, description="Verify output integrity")
    max_workers: int = Field(default=1, description="Maximum parallel workers")

    # Modern models (BiRefNet/BRIA) already produce a clean mask, so the heavy
    # refinement is OFF by default — it was the main source of the "smudged"
    # look. Prefer ``alpha_edge_sharpening`` for crisp edges; only enable
    # ``alpha_matting`` for hair-heavy portraits with weaker models.
    alpha_matting: bool = Field(
        default=False,
        description="Use pymatting to refine hair/fur edges. Slower; can soften edges on strong models.",
    )
    alpha_edge_sharpening: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Solidify the foreground and crispen the alpha edge via a levels remap "
        "around 0.5. 0 disables it; ~0.5 snaps the (faintly translucent) subject interior to "
        "fully opaque and removes the semi-transparent halo while keeping a thin anti-aliased "
        "edge; 1.0 is the hardest (near-binary) edge.",
    )
    edge_decontamination_radius: int = Field(
        default=0,
        description="Neighbourhood radius for RGB fringe decontamination. 0 disables the (slow, per-pixel) pass.",
    )
    erode_alpha: bool = Field(
        default=False,
        description="Trim one pixel from the alpha boundary. thins fine details for when decontamination alone is insufficient.",
    )

    output_suffix: str | None = Field(
        default="_optimized", description="Suffix for optimized files"
    )
    preserve_directory_structure: bool = Field(
        default=True, description="Preserve input directory structure in output"
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Validate max workers is positive."""
        if v < 1:
            raise ValueError("max_workers must be >= 1")
        return v

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, v: MediaFormat) -> MediaFormat:
        """Image inputs can only target an image container."""
        if not v.is_image:
            raise ValueError(f"image_format must be an image format, got {v.value!r}")
        return v

    @field_validator("video_format")
    @classmethod
    def validate_video_format(cls, v: MediaFormat) -> MediaFormat:
        """Video inputs can only target a video container."""
        if not v.is_video:
            raise ValueError(f"video_format must be a video format, got {v.value!r}")
        return v

    def model_post_init(self, __context) -> None:
        """Create necessary directories."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# BRIA RMBG-2.0 is the SOTA default for general subjects (creatures, humanoids,
# tokens); BiRefNet-DIS isolates single objects best (items); BiRefNet-Portrait
# wins on faces/hair.
MODEL_BY_CATEGORY = {
    AssetCategory.HUMANOID: RembgModel.BRIA_RMBG,
    AssetCategory.CREATURE: RembgModel.BRIA_RMBG,
    AssetCategory.SPIRIT: RembgModel.BRIA_RMBG,
    AssetCategory.ITEM: RembgModel.BIREFNET_DIS,
    AssetCategory.PORTRAIT: RembgModel.BIREFNET_PORTRAIT,
    AssetCategory.TOKEN: RembgModel.BRIA_RMBG,
    AssetCategory.GENERAL: RembgModel.BRIA_RMBG,
}

# Commits verified on 2026-10-15; pinning keeps downloads reproducible and stops a
# re-pushed ``main`` from swapping weights. Bump deliberately, together with a test run.
HF_REVISIONS: dict[str, str] = {
    "Salesforce/blip-image-captioning-base": "82a37760796d32b1411fe092ab5d4e227313294b",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "google/siglip-so400m-patch14-384": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "facebook/metaclip-2-worldwide-huge-quickgelu": "c139061af7b10fdb2e754b60d2b1182a3d5526c2",
}


def hf_revision(hf_id: str) -> str:
    """Pinned commit for a bundled model id; ids the user supplies track ``main``."""
    return HF_REVISIONS.get(hf_id, "main")
