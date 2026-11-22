"""Configuration management for asset optimizer."""

from enum import Enum
from pathlib import Path
from typing import Optional

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

    # Images
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"

    # Videos
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


class WebmCodec(str, Enum):
    """Video codec used inside a WebM container.

    ``AV1`` (SVT-AV1) compresses ~30% better than VP9 at equal quality and is
    decoded by every current browser/Electron-based VTT, but it **cannot encode
    an alpha channel** — SVT-AV1 fed ``yuva420p`` emits a corrupt file. ``VP9``
    is the universal WebM baseline and is the *only* encoder that carries alpha
    (stored as a separate plane, flagged ``alpha_mode=1`` on the container).

    ``AUTO`` therefore picks per file: VP9 whenever the source has alpha, AV1
    otherwise. Force ``VP9`` if your target player predates AV1.
    """

    AUTO = "auto"
    VP9 = "vp9"
    AV1 = "av1"


class CompressionLevel(int, Enum):
    """Compression levels for media optimization.

    ``LOSSLESS`` means *truly* lossless for images (WebP ``lossless=True``) but
    **visually** lossless for video (a very low CRF), not bit-exact. Bit-exact
    video (``vpx -lossless 1``) is counterproductive here: asset sources are
    already lossy (h264/VP9), so it faithfully preserves their compression
    artifacts while producing files 5-20x *larger* than the input. A low-CRF
    constant-quality encode is what "lossless as far as possible" means for video.
    """

    LOSSLESS = 0
    HIGH_QUALITY = 1
    BALANCED = 2
    AGGRESSIVE = 3


class Config(BaseModel):
    """Main configuration for asset optimizer."""

    # Paths
    input_dir: Path = Field(default=Path("assets"), description="Input assets directory")
    output_dir: Path = Field(default=Path("output"), description="Output assets directory")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory for models")

    # GPU settings
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
        default="",
        description="Override the conditional prompt fed to BLIP (biases captions; "
        "stripped before slugifying). Empty means use the active taxonomy "
        "profile's caption_prefix. Ignored when caption_backend='qwen'.",
    )
    caption_vlm_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="HuggingFace id of the Qwen2.5-VL model used when "
        "caption_backend='qwen'. The 3B model is a good speed/quality balance for "
        "per-image captioning; '7B-Instruct' is sharper but heavier.",
    )
    caption_vlm_prompt: str = Field(
        default="",
        description="Override the instruction sent to the Qwen2.5-VL captioner. "
        "Empty means use the active taxonomy profile's caption_instruction "
        "(profile-aware, e.g. fantasy names for 'dnd', plain names for 'mundane'). "
        "Only used when caption_backend='qwen'.",
    )
    caption_vlm_namebank: bool = Field(
        default=True,
        description="When true (and caption_backend='qwen'), append the active "
        "taxonomy profile's name bank to the prompt as a soft hint so Qwen prefers "
        "canonical names. Profiles without a name bank caption freely. Set false "
        "to always let Qwen name freely.",
    )
    taxonomy_profile: str = Field(
        default="dnd",
        description="Taxonomy profile driving classification labels/prompts and "
        "the caption name-bank/instruction. A built-in name ('dnd', 'mundane', "
        "'call_of_cthulhu') or a path to a custom profile JSON.",
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

    # Creature classification settings (zero-shot CLIP-family)
    classification_enabled: bool = Field(
        default=False,
        description="Enable creature classification (taxonomy-driven; see taxonomy_profile)",
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

    # Background removal settings
    rembg_enabled: bool = Field(default=True, description="Enable background removal")
    rembg_model: RembgModel = Field(default=RembgModel.BRIA_RMBG, description="Model for rembg")
    rembg_category: AssetCategory = Field(
        default=AssetCategory.GENERAL, description="Asset category for model selection"
    )

    # Input conditioning for background removal.
    #
    # Many assets are RGBA with a semi-transparent painted wash over a
    # transparent background. rembg feeds the model ``img.convert("RGB")``, which
    # DROPS the alpha channel rather than compositing it, so the wash is handed to
    # the segmenter at full strength and kept as foreground. Flattening onto a
    # solid matte first makes the model correctly drop it.
    matte_background: bool = Field(
        default=True,
        description="Composite transparent (RGBA) inputs onto a solid matte before "
        "segmentation. Fixes semi-transparent washes being retained as foreground. "
        "Opaque inputs are unaffected.",
    )
    matte_color: tuple[int, int, int] = Field(
        default=(255, 255, 255),
        description="RGB matte colour used when matte_background is on. White suits "
        "art-on-paper assets and is the background the segmenter expects.",
    )

    # GPU memory / robustness. Applies to both GPU stages: rembg inference and
    # NVENC video encoding. In each case one uncooperative file falls back to CPU
    # rather than failing, so a batch never cascade-fails on a transient GPU issue.
    cpu_fallback_on_oom: bool = Field(
        default=True,
        description="Fall back to CPU for a single file when the GPU refuses it — VRAM "
        "exhaustion during rembg, or a failed NVENC encode session — instead of failing "
        "that file. Same or better quality, just slower.",
    )
    gpu_mem_limit_gb: float = Field(
        default=0,
        description="Hard cap on the ONNXRuntime CUDA arena, in GiB. 0 disables the cap "
        "(recommended). Set e.g. 6 to leave headroom on an 8 GB laptop GPU.",
    )

    # Execution provider / acceleration for background removal.
    #
    # Default is plain CUDA. The TensorRT FP16 engine for BRIA/BiRefNet was found
    # to produce a BROKEN mask (near-total foreground — FP16 attention overflow),
    # so 'auto'/TensorRT are opt-in only and FP16 is off by default. 'auto' still
    # prepends TensorRT→CUDA→CPU for anyone who explicitly wants to try it; ORT
    # silently skips TRT if its EP is unavailable.
    rembg_provider: str = Field(
        default="cuda",
        description="ONNXRuntime execution provider for rembg: 'cuda' (default, reliable), "
        "'auto' (TensorRT→CUDA→CPU), 'tensorrt', or 'cpu'. TensorRT FP16 mis-segments "
        "BRIA — leave on 'cuda' unless you have verified a TRT engine on your GPU.",
    )
    trt_fp16: bool = Field(
        default=False,
        description="Enable FP16 in the TensorRT EP. Faster and halves VRAM but was "
        "observed to produce a broken (near-total) BRIA mask via FP16 overflow. Keep "
        "false; only relevant when provider is 'auto'/'tensorrt'.",
    )
    trt_engine_cache: bool = Field(
        default=True,
        description="Cache compiled TensorRT engines under cache_dir/trt_engines so the "
        "one-time (minutes-long) engine build only happens once.",
    )
    cudnn_algo_exhaustive: bool = Field(
        default=False,
        description="Use EXHAUSTIVE cuDNN conv-algorithm search on the CUDA provider. "
        "Faster steady-state inference at the cost of more workspace VRAM and a one-time "
        "search; worth it for big homogeneous batches. Ignored when TensorRT is active.",
    )

    # Media optimization settings
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
    webm_codec: WebmCodec = Field(
        default=WebmCodec.AUTO,
        description="Video codec inside WebM: 'auto' (AV1, or VP9 when the source "
        "has alpha), 'vp9' (universal, alpha-capable), or 'av1' (best compression, "
        "no alpha support). Alpha sources are always encoded with VP9 regardless.",
    )
    passthrough_matching_format: bool = Field(
        default=True,
        description="Copy a file verbatim when it is already in the target format "
        "(e.g. a .webp input with image_format=webp) instead of re-encoding it. "
        "Re-encoding would be pure generational loss at any lossy level.",
    )
    keep_smallest_output: bool = Field(
        default=True,
        description="If a transcode comes out LARGER than its source, throw it away and "
        "copy the original instead (keeping its extension). Guarantees the run can never "
        "inflate the library. This fires on sources that are already efficiently "
        "compressed — e.g. a well-encoded h264 mp4, which AV1 cannot beat without "
        "discarding quality. Set false to force the target format unconditionally.",
    )
    nvenc_max_sessions: int = Field(
        default=3,
        ge=1,
        description="Max concurrent NVENC encodes. Consumer GeForce drivers cap "
        "simultaneous encode sessions; exceeding the cap fails with "
        "'OpenEncodeSessionEx failed'. Ignored when encoding on CPU.",
    )
    ffmpeg_timeout_seconds: int = Field(
        default=3600, gt=0, description="Per-file timeout for an ffmpeg transcode."
    )

    # Processing settings
    recursive: bool = Field(default=True, description="Process directories recursively")
    skip_existing: bool = Field(default=True, description="Skip already processed files")
    verify_integrity: bool = Field(default=True, description="Verify output integrity")
    max_workers: int = Field(default=1, description="Maximum parallel workers")
    dedup_enabled: bool = Field(
        default=True,
        description="Skip files whose perceptual hash was already processed. Disable to "
        "avoid hashing every input on large batches where duplicates are not a concern.",
    )

    # File discovery filters (applied once in get_files_recursively).
    ignore_dirs: list[str] = Field(
        default_factory=list,
        description="Directory names to exclude from discovery, matched at any depth "
        "(e.g. 'archive' skips every folder named 'archive' under the input dir).",
    )
    ignore_extensions: list[str] = Field(
        default_factory=list,
        description="File extensions to exclude from discovery (e.g. '.mp4'). Normalized "
        "to lowercase with a leading dot; values with or without a leading dot are accepted.",
    )

    # PNG save speed (background-removal output). zlib level only trades CPU time for
    # file size — it never changes decoded pixels, so output stays lossless at any level.
    png_compress_level: int = Field(
        default=6,
        ge=0,
        le=9,
        description="PNG zlib compression level for background-removal output. Lower is "
        "faster and larger (still lossless); 1 is a good choice for big batches.",
    )

    # Alpha and Edges
    #
    # Modern models (BiRefNet/BRIA) already produce a clean mask, so the heavy
    # refinement is OFF by default — it was the main source of the "smudged"
    # look. Prefer ``alpha_edge_sharpening`` for crisp edges; only enable
    # ``alpha_matting`` for hair-heavy portraits with weaker models.
    alpha_matting: bool = Field(default=False, description="Use pymatting to refine hair/fur edges. Slower; can soften edges on strong models.")
    alpha_edge_sharpening: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Solidify the foreground and crispen the alpha edge via a levels remap "
        "around 0.5. 0 disables it; ~0.5 snaps the (faintly translucent) subject interior to "
        "fully opaque and removes the semi-transparent halo while keeping a thin anti-aliased "
        "edge; 1.0 is the hardest (near-binary) edge.",
    )
    edge_decontamination_radius: int = Field(default=0, description="Neighbourhood radius for RGB fringe decontamination. 0 disables the (slow, per-pixel) pass.")
    erode_alpha: bool = Field(default=False, description="Trim one pixel from the alpha boundary. thins fine details for when decontamination alone is insufficient.")

    # Output format
    output_suffix: Optional[str] = Field(
        default="_optimized", description="Suffix for optimized files"
    )
    preserve_directory_structure: bool = Field(
        default=True, description="Preserve input directory structure in output"
    )

    @field_validator("input_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def strip_path_whitespace(cls, v):
        """Trim surrounding whitespace from directory paths.

        A trailing space is silently destructive on Windows: ``mkdir`` drops it, so
        ``output\\optimized `` creates ``output\\optimized`` and looks fine — but any
        *subdirectory* under it resolves as ``output\\optimized \\sub``, which
        Windows will not open (trailing spaces are only stripped from a path's final
        component). Every file then dies with WinError 3. It is never intentional,
        and it is trivially introduced by a quoted CLI argument.
        """
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, Path):
            return Path(str(v).strip())
        return v

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


# Model selection mapping based on asset category.
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