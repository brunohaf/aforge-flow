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

class RembgModel(str, Enum):
    """Available rembg models for background removal
    See more:
        https://github.com/danielgatis/rembg#available-models
    """

    # Best overall quality for most workflows.
    # Recommended default for fantasy artwork, humanoids, creatures,
    # dragons, ghosts, fur, hair, wings, transparency, and complex edges.
    BIREFNET_GENERAL = "birefnet-general"

    # Optimized for human portraits and facial subjects.
    # Excels at hair extraction and facial boundaries but is less suitable
    # for creatures, items, and full-body fantasy compositions.
    BIREFNET_PORTRAIT = "birefnet-portrait"

    # Dichotomous Image Segmentation (DIS).
    # Optimized for extracting a single salient foreground object.
    # Particularly effective for items, products, weapons, armor, and props.
    BIREFNET_DIS = "birefnet-dis"

    # Modern general-purpose segmentation model with strong quality and
    # performance. Good fallback when BiRefNet preserves too much of
    # the background or visual effects.
    ISNET = "isnet-general-use"

    # Alias retained for backward compatibility.
    ISNET_GEN = "isnet-general-use"

    # Optimized for anime, manga, cel-shaded, and illustrated artwork.
    # Not recommended for realistic photography.
    ISNET_ANIME = "isnet-anime"

    # Legacy high-quality general segmentation model.
    # Stable and widely tested but generally surpassed by ISNet and
    # BiRefNet in quality, speed, and edge fidelity.
    U2NET = "u2net"

    # Lightweight U²-Net variant.
    # Fastest and smallest model; useful for low-resource environments
    # and bulk processing where quality is secondary.
    U2NETP = "u2netp"

    # Human body segmentation model optimized for people and portraits.
    # Not recommended for monsters, creatures, or items.
    U2NET_HUMAN_SEG = "u2net_human_seg"

    # Anime and manga segmentation model optimized for illustrated
    # characters and cel-shaded artwork.
    U2NET_ANIME = "u2net_anime"

    # Clothing segmentation model for fashion, apparel extraction,
    # garment analysis, and virtual try-on workflows.
    U2NET_CLOTH_SEG = "u2net_cloth_seg"

    # Lightweight human silhouette extraction model.
    # Fast and resource-efficient, suitable for portraits and simple
    # human subjects but less accurate on complex scenes.
    SILUETA = "silueta"

    # Salient object amodal segmentation model.
    # Attempts to recover the complete object shape even when portions
    # are partially occluded. Best for isolated objects and products.
    SALIENT_OBJECT_AMODAL = "salient_object_amodal"

    # Prompt-guided interactive segmentation model.
    # Best for manual or assisted workflows rather than fully automated
    # background removal pipelines.
    SAM = "sam"

    # High-quality Segment Anything variant with improved edge accuracy
    # and fine-detail preservation. Intended for premium manual editing
    # workflows and difficult segmentation tasks.
    SAM_HQ = "sam_hq"


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


class CompressionLevel(int, Enum):
    """Compression levels for media optimization."""

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

    # Background removal settings
    rembg_enabled: bool = Field(default=True, description="Enable background removal")
    rembg_model: RembgModel = Field(default=RembgModel.ISNET, description="Model for rembg")
    rembg_category: AssetCategory = Field(
        default=AssetCategory.GENERAL, description="Asset category for model selection"
    )

    # Media optimization settings
    media_optimization_enabled: bool = Field(default=True, description="Enable media optimization")
    output_format: MediaFormat = Field(default=MediaFormat.WEBP, description="Output format")
    compression_level: CompressionLevel = Field(
        default=CompressionLevel.BALANCED, description="Compression level"
    )

    # Processing settings
    recursive: bool = Field(default=True, description="Process directories recursively")
    skip_existing: bool = Field(default=True, description="Skip already processed files")
    verify_integrity: bool = Field(default=True, description="Verify output integrity")
    max_workers: int = Field(default=1, description="Maximum parallel workers")

    # Alfa and Edges
    alpha_matting: bool = Field(default=True, description="Use pymatting to refine hair/fur edges. Slower but much cleaner.")
    edge_decontamination_radius: int = Field(default=1, description="Neighbourhood radius for RGB fringe decontamination.")
    erode_alpha: bool = Field(default=False, description="Trim one pixel from the alpha boundary. thins fine details for when decontamination alone is insufficient.")

    # Output format
    output_suffix: Optional[str] = Field(
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

    def model_post_init(self, __context) -> None:
        """Create necessary directories."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Model selection mapping based on asset category
MODEL_BY_CATEGORY = {
    AssetCategory.HUMANOID: RembgModel.BIREFNET_GENERAL,
    AssetCategory.CREATURE: RembgModel.BIREFNET_GENERAL,
    AssetCategory.SPIRIT: RembgModel.BIREFNET_GENERAL,
    AssetCategory.ITEM: RembgModel.BIREFNET_DIS,
    AssetCategory.PORTRAIT: RembgModel.BIREFNET_PORTRAIT,
    AssetCategory.TOKEN: RembgModel.BIREFNET_GENERAL,
    AssetCategory.GENERAL: RembgModel.BIREFNET_GENERAL,
}