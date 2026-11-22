"""``aforge process`` — full pipeline (background removal + media optimization)."""

from pathlib import Path

import click

from aforge_flow import __version__
from aforge_flow.cli._shared import _apply_auto_tune, logger
from aforge_flow.core.config import (
    AssetCategory,
    CaptionBackend,
    ClipModelChoice,
    CompressionLevel,
    Config,
    MediaFormat,
    RembgModel,
)
from aforge_flow.core.pipeline import AssetPipeline
from aforge_flow.utils import configure_logging


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="assets",
    help="Input directory containing assets",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory for processed assets",
)
@click.option(
    "--background-removal/--no-background-removal",
    default=True,
    help="Enable/disable background removal",
)
@click.option(
    "--media-optimization/--no-media-optimization",
    default=True,
    help="Enable/disable media optimization",
)
@click.option(
    "--category",
    type=click.Choice(["humanoid", "creature", "spirit", "item", "portrait", "token", "general"]),
    default="general",
    help="Asset category for model selection",
)
@click.option(
    "--model",
    type=click.Choice([m.value for m in RembgModel]),
    default=RembgModel.BRIA_RMBG.value,
    help="Background removal model (default selected per --category)",
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default="webp",
    help="Output format for image inputs",
)
@click.option(
    "--video-format",
    type=click.Choice(["webm", "mp4", "mkv"]),
    default="webm",
    help="Output format for video inputs",
)
@click.option(
    "--compression",
    type=click.Choice(["lossless", "high_quality", "balanced", "aggressive"]),
    default="balanced",
    help="Compression level",
)
@click.option(
    "--caption-rename/--no-caption-rename",
    "caption_rename",
    default=False,
    help="Rename images from AI-generated captions (disabled by default)",
)
@click.option(
    "--caption-backend",
    type=click.Choice([b.value for b in CaptionBackend]),
    default=CaptionBackend.QWEN.value,
    help="Captioning backend for non-meaningful names: 'qwen' (Qwen2.5-VL, names "
    "fantasy creatures directly) or 'blip' (light)",
)
@click.option(
    "--preserve-source-names/--caption-all-names",
    "preserve_source_names",
    default=True,
    help="Keep meaningful source filenames (slugified) and only caption files "
    "with junk names (numeric IDs, hashes); --caption-all-names re-captions every "
    "file from image content",
)
@click.option(
    "--classify/--no-classify",
    "classification",
    default=False,
    help="Enable creature classification (zero-shot CLIP, disabled by default)",
)
@click.option(
    "--clip-model",
    type=click.Choice([m.value for m in ClipModelChoice]),
    default=ClipModelChoice.SIGLIP2.value,
    help="CLIP-family encoder for classification",
)
@click.option(
    "--vlm-fallback/--no-vlm-fallback",
    default=False,
    help="Use a VLM to resolve uncertain classifications",
)
@click.option(
    "--taxonomy",
    default="dnd",
    help="Taxonomy profile for --classify/--caption-rename: a built-in name "
    "(dnd, mundane, call_of_cthulhu) or a path to a custom profile JSON",
)
@click.option(
    "--gpu/--cpu",
    default=True,
    help="Use GPU acceleration",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--auto-tune",
    is_flag=True,
    default=False,
    help="Auto-detect hardware + scan input to pick --gpu/--cpu and --workers "
    "(overrides those flags)",
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "tensorrt", "cuda", "cpu"]),
    default="cuda",
    help="ONNXRuntime execution provider for background removal (default 'cuda'). "
    "'auto' tries TensorRT→CUDA→CPU, but TensorRT FP16 mis-segments BRIA — see --fp16.",
)
@click.option(
    "--fp16/--no-fp16",
    default=False,
    help="Use FP16 in the TensorRT EP. Off by default: FP16 was observed to "
    "produce a broken (near-total) BRIA mask. Only affects 'auto'/'tensorrt'.",
)
@click.option(
    "--png-compress-level",
    type=click.IntRange(0, 9),
    default=6,
    help="PNG zlib level for background-removal output (lower = faster; lossless)",
)
@click.option(
    "--matte/--no-matte",
    default=True,
    help="Composite transparent (RGBA) inputs onto a white matte before "
    "segmentation. Fixes semi-transparent washes being kept as foreground.",
)
@click.option(
    "--dedup/--no-dedup",
    default=True,
    help="Skip files whose perceptual hash was already processed",
)
@click.option(
    "--ignore-dir",
    "ignore_dirs",
    multiple=True,
    help="Directory name to skip (repeatable; matched at any depth)",
)
@click.option(
    "--ignore-extension",
    "ignore_extensions",
    multiple=True,
    help="File extension to skip (repeatable). Empty by default; process transcodes "
    "videos, so they are kept unless excluded here.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip already processed files",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def process(
    input: str,
    output: str,
    background_removal: bool,
    media_optimization: bool,
    category: str,
    model: str,
    image_format: str,
    video_format: str,
    compression: str,
    caption_rename: bool,
    caption_backend: str,
    preserve_source_names: bool,
    classification: bool,
    clip_model: str,
    vlm_fallback: bool,
    taxonomy: str,
    gpu: bool,
    workers: int,
    auto_tune: bool,
    provider: str,
    fp16: bool,
    png_compress_level: int,
    matte: bool,
    dedup: bool,
    ignore_dirs: tuple,
    ignore_extensions: tuple,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Process assets with background removal and media optimization."""
    configure_logging(level=log_level)

    logger.info(
        "starting_aforge",
        input=input,
        output=output,
        version=__version__,
    )

    if auto_tune:
        gpu, workers = _apply_auto_tune(input, video_format, full_pipeline=True)

    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=background_removal,
        media_optimization_enabled=media_optimization,
        rembg_category=AssetCategory(category),
        rembg_model=RembgModel(model),
        image_format=MediaFormat(image_format),
        video_format=MediaFormat(video_format),
        compression_level=CompressionLevel[compression.upper()],
        caption_rename_enabled=caption_rename,
        caption_backend=CaptionBackend(caption_backend),
        caption_preserve_source_names=preserve_source_names,
        classification_enabled=classification,
        clip_model=ClipModelChoice(clip_model),
        vlm_fallback_enabled=vlm_fallback,
        taxonomy_profile=taxonomy,
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
        rembg_provider=provider,
        trt_fp16=fp16,
        matte_background=matte,
        png_compress_level=png_compress_level,
        dedup_enabled=dedup,
        ignore_dirs=list(ignore_dirs),
        ignore_extensions=list(ignore_extensions),
    )

    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(
            remove_background=background_removal,
            optimize_media=media_optimization,
        )
        pipeline.print_summary()
        logger.info("processing_complete")
        return
    except Exception as e:
        logger.error("processing_error", error=str(e))
        raise
