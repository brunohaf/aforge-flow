"""Command-line interface for AForge Flow (``aforge``)."""

from pathlib import Path

import click

from aforge_flow import __version__
from aforge_flow.config import (
    AssetCategory,
    ClipModelChoice,
    CompressionLevel,
    Config,
    MediaFormat,
    RembgModel,
)
from aforge_flow.pipeline import AssetPipeline
from aforge_flow.utils import (
    as_dict,
    build_command_line,
    configure_logging,
    detect_hardware,
    get_logger,
    recommend_settings,
    scan_workload,
)

logger = get_logger("aforge-cli")


def _apply_auto_tune(
    input_dir: str,
    video_format: str,
    full_pipeline: bool,
) -> tuple[bool, int]:
    """Detect hardware + scan the input dir and return recommended (use_gpu, workers).

    Logs the chosen settings and the reasoning so the override is visible.
    """
    hw = detect_hardware()
    workload = scan_workload(Path(input_dir))
    rec = recommend_settings(hw, workload, video_format, full_pipeline=full_pipeline)
    logger.info(
        "auto_tune",
        use_gpu=rec.use_gpu,
        workers=rec.workers,
        notes=rec.notes,
        warnings=rec.warnings,
    )
    return rec.use_gpu, rec.workers

# Make ``-h`` an alias for ``--help`` everywhere. Click propagates a parent
# context's ``help_option_names`` to every subcommand, so setting it once on
# the group covers ``aforge -h`` and ``aforge <command> -h`` alike.
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="aforge")
def main() -> None:
    """aforge — AForge Flow: GPU-accelerated tabletop asset pipeline.

    Batch-prepare RPG/VTT artwork: strip backgrounds, transcode images and
    video to web-friendly formats, and classify creatures into a D&D taxonomy.

    \b
    Commands:
      process            Run the full pipeline (background removal + media + classify)
      remove-background  Strip image backgrounds with rembg
      optimize-media     Transcode images/video to optimized formats
      classify           Tag images with a D&D creature taxonomy (CLIP/VLM)
      recommend          Suggest --workers/--gpu for this machine + workload
      info               Show version, PyTorch, and CUDA/GPU details

    \b
    Examples:
      aforge recommend -i assets --video-format webm
      aforge process -i assets -o output --auto-tune
      aforge optimize-media -i assets -o output --image-format webp --video-format webm
      aforge classify -i assets --clip-model siglip2 --vlm-fallback

    Run ``aforge <command> -h`` for per-command options.
    """
    pass


@main.command()
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
    "--classification/--no-classification",
    default=False,
    help="Enable D&D creature classification (zero-shot CLIP)",
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
    classification: bool,
    clip_model: str,
    vlm_fallback: bool,
    gpu: bool,
    workers: int,
    auto_tune: bool,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Process assets with background removal and media optimization."""
    # Configure logging
    configure_logging(level=log_level)

    logger.info(
        "starting_aforge",
        input=input,
        output=output,
        version=__version__,
    )

    if auto_tune:
        gpu, workers = _apply_auto_tune(input, video_format, full_pipeline=True)

    # Create config
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
        classification_enabled=classification,
        clip_model=ClipModelChoice(clip_model),
        vlm_fallback_enabled=vlm_fallback,
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
    )

    # Create and run pipeline
    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(
            remove_background=background_removal,
            optimize_media=media_optimization,
        )
        pipeline.print_summary()
        logger.info("processing_complete")
    except Exception as e:
        logger.error("processing_error", error=str(e))
        raise


@main.command()
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
def remove_background(
    input: str,
    output: str,
    category: str,
    model: str,
    gpu: bool,
    workers: int,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Remove backgrounds from images using rembg."""
    # Configure logging
    configure_logging(level=log_level)

    logger.info(
        "starting_background_removal",
        input=input,
        output=output,
        category=category,
        model=model,
    )

    # Create config
    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=True,
        media_optimization_enabled=False,
        rembg_category=AssetCategory(category),
        rembg_model=RembgModel(model),
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
    )

    # Create and run pipeline
    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=True, optimize_media=False)
        pipeline.print_summary()
        logger.info("background_removal_complete")
    except Exception as e:
        logger.error("background_removal_error", error=str(e))
        raise


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="assets",
    help="Input directory containing media",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory for optimized media",
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
    "--gpu/--cpu",
    default=True,
    help="Use GPU acceleration (AV1 NVENC for WebM; falls back to CPU if unavailable)",
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
def optimize_media(
    input: str,
    output: str,
    image_format: str,
    video_format: str,
    compression: str,
    gpu: bool,
    workers: int,
    auto_tune: bool,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Optimize and transcode media files to WebM/WebP."""
    # Configure logging
    configure_logging(level=log_level)

    logger.info(
        "starting_media_optimization",
        input=input,
        output=output,
        image_format=image_format,
        video_format=video_format,
    )

    if auto_tune:
        gpu, workers = _apply_auto_tune(input, video_format, full_pipeline=False)

    # Create config
    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=True,
        image_format=MediaFormat(image_format),
        video_format=MediaFormat(video_format),
        compression_level=CompressionLevel[compression.upper()],
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
    )

    # Create and run pipeline
    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=False, optimize_media=True)
        pipeline.print_summary()
        logger.info("media_optimization_complete")
    except Exception as e:
        logger.error("media_optimization_error", error=str(e))
        raise


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="assets",
    help="Input directory containing images",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory (sidecars are written next to inputs)",
)
@click.option(
    "--clip-model",
    type=click.Choice([m.value for m in ClipModelChoice]),
    default=ClipModelChoice.SIGLIP2.value,
    help="CLIP-family encoder for classification",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.85,
    help="Accept CLIP result at/above this confidence",
)
@click.option(
    "--vlm-fallback/--no-vlm-fallback",
    default=False,
    help="Use a VLM to resolve uncertain classifications",
)
@click.option(
    "--vlm-model",
    type=str,
    default="Qwen/Qwen2.5-VL-7B-Instruct",
    help="HuggingFace id of the VLM fallback model",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Image batch size for the embedding pre-pass",
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
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
def classify(
    input: str,
    output: str,
    clip_model: str,
    confidence_threshold: float,
    vlm_fallback: bool,
    vlm_model: str,
    batch_size: int,
    gpu: bool,
    workers: int,
    log_level: str,
) -> None:
    """Classify images into D&D creature taxonomy from image content alone."""
    configure_logging(level=log_level)

    logger.info(
        "starting_classification",
        input=input,
        output=output,
        clip_model=clip_model,
        vlm_fallback=vlm_fallback,
    )

    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=False,
        classification_enabled=True,
        clip_model=ClipModelChoice(clip_model),
        classification_confidence_threshold=confidence_threshold,
        classification_batch_size=batch_size,
        vlm_fallback_enabled=vlm_fallback,
        vlm_model=vlm_model,
        use_gpu=gpu,
        max_workers=workers,
    )

    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=False, optimize_media=False)
        pipeline.print_summary()
        logger.info("classification_complete")
    except Exception as e:
        logger.error("classification_error", error=str(e))
        raise


@main.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="assets",
    help="Input directory to scan for the recommendation",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory (used in the suggested command line)",
)
@click.option(
    "--video-format",
    type=click.Choice(["webm", "mp4", "mkv"]),
    default="webm",
    help="Planned output format for video inputs",
)
@click.option(
    "--full-pipeline/--media-only",
    default=False,
    help="Tune for the full 'process' pipeline (bg-removal + classification) vs "
    "media optimization only",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON")
def recommend(
    input: str,
    output: str,
    video_format: str,
    full_pipeline: bool,
    as_json: bool,
) -> None:
    """Recommend --workers and --gpu/--cpu for this machine and input."""
    import json as _json

    hw = detect_hardware()
    workload = scan_workload(Path(input))
    rec = recommend_settings(hw, workload, video_format, full_pipeline=full_pipeline)
    command = "process" if full_pipeline else "optimize-media"
    cmdline = build_command_line(command, input, output, video_format, rec)

    if as_json:
        payload = as_dict(hw, workload, rec)
        payload["command_line"] = cmdline
        click.echo(_json.dumps(payload, indent=2))
        return

    gpu_str = "Yes" if hw.has_cuda else "No"
    vram = (
        f"{hw.free_vram_gb:.1f} GB free / {hw.total_vram_gb:.1f} GB total"
        if hw.total_vram_gb is not None
        else "n/a"
    )
    click.echo("\n" + "=" * 60)
    click.echo("AFORGE - PERFORMANCE RECOMMENDATION")
    click.echo("=" * 60)
    click.echo("Hardware:")
    click.echo(f"  CPU cores:      {hw.physical_cores} physical / {hw.logical_cores} logical")
    click.echo(f"  RAM:            {hw.available_ram_gb:.1f} GB free / {hw.total_ram_gb:.1f} GB total")
    click.echo(f"  GPU (CUDA):     {gpu_str}" + (f"  ({hw.gpu_name})" if hw.gpu_name else ""))
    click.echo(f"  VRAM:           {vram}")
    click.echo(f"  av1_nvenc:      {'Yes' if hw.has_av1_nvenc else 'No'}")
    click.echo("Workload:")
    click.echo(f"  Images:         {workload.num_images}")
    click.echo(
        f"  Videos:         {workload.num_videos}"
        + (
            f"  (total {workload.total_video_bytes / (1024**3):.2f} GB, "
            f"largest {workload.largest_video_bytes / (1024**2):.0f} MB)"
            if workload.num_videos
            else ""
        )
    )
    click.echo("Recommendation:")
    click.echo(f"  Encoder:        {'GPU (--gpu)' if rec.use_gpu else 'CPU (--cpu)'}")
    click.echo(f"  Workers:        {rec.workers}")
    for note in rec.notes:
        click.echo(f"  - {note}")
    for warn in rec.warnings:
        click.echo(f"  ! {warn}")
    click.echo("\nSuggested command:")
    click.echo(f"  {cmdline}")
    click.echo("=" * 60 + "\n")


@main.command()
def info() -> None:
    """Display system and configuration information."""
    import torch

    click.echo("\n" + "=" * 60)
    click.echo("AFORGE FOUNDRY (aforge) - SYSTEM INFORMATION")
    click.echo("=" * 60)
    click.echo(f"Version:          {__version__}")
    click.echo(f"PyTorch:          {torch.__version__}")
    click.echo(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA device:      {torch.cuda.get_device_name(0)}")
        click.echo(f"GPU memory:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    click.echo("=" * 60 + "\n")


if __name__ == "__main__":
    main()
