"""Command-line interface for asset optimizer."""

from pathlib import Path

import click

from asset_optimizer import __version__
from asset_optimizer.config import AssetCategory, CompressionLevel, Config, MediaFormat, RembgModel
from asset_optimizer.pipeline import AssetPipeline
from asset_optimizer.utils import configure_logging, get_logger

logger = get_logger("asset-optimizer-cli")


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Asset Optimizer - Professional GPU-accelerated asset optimization tool."""
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
    type=click.Choice(["birefnet-general", "birefnet-portrait", "birefnet-dis", "isnet-general-use", "isnet-anime", "u2net", "u2net_human_seg", "u2net_anime", "u2net_cloth_seg", "silueta", "salient_object_amodal"]),
    default="birefnet-general",
    help="Background removal model",
)
@click.option(
    "--output-format",
    type=click.Choice(["png", "jpeg", "webp", "webm", "mp4", "mkv"]),
    default="webp",
    help="Output format for media optimization",
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
def process(
    input: str,
    output: str,
    background_removal: bool,
    media_optimization: bool,
    category: str,
    model: str,
    output_format: str,
    compression: str,
    gpu: bool,
    workers: int,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Process assets with background removal and media optimization."""
    # Configure logging
    configure_logging(level=log_level)

    logger.info(
        "starting_asset_optimizer",
        input=input,
        output=output,
        version=__version__,
    )

    # Create config
    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=background_removal,
        media_optimization_enabled=media_optimization,
        rembg_category=AssetCategory(category),
        rembg_model=RembgModel(model),
        output_format=MediaFormat(output_format),
        compression_level=CompressionLevel[compression.upper()],
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
    type=click.Choice(["u2net", "u2net_human_seg", "isnet-general-use", "siluette"]),
    default="isnet-general-use",
    help="Background removal model",
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
    "--output-format",
    type=click.Choice(["png", "jpeg", "webp", "webm", "mp4", "mkv"]),
    default="webp",
    help="Output format",
)
@click.option(
    "--compression",
    type=click.Choice(["lossless", "high_quality", "balanced", "aggressive"]),
    default="balanced",
    help="Compression level",
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
def optimize_media(
    input: str,
    output: str,
    output_format: str,
    compression: str,
    workers: int,
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
        format=output_format,
    )

    # Create config
    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=True,
        output_format=MediaFormat(output_format),
        compression_level=CompressionLevel[compression.upper()],
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
def info() -> None:
    """Display system and configuration information."""
    import torch

    click.echo("\n" + "=" * 60)
    click.echo("ASSET OPTIMIZER - SYSTEM INFORMATION")
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
