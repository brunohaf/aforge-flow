"""``aforge optimize`` — transcode images/video to optimized formats."""

from pathlib import Path

import click

from aforge_flow.cli._shared import _apply_auto_tune, logger
from aforge_flow.core.config import CompressionLevel, Config, MediaFormat, WebmCodec
from aforge_flow.core.pipeline import AssetPipeline
from aforge_flow.utils import configure_logging


@click.command("optimize")
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
    help="Compression level. 'lossless' is truly lossless for images and visually "
    "lossless (very low CRF) for video — bit-exact video from an already-lossy "
    "source is 5-20x larger for no visible gain",
)
@click.option(
    "--webm-codec",
    type=click.Choice(["auto", "vp9", "av1"]),
    default="auto",
    help="Codec inside WebM. 'auto' uses AV1 (~30%% smaller), falling back to VP9 "
    "for sources with alpha — AV1 cannot encode transparency. Force 'vp9' for "
    "players that predate AV1",
)
@click.option(
    "--passthrough/--no-passthrough",
    default=True,
    help="Copy files already in the target format (.webp/.webm) instead of "
    "re-encoding them, which would only cost a generation of quality",
)
@click.option(
    "--keep-smallest/--force-transcode",
    default=True,
    help="Discard a transcode that comes out bigger than its source and keep the "
    "original instead, so a run can never inflate the library. Fires on sources "
    "already efficiently compressed (e.g. well-encoded h264)",
)
@click.option(
    "--gpu/--cpu",
    default=True,
    help="Allow NVENC hardware encoding. It is a speed option, not a quality one: "
    "CPU encoders compress better per bit, and are used anyway for alpha sources "
    "or when NVENC fails its trial encode",
)
@click.option(
    "--ignore-ext",
    multiple=True,
    help="Extension to exclude from discovery, repeatable (e.g. --ignore-ext mp4). "
    "Worth using on sources a transcode cannot improve: a well-encoded h264 mp4 is "
    "already smaller than any quality-preserving AV1 re-encode of it",
)
@click.option("--workers", type=int, default=1, help="Number of parallel workers")
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
    webm_codec: str,
    passthrough: bool,
    keep_smallest: bool,
    gpu: bool,
    ignore_ext: tuple,
    workers: int,
    auto_tune: bool,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Optimize and transcode media files to WebM/WebP."""
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

    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=True,
        image_format=MediaFormat(image_format),
        video_format=MediaFormat(video_format),
        compression_level=CompressionLevel[compression.upper()],
        webm_codec=WebmCodec(webm_codec),
        passthrough_matching_format=passthrough,
        keep_smallest_output=keep_smallest,
        ignore_extensions=list(ignore_ext),
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
    )

    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=False, optimize_media=True)
        pipeline.print_summary()
        logger.info("media_optimization_complete")
        return
    except Exception as e:
        logger.error("media_optimization_error", error=str(e))
        raise
