"""``aforge classify`` — tag images with a creature taxonomy (CLIP/VLM)."""

from pathlib import Path

import click

from aforge_flow.cli._shared import logger
from aforge_flow.core.config import ClipModelChoice, Config
from aforge_flow.core.pipeline import AssetPipeline
from aforge_flow.utils import configure_logging


@click.command()
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
    "--taxonomy",
    default="dnd",
    help="Taxonomy profile: a built-in name (dnd, mundane, call_of_cthulhu) or "
    "a path to a custom profile JSON",
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
@click.option("--gpu/--cpu", default=True, help="Use GPU acceleration")
@click.option("--workers", type=int, default=1, help="Number of parallel workers")
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
    taxonomy: str,
    confidence_threshold: float,
    vlm_fallback: bool,
    vlm_model: str,
    batch_size: int,
    gpu: bool,
    workers: int,
    log_level: str,
) -> None:
    """Classify images into a creature taxonomy from image content alone."""
    configure_logging(level=log_level)

    logger.info(
        "starting_classification",
        input=input,
        output=output,
        clip_model=clip_model,
        taxonomy=taxonomy,
        vlm_fallback=vlm_fallback,
    )

    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=False,
        classification_enabled=True,
        clip_model=ClipModelChoice(clip_model),
        taxonomy_profile=taxonomy,
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
        return
    except Exception as e:
        logger.error("classification_error", error=str(e))
        raise
