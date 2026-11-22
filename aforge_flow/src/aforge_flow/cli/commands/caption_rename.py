"""``aforge caption-rename`` — rename images (and pass videos through) by caption."""

from pathlib import Path

import click

from aforge_flow.cli._shared import logger
from aforge_flow.core.config import CaptionBackend, Config
from aforge_flow.core.pipeline import AssetPipeline
from aforge_flow.utils import configure_logging


@click.command("caption-rename")
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
    help="Output directory for renamed images (videos are copied through as-is)",
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
    "--taxonomy",
    default="dnd",
    help="Taxonomy profile that supplies the captioning instruction/name-bank: "
    "a built-in name (dnd, mundane, call_of_cthulhu) or a path to a profile JSON",
)
@click.option(
    "--caption-prompt",
    type=str,
    default="",
    help="Override the conditional BLIP prefix (blip backend only). Empty uses "
    "the active taxonomy profile's default.",
)
@click.option(
    "--caption-vlm-model",
    type=str,
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    help="HuggingFace id of the Qwen2.5-VL model (qwen backend only)",
)
@click.option(
    "--hash-threshold",
    type=click.IntRange(0, 64),
    default=5,
    help="Max perceptual-hash Hamming distance for two images to count as replicas",
)
@click.option(
    "--slug-max-words",
    type=click.IntRange(1, None),
    default=5,
    help="Maximum number of words kept in a generated slug",
)
@click.option("--gpu/--cpu", default=True, help="Use GPU acceleration")
@click.option("--workers", type=int, default=1, help="Number of parallel workers")
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
def caption_rename(
    input: str,
    output: str,
    caption_backend: str,
    preserve_source_names: bool,
    taxonomy: str,
    caption_prompt: str,
    caption_vlm_model: str,
    hash_threshold: int,
    slug_max_words: int,
    gpu: bool,
    workers: int,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Rename images from AI-generated captions (replicas share a slug).

    Videos (webm, mp4, …) are copied to the output unchanged rather than skipped.
    """
    configure_logging(level=log_level)

    logger.info(
        "starting_caption_rename",
        input=input,
        output=output,
        caption_backend=caption_backend,
        taxonomy=taxonomy,
    )

    config = Config(
        input_dir=Path(input),
        output_dir=Path(output),
        rembg_enabled=False,
        media_optimization_enabled=False,
        caption_rename_enabled=True,
        caption_backend=CaptionBackend(caption_backend),
        caption_preserve_source_names=preserve_source_names,
        taxonomy_profile=taxonomy,
        caption_prompt=caption_prompt,
        caption_vlm_model=caption_vlm_model,
        replica_hash_threshold=hash_threshold,
        slug_max_words=slug_max_words,
        use_gpu=gpu,
        max_workers=workers,
        skip_existing=skip_existing,
    )

    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=False, optimize_media=False)
        pipeline.print_summary()
        logger.info("caption_rename_complete")
        return
    except Exception as e:
        logger.error("caption_rename_error", error=str(e))
        raise
