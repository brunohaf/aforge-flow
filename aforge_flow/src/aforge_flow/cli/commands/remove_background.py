"""``aforge remove-background`` — strip image backgrounds with rembg."""

from pathlib import Path

import click

from aforge_flow.cli._shared import logger
from aforge_flow.core.config import AssetCategory, Config, RembgModel
from aforge_flow.core.pipeline import AssetPipeline
from aforge_flow.utils import configure_logging, get_files_recursively, is_image
from aforge_flow.utils.validators import SUPPORTED_VIDEO_FORMATS


def _run_benchmark(config: Config, n: int) -> None:
    """Time *n* sample images and report active providers + projected 20k runtime.

    Builds the pipeline (which builds the session and runs warmup), processes the
    first *n* images into a throwaway temp dir with ``skip_existing`` off so each
    file actually runs inference, then prints the active ONNXRuntime providers and
    timing. Does not touch the real output dir or run the full batch.
    """
    import tempfile
    import time as _time

    images = [
        f
        for f in get_files_recursively(
            config.input_dir,
            supported_only=True,
            ignore_dirs=config.ignore_dirs,
            ignore_extensions=config.ignore_extensions,
        )
        if is_image(f)
    ][:n]

    if not images:
        logger.warning("benchmark_no_images", directory=str(config.input_dir))
        return

    with tempfile.TemporaryDirectory(prefix="aforge_bench_") as tmp:
        config.output_dir = Path(tmp)
        config.skip_existing = False

        pipeline = AssetPipeline(config)
        pipeline.setup()
        proc = pipeline.background_removal_processor
        active = (
            proc.session.inner_session.get_providers()
            if proc is not None and proc.session is not None
            else []
        )

        durations = []
        for image_path in images:
            start = _time.perf_counter()
            result = proc.process(image_path)
            if not result.success:
                logger.warning("benchmark_file_failed", input=str(image_path), error=result.error)
                continue
            durations.append(_time.perf_counter() - start)

    if not durations:
        logger.error("benchmark_no_successful_files")
        return

    mean_ms = 1000 * sum(durations) / len(durations)
    projected_hours = (mean_ms / 1000) * 20_000 / 3600

    print("\n" + "=" * 60)
    print("REMBG BENCHMARK")
    print("=" * 60)
    print(f"Active providers:   {active}")
    print(f"Images timed:       {len(durations)}")
    print(f"Mean per image:     {mean_ms:.1f} ms")
    print(f"Throughput:         {1000 / mean_ms:.2f} img/s")
    print(f"Projected 20k run:  {projected_hours:.2f} h")
    print("=" * 60 + "\n")


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
@click.option("--gpu/--cpu", default=True, help="Use GPU acceleration")
@click.option("--workers", type=int, default=1, help="Number of parallel workers")
@click.option(
    "--provider",
    type=click.Choice(["auto", "tensorrt", "cuda", "cpu"]),
    default="cuda",
    help="ONNXRuntime execution provider (default 'cuda'). 'auto' tries "
    "TensorRT→CUDA→CPU, but TensorRT FP16 mis-segments BRIA — see --fp16.",
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
    default=3,
    show_default=True,
    help="PNG zlib level for output (lower = faster + larger; always lossless). "
    "3 keeps files ~as small as 6 for a fraction of the encode time.",
)
@click.option(
    "--gpu-mem-limit",
    type=float,
    default=0.0,
    show_default=True,
    help="Hard cap on the ONNXRuntime CUDA arena, in GiB. 0 auto-caps to ~90% of "
    "free VRAM at session build; set e.g. 6 to pin it manually.",
)
@click.option(
    "--cudnn-exhaustive/--no-cudnn-exhaustive",
    default=False,
    help="Use EXHAUSTIVE cuDNN conv-algorithm search: a one-time autotune for "
    "faster steady-state inference. Worth enabling for big homogeneous batches.",
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
    default=tuple(sorted(SUPPORTED_VIDEO_FORMATS)),
    show_default=True,
    help="File extension to skip (repeatable). Defaults to video extensions, "
    "which background removal cannot process.",
)
@click.option(
    "--benchmark",
    type=int,
    default=0,
    help="Diagnostic: time N sample images, print active providers + ms/image + "
    "projected 20k runtime, then exit (does not run the full batch)",
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
    provider: str,
    fp16: bool,
    png_compress_level: int,
    gpu_mem_limit: float,
    cudnn_exhaustive: bool,
    matte: bool,
    dedup: bool,
    ignore_dirs: tuple,
    ignore_extensions: tuple,
    benchmark: int,
    skip_existing: bool,
    log_level: str,
) -> None:
    """Remove backgrounds from images using rembg."""
    configure_logging(level=log_level)

    logger.info(
        "starting_background_removal",
        input=input,
        output=output,
        category=category,
        model=model,
        provider=provider,
    )

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
        rembg_provider=provider,
        trt_fp16=fp16,
        matte_background=matte,
        png_compress_level=png_compress_level,
        gpu_mem_limit_gb=gpu_mem_limit,
        cudnn_algo_exhaustive=cudnn_exhaustive,
        dedup_enabled=dedup,
        ignore_dirs=list(ignore_dirs),
        ignore_extensions=list(ignore_extensions),
    )

    if benchmark > 0:
        _run_benchmark(config, benchmark)
        return

    pipeline = AssetPipeline(config)
    pipeline.setup()

    try:
        pipeline.process_directory(remove_background=True, optimize_media=False)
        pipeline.print_summary()
        logger.info("background_removal_complete")
        return
    except Exception as e:
        logger.error("background_removal_error", error=str(e))
        raise
