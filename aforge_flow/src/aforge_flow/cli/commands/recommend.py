"""``aforge recommend`` — suggest --workers/--gpu for this machine + workload."""

from pathlib import Path

import click

from aforge_flow.utils import (
    as_dict,
    build_command_line,
    detect_hardware,
    recommend_settings,
    scan_workload,
)


@click.command()
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
    command = "process" if full_pipeline else "optimize"
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
