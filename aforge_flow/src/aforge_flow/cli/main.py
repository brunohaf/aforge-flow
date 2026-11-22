"""``aforge`` CLI entrypoint — assembles the command group and aliases."""

import click

from aforge_flow import __version__
from aforge_flow.cli._shared import CONTEXT_SETTINGS
from aforge_flow.cli.commands.caption_rename import caption_rename
from aforge_flow.cli.commands.classify import classify
from aforge_flow.cli.commands.info import info
from aforge_flow.cli.commands.optimize import optimize_media
from aforge_flow.cli.commands.process import process
from aforge_flow.cli.commands.recommend import recommend
from aforge_flow.cli.commands.remove_background import remove_background


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="aforge")
def main() -> None:
    """aforge — AForge Flow: GPU-accelerated tabletop asset pipeline.

    Batch-prepare RPG/VTT artwork: strip backgrounds, transcode images and
    video to web-friendly formats, and classify creatures into a pluggable
    taxonomy (D&D by default; swap with --taxonomy).

    \b
    Commands:
      process              Run the full pipeline (background removal + media
                           optimization; --caption-rename and --classify opt-in).
                           Aliases: go, pipe
      remove-background    Strip image backgrounds with rembg
      optimize             Transcode images/video to optimized formats
                           (alias: optimize-media)
      caption-rename       Rename images, keeping meaningful source names and
                           captioning only junk-named files (alias: rename)
      classify             Tag images with a creature taxonomy (CLIP/VLM)
      recommend            Suggest --workers/--gpu for this machine + workload
      info                 Show version, PyTorch, and CUDA/GPU details

    \b
    Examples:
      aforge recommend -i assets --video-format webm
      aforge go -i assets -o output --auto-tune --caption-rename --classify
      aforge optimize -i assets -o output --image-format webp --video-format webm
      aforge rename -i assets -o output
      aforge classify -i assets --clip-model siglip2 --taxonomy dnd

    Run ``aforge <command> -h`` for per-command options.
    """
    pass


# Register the subcommands (each defines its own primary name).
for _command in (
    process,
    remove_background,
    optimize_media,
    classify,
    caption_rename,
    recommend,
    info,
):
    main.add_command(_command)

# Command aliases — the common verbs all work: ``go``/``pipe`` run the full
# ``process`` pipeline, ``optimize-media`` stays valid, ``rename`` maps to
# ``caption-rename``.
main.add_command(process, name="go")
main.add_command(process, name="pipe")
main.add_command(optimize_media, name="optimize-media")
main.add_command(caption_rename, name="rename")


if __name__ == "__main__":
    main()
