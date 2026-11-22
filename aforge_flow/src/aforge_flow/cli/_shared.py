"""Shared helpers and constants for the ``aforge`` CLI commands."""

from pathlib import Path

from aforge_flow.utils import (
    detect_hardware,
    get_logger,
    recommend_settings,
    scan_workload,
)

logger = get_logger("aforge-cli")

# Make ``-h`` an alias for ``--help`` everywhere. Click propagates a parent
# context's ``help_option_names`` to every subcommand, so setting it once on
# the group covers ``aforge -h`` and ``aforge <command> -h`` alike.
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


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
