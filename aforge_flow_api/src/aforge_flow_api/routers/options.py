"""GET /api/options — enum choices + defaults so the UI never desyncs."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from aforge_flow.core.config import (
    AssetCategory,
    CompressionLevel,
    MediaFormat,
    RembgModel,
)

router = APIRouter()


@router.get("/api/options")
def get_options(request: Request) -> dict:
    """Expose the choices the form needs, derived from the config enums."""
    root: Path = request.app.state.root.resolve()
    input_dir: Path = getattr(request.app.state, "input_dir", root).resolve()

    input_dir_rel = ""
    if input_dir != root:
        try:
            rel = input_dir.relative_to(root)
            input_dir_rel = str(rel).replace("\\", "/")
        except ValueError:
            input_dir_rel = ""

    image_formats = [f.value for f in MediaFormat if f.is_image]
    video_formats = [f.value for f in MediaFormat if f.is_video]
    return {
        "categories": [c.value for c in AssetCategory],
        "rembg_models": [m.value for m in RembgModel],
        "image_formats": image_formats,
        "video_formats": video_formats,
        "compression_levels": [c.name.lower() for c in CompressionLevel],
        "input_dir": input_dir_rel,
        "defaults": {
            "mode": "directory",
            "remove_background": True,
            "optimize_media": True,
            "category": AssetCategory.GENERAL.value,
            "rembg_model": None,
            "image_format": MediaFormat.WEBP.value,
            "video_format": MediaFormat.WEBM.value,
            "compression": CompressionLevel.BALANCED.name.lower(),
            "use_gpu": True,
            "workers": 1,
            "skip_existing": True,
            "output_dir": "output",
        },
    }
