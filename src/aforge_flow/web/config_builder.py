"""Translate a :class:`JobRequest` into a pipeline :class:`Config`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException

from aforge_flow.config import (
    MODEL_BY_CATEGORY,
    AssetCategory,
    CompressionLevel,
    Config,
    MediaFormat,
    RembgModel,
)
from aforge_flow.web.schemas import JobRequest
from aforge_flow.web.security import safe_resolve


@dataclass
class BuiltJob:
    """Resolved, ready-to-run inputs for a job."""

    config: Config
    # ``None`` => run ``process_directory`` over ``config.input_dir``.
    # Otherwise => the explicit list of files to run through ``process_asset``.
    files: Optional[List[Path]]


def build_job(req: JobRequest, root: Path) -> BuiltJob:
    """Validate request paths and assemble a pipeline ``Config``.

    Enforces the ``input_dir`` constraint from ``BaseProcessor.get_output_path``
    (every processed file must live under ``config.input_dir``).
    """
    if not req.targets:
        raise HTTPException(status_code=400, detail="No targets selected")

    resolved = [safe_resolve(root, t) for t in req.targets]
    output_dir = safe_resolve(root, req.output_dir)

    files: Optional[List[Path]]
    if req.mode == "directory":
        if len(resolved) != 1 or not resolved[0].is_dir():
            raise HTTPException(status_code=400, detail="Directory mode needs exactly one folder")
        input_dir = resolved[0]
        files = None
    else:
        for p in resolved:
            if not p.is_file():
                raise HTTPException(status_code=400, detail=f"Not a file: {p}")
        # input_dir must contain every target so relative_to() works downstream.
        input_dir = Path(os.path.commonpath([str(p.parent) for p in resolved]))
        files = resolved

    category = _enum(AssetCategory, req.category, "category")
    rembg_model = (
        _enum(RembgModel, req.rembg_model, "rembg_model")
        if req.rembg_model
        else MODEL_BY_CATEGORY[category]
    )
    try:
        compression = CompressionLevel[req.compression.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid compression: {req.compression}")

    config = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        rembg_enabled=req.remove_background,
        media_optimization_enabled=req.optimize_media,
        rembg_category=category,
        rembg_model=rembg_model,
        image_format=_enum(MediaFormat, req.image_format, "image_format"),
        video_format=_enum(MediaFormat, req.video_format, "video_format"),
        compression_level=compression,
        use_gpu=req.use_gpu,
        max_workers=req.workers,
        skip_existing=req.skip_existing,
        # Out of scope for this phase.
        classification_enabled=False,
        caption_rename_enabled=False,
    )
    return BuiltJob(config=config, files=files)


def _enum(enum_cls, value, field):
    try:
        return enum_cls(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value}")
