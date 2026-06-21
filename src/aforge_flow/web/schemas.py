"""Request/response models for the web API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    """A processing request coming from the UI.

    Fields map directly onto :class:`aforge_flow.config.Config`; the conversion
    lives in :mod:`aforge_flow.web.config_builder`.
    """

    # What to process. ``directory`` => ``targets`` holds a single folder;
    # ``files`` => ``targets`` holds one or more files (browsed or uploaded).
    mode: Literal["directory", "files"] = "directory"
    targets: List[str] = Field(default_factory=list, description="Paths relative to root (or absolute under root)")
    output_dir: str = Field(default="output", description="Output directory (relative to root or absolute under root)")

    # Operations
    remove_background: bool = True
    optimize_media: bool = True

    # Background removal
    category: str = "general"
    rembg_model: Optional[str] = None  # None => auto-select per category

    # Media optimization
    image_format: str = "webp"
    video_format: str = "webm"
    compression: str = "balanced"

    # Runtime
    use_gpu: bool = True
    workers: int = Field(default=1, ge=1)
    skip_existing: bool = True


class JobCreated(BaseModel):
    job_id: str


class UploadResponse(BaseModel):
    """Paths (relative to root) of the freshly uploaded files."""

    targets: List[str]
