"""POST /api/upload — accept user files into a per-session temp dir under root."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile

from aforge_flow.utils import is_supported_format
from aforge_flow.web.schemas import UploadResponse

router = APIRouter()

UPLOAD_DIRNAME = ".uploads"


@router.post("/api/upload", response_model=UploadResponse)
async def upload(request: Request, files: List[UploadFile]) -> UploadResponse:
    """Save uploaded files under ``<root>/.uploads/<uuid>/`` and return their paths.

    The returned paths (relative to root) are meant to be passed straight back
    as a ``files`` job's ``targets``.
    """
    root: Path = request.app.state.root
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    dest_dir = root / UPLOAD_DIRNAME / uuid.uuid4().hex
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for upload_file in files:
        name = Path(upload_file.filename or "").name  # strip any path components
        if not name:
            continue
        target = dest_dir / name
        if not is_supported_format(target):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {name}")
        with target.open("wb") as out:
            shutil.copyfileobj(upload_file.file, out)
        saved.append(str(target.relative_to(root)))

    if not saved:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    return UploadResponse(targets=saved)
