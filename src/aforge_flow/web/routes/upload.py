"""POST /api/upload — accept user files into a per-session temp dir under root."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from aforge_flow.utils import is_supported_format
from aforge_flow.web.schemas import UploadResponse

router = APIRouter()

UPLOAD_DIRNAME = ".uploads"
MAX_FILES = 500
MAX_FILE_BYTES = 2 * 1024**3
_CHUNK = 1024 * 1024


@router.post("/api/upload", response_model=UploadResponse)
async def upload(request: Request, files: list[UploadFile]) -> UploadResponse:
    """Save uploaded files under ``<root>/.uploads/<uuid>/`` and return their paths.

    The returned paths (relative to root) are meant to be passed straight back
    as a ``files`` job's ``targets``.
    """
    root: Path = request.app.state.root
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"At most {MAX_FILES} files per upload")

    names = [Path(f.filename or "").name for f in files]  # strip any path components
    for name in names:
        if name and not is_supported_format(Path(name)):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {name}")

    dest_dir = root / UPLOAD_DIRNAME / uuid.uuid4().hex
    saved: list[str] = []
    try:
        for upload_file, name in zip(files, names, strict=True):
            if not name:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / name
            _copy_capped(upload_file, target)
            saved.append(str(target.relative_to(root)))
    except BaseException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise

    if not saved:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    return UploadResponse(targets=saved)


def _copy_capped(upload_file: UploadFile, target: Path) -> None:
    """Stream to disk, deleting the partial file if it exceeds ``MAX_FILE_BYTES``."""
    written = 0
    with target.open("wb") as out:
        while chunk := upload_file.file.read(_CHUNK):
            written += len(chunk)
            if written > MAX_FILE_BYTES:
                break
            out.write(chunk)
    if written > MAX_FILE_BYTES:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"{target.name} exceeds the upload size limit")
