"""GET /api/file — serve an image/video (optionally as a thumbnail) for preview."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from aforge_flow.utils import is_image
from aforge_flow.web.security import safe_resolve

router = APIRouter()

_THUMB_MAX = 384  # px, longest edge


@router.get("/api/file")
def get_file(request: Request, path: str, thumb: bool = False) -> Response:
    """Serve the file at ``path`` (sandboxed). ``thumb=true`` downsizes images."""
    root: Path = request.app.state.root
    target = safe_resolve(root, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if thumb and is_image(target):
        return _thumbnail(target)
    return FileResponse(target)


def _thumbnail(target: Path) -> Response:
    from PIL import Image

    try:
        with Image.open(target) as im:
            im = im.convert("RGBA")
            im.thumbnail((_THUMB_MAX, _THUMB_MAX))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
    except Exception as exc:  # corrupt/unsupported image
        raise HTTPException(status_code=415, detail=f"Cannot render image: {exc}")
    return Response(content=buf.getvalue(), media_type="image/png")
