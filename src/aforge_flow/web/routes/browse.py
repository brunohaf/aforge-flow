"""GET /api/browse — list folders/files under the sandboxed root."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from aforge_flow.utils import is_supported_format
from aforge_flow.web.security import safe_resolve

router = APIRouter()


@router.get("/api/browse")
def browse(request: Request, path: str = "") -> dict:
    """List immediate sub-folders and supported files of ``path``.

    ``path`` is relative to the server ``--root`` (empty => root itself).
    """
    root: Path = request.app.state.root
    target = safe_resolve(root, path)
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    try:
        children = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    for child in children:
        if child.name.startswith("."):
            continue
        is_dir = child.is_dir()
        supported = (not is_dir) and is_supported_format(child)
        if not is_dir and not supported:
            continue
        try:
            size = child.stat().st_size if not is_dir else None
        except OSError:
            size = None
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(root)),
                "is_dir": is_dir,
                "is_supported": supported,
                "size": size,
            }
        )

    rel = "" if target == root.resolve() else str(target.relative_to(root))
    return {
        "root": str(root),
        "path": rel,
        "parent": str(target.parent.relative_to(root)) if target != root.resolve() else None,
        "entries": entries,
    }
