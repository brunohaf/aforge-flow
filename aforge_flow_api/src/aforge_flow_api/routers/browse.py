"""GET /api/browse — list folders/files restricted to the configured input_dir."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from aforge_flow.utils import is_supported_format
from aforge_flow_api.core.security import safe_resolve

router = APIRouter()


def _rel(p: Path, root: Path) -> str:
    """Path relative to root as a forward-slash string; '' for root itself."""
    rel = p.relative_to(root)
    s = str(rel)
    return "" if s == "." else s.replace("\\", "/")


@router.get("/api/browse")
def browse(request: Request, path: str = "") -> dict:
    """List immediate sub-folders and supported files of ``path``.

    ``path`` is relative to the server ``--root`` (empty => input_dir).
    Navigation is sandboxed: users cannot browse above ``--input-dir``
    (which defaults to ``--root`` when not set).
    """
    root: Path = request.app.state.root.resolve()
    input_dir: Path = getattr(request.app.state, "input_dir", root).resolve()

    if not path or path in (".", "/"):
        target = input_dir
    else:
        target = safe_resolve(root, path)
        # Extra guard: must stay within input_dir
        if target != input_dir:
            try:
                target.relative_to(input_dir)
            except ValueError:
                raise HTTPException(status_code=403, detail="Path is outside the browseable area")

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
                "path": _rel(child, root),
                "is_dir": is_dir,
                "is_supported": supported,
                "size": size,
            }
        )

    at_input_dir = target == input_dir
    parent = None if at_input_dir else _rel(target.parent, root)

    return {
        "root": str(root),
        "path": _rel(target, root),
        "parent": parent,
        "entries": entries,
    }
