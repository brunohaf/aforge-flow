"""Path sandboxing for the web file browser / file-serving endpoints.

Every user-supplied path is resolved and checked to live under the configured
``--root``. This is the single guard that keeps the browser, file-preview,
upload and download endpoints from touching arbitrary parts of the disk.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from aforge_flow.utils import is_supported_format


def require_media(path: Path) -> Path:
    """Refuse anything that is not a supported image/video (e.g. ``.env``, keys, ``.git``)."""
    if not is_supported_format(path):
        raise HTTPException(status_code=403, detail="Only supported media files can be served")
    return path


def safe_resolve(root: Path, user_path: str | None) -> Path:
    """Resolve ``user_path`` against ``root`` and ensure it stays inside it.

    ``user_path`` may be empty (means ``root`` itself), relative to ``root``, or
    absolute — but the resolved result must be within ``root`` or a 403 is
    raised. Symlinks are resolved before the check so they cannot escape.
    """
    root = root.resolve()
    if not user_path or user_path in (".", "/"):
        return root

    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=403, detail="Path is outside the allowed root")
    return resolved
