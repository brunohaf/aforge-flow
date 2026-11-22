"""Runtime settings for the API: sandbox root and built-SPA location."""

from __future__ import annotations

import os
from pathlib import Path

# .../aforge_flow_api/src/aforge_flow_api/core/settings.py -> parents[4] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]


def default_root() -> Path:
    """Default sandbox root: ``AFORGE_ROOT`` env var, else the current directory."""
    env = os.environ.get("AFORGE_ROOT")
    return Path(env).resolve() if env else Path.cwd().resolve()


def frontend_dist() -> Path:
    """Directory of the built SPA (``aforge_flow_ui/dist``).

    Override with the ``AFORGE_FRONTEND_DIST`` env var (useful when the API is
    installed somewhere other than the monorepo checkout). Defaults to the
    sibling ``aforge_flow_ui/dist`` next to the two Python projects.
    """
    env = os.environ.get("AFORGE_FRONTEND_DIST")
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT / "aforge_flow_ui" / "dist"
