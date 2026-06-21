"""FastAPI application factory for the aforge-flow web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aforge_flow import __version__
from aforge_flow.web.jobs import JobManager
from aforge_flow.web.routes import (
    browse,
    download,
    files,
    jobs,
    options,
    upload,
)

# Built SPA lives here in production (``npm run build`` output).
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(root: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        root: Directory the file browser / uploads / downloads are confined to.
              Defaults to the current working directory.
    """
    root = (root or Path.cwd()).resolve()

    app = FastAPI(title="AForge Flow", version=__version__)
    app.state.root = root
    app.state.jobs = JobManager()

    # Dev: Vite dev server (other origin) talking to this backend directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (options, browse, jobs, files, upload, download):
        app.include_router(module.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "root": str(root)}

    # Serve the built SPA last so it doesn't shadow the API routes above.
    if _FRONTEND_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="spa")

    return app
