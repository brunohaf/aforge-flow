"""FastAPI application factory for the aforge-flow web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aforge_flow import __version__
from aforge_flow_api.core.settings import default_root, frontend_dist
from aforge_flow_api.routers import (
    browse,
    download,
    files,
    jobs,
    options,
    upload,
)
from aforge_flow_api.services.job_manager import JobManager


def create_app(root: Optional[Path] = None, input_dir: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        root: Directory uploads, downloads and outputs are confined to.
              Defaults to :func:`~aforge_flow_api.core.settings.default_root`.
        input_dir: Sub-directory the file browser is restricted to (must be
                   within ``root``).  Defaults to ``root`` when not set.
    """
    root = (root or default_root()).resolve()
    input_dir = (input_dir or root).resolve()
    if root not in input_dir.parents and input_dir != root:
        raise ValueError(f"input_dir {input_dir} must be inside root {root}")
    input_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="AForge Flow API", version=__version__)
    app.state.root = root
    app.state.input_dir = input_dir
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
    dist = frontend_dist()
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")

    return app


# Module-level app for ``uvicorn aforge_flow_api.main:app`` (root via AFORGE_ROOT/cwd).
app = create_app()
