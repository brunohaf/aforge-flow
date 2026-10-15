"""FastAPI application factory for the aforge-flow web UI."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

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

LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_LOCAL_ORIGIN = re.compile(r"http://(localhost|127\.0\.0\.1|\[::1\]):\d+")

_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(root: Path | None = None, host: str = "127.0.0.1") -> FastAPI:
    """Build the FastAPI app.

    Args:
        root: Directory the file browser / uploads / downloads are confined to.
              Defaults to the current working directory.
        host: Interface the server binds to; accepted as a Host header in
              addition to the loopback names.
    """
    root = (root or Path.cwd()).resolve()

    app = FastAPI(title="aforge-flow", version=__version__)
    app.state.root = root
    app.state.jobs = JobManager()

    # The API has no auth, so a Host allowlist blocks DNS-rebinding pages and an
    # Origin check blocks cross-site requests (multipart POSTs skip CORS preflight).
    @app.middleware("http")
    async def reject_foreign_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and not _LOCAL_ORIGIN.fullmatch(origin):
            return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        return await call_next(request)

    # Dev: Vite dev server (other origin) talking to this backend directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCAL_ORIGIN.pattern,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*LOCAL_HOSTS, host])

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
