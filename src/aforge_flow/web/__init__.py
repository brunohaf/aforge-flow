"""Local web UI for the aforge-flow pipeline.

A FastAPI backend that reuses :class:`aforge_flow.pipeline.AssetPipeline` to run
background removal and media optimization from a browser, with a server-side
file browser, uploads, live progress (SSE) and result downloads.

Optional install:  ``pip install -e ".[web]"``
"""

from aforge_flow.web.server import create_app

__all__ = ["create_app"]
