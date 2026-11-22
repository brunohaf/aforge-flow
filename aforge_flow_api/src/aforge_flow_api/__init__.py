"""aforge-flow-api — FastAPI service exposing the pipeline as a local web UI.

Browse/upload assets, set options visually, run background removal + media
optimization, watch live progress (SSE) and download results. Reuses
:class:`aforge_flow.core.pipeline.AssetPipeline`.

Launch with the ``aforge-api`` console script, or
``uvicorn aforge_flow_api.main:app``.
"""

__version__ = "0.1.0"

from aforge_flow_api.main import create_app

__all__ = ["create_app", "__version__"]
