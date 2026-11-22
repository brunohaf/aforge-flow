"""``aforge-api`` — launch the local web UI server (uvicorn)."""

from pathlib import Path
from typing import Optional

import click


@click.command()
@click.version_option(package_name="aforge-flow-api", prog_name="aforge-api")
@click.option("--host", default="127.0.0.1", help="Host/interface to bind")
@click.option("--port", type=int, default=8000, help="Port to listen on")
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Directory uploads, downloads and outputs are confined to",
)
@click.option(
    "--input-dir",
    "input_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Sub-folder shown in the file browser (must be inside --root). Defaults to --root.",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    help="Uvicorn log level",
)
def main(host: str, port: int, root: str, input_dir: Optional[str], log_level: str) -> None:
    """Launch the AForge Flow web UI (FastAPI + the built React SPA).

    Open http://HOST:PORT in a browser to browse/upload assets, set options
    visually, run the pipeline, watch live progress and download results.

    Build the frontend first (``cd aforge_flow_ui && npm run build``) to have the
    SPA served at ``/``; otherwise only the JSON API under ``/api`` is exposed.
    """
    import uvicorn

    from aforge_flow_api.main import create_app

    resolved_input_dir = Path(input_dir).resolve() if input_dir else None
    app = create_app(Path(root).resolve(), resolved_input_dir)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
