"""Download a single result file or a zip of a whole job's outputs."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from aforge_flow.web.security import safe_resolve

router = APIRouter()


@router.get("/api/download")
def download_file(request: Request, path: str) -> FileResponse:
    """Download a single (sandboxed) file as an attachment."""
    root: Path = request.app.state.root
    target = safe_resolve(root, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=target.name)


@router.get("/api/jobs/{job_id}/download.zip")
def download_job_zip(job_id: str, request: Request) -> StreamingResponse:
    """Bundle every successful output of a job into a single zip."""
    root: Path = request.app.state.root
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.outputs:
        raise HTTPException(status_code=404, detail="No outputs to download yet")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for out in job.outputs:
            # Re-validate each output path through the sandbox before reading.
            p = safe_resolve(root, out)
            if not p.is_file():
                continue
            arcname = p.name
            n = 1
            while arcname in seen:  # avoid clobbering same-named outputs
                arcname = f"{p.stem}_{n}{p.suffix}"
                n += 1
            seen.add(arcname)
            zf.write(p, arcname=arcname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="aforge_{job_id[:8]}.zip"'},
    )
