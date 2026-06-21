"""Job creation, status, and live SSE progress."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from aforge_flow.web.config_builder import build_job
from aforge_flow.web.schemas import JobCreated, JobRequest

router = APIRouter()


@router.post("/api/jobs", response_model=JobCreated)
def create_job(req: JobRequest, request: Request) -> JobCreated:
    built = build_job(req, request.app.state.root)
    job_id = request.app.state.jobs.start(built)
    return JobCreated(job_id=job_id)


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "status": job.status,
        "events": job.events,
        "outputs": job.outputs,
        "summary": job.summary,
        "error": job.error,
    }


@router.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        for event in job.stream():
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
