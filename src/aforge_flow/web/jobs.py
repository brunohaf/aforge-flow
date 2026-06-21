"""Background job manager: runs the pipeline off the request thread.

Each job runs in its own thread and appends JSON-serializable events to an
ordered log guarded by a condition variable. SSE consumers walk the log by
index, so a client connecting (or reconnecting) mid-run replays every event
from the start with no duplicates.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import structlog

from aforge_flow.pipeline import AssetPipeline, result_event
from aforge_flow.web.config_builder import BuiltJob

logger = structlog.get_logger("aforge.web.jobs")


@dataclass
class Job:
    id: str
    status: str = "running"  # running | completed | error
    events: List[dict] = field(default_factory=list)  # ordered event log
    outputs: List[str] = field(default_factory=list)  # successful output paths
    summary: Optional[dict] = None
    error: Optional[str] = None
    finished: bool = False
    _cond: threading.Condition = field(default_factory=threading.Condition)

    def append(self, event: dict) -> None:
        with self._cond:
            self.events.append(event)
            if (
                event.get("type") == "file_done"
                and event.get("success")
                and event.get("output_path")
            ):
                self.outputs.append(event["output_path"])
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    def stream(self) -> Iterator[dict]:
        """Yield every event in order, blocking for new ones until finished."""
        idx = 0
        while True:
            with self._cond:
                while idx >= len(self.events) and not self.finished:
                    self._cond.wait()
                pending = self.events[idx:]
                idx = len(self.events)
                done = self.finished and idx >= len(self.events)
            yield from pending
            if done:
                return


class JobManager:
    """Owns running/finished jobs for the lifetime of the server process."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def start(self, built: BuiltJob) -> str:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, built), daemon=True).start()
        return job.id

    # -- internal ---------------------------------------------------------

    def _run(self, job: Job, built: BuiltJob) -> None:
        try:
            pipeline = AssetPipeline(built.config)
            pipeline.setup()

            if built.files is None:
                pipeline.process_directory(
                    remove_background=built.config.rembg_enabled,
                    optimize_media=built.config.media_optimization_enabled,
                    progress_callback=job.append,
                )
            else:
                self._run_files(job, pipeline, built.files)

            job.summary = pipeline.get_summary()
            job.status = "completed"
            job.append({"type": "done", "summary": job.summary})
        except Exception as exc:  # surfaced to the client
            logger.error("job_failed", job_id=job.id, error=str(exc))
            job.status = "error"
            job.error = str(exc)
            job.append({"type": "error", "error": str(exc)})
        finally:
            job.finish()

    def _run_files(self, job: Job, pipeline: AssetPipeline, files: List[Path]) -> None:
        """Single-file mode: run each selected/uploaded file individually."""
        total = len(files)
        job.append({"type": "start", "total": total})
        for i, path in enumerate(files, start=1):
            result = pipeline.process_asset(
                path,
                remove_background=pipeline.config.rembg_enabled,
                optimize_media=pipeline.config.media_optimization_enabled,
            )
            job.append(result_event(result, i, total))
