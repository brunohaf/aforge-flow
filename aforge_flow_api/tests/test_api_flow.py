"""End-to-end API tests via FastAPI's TestClient (no GPU, no model downloads).

Exercises the options endpoint, the path sandbox, and the full
upload -> job -> SSE -> download flow in media-optimization-only mode (so the
heavy background-removal models are never touched). Everything is written to a
``tmp_path`` sandbox — never the repo's ``assets/`` or ``output/``.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from PIL import Image

from aforge_flow_api.main import create_app


def _make_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (120, 30, 200)).save(path)
    return path


def test_options_and_sandbox(tmp_path):
    client = TestClient(create_app(tmp_path))

    options = client.get("/api/options").json()
    assert {
        "categories",
        "rembg_models",
        "image_formats",
        "video_formats",
        "compression_levels",
        "defaults",
    } <= set(options)
    assert "webp" in options["image_formats"]
    assert "webm" in options["video_formats"]

    # Sandbox: escaping the root is rejected; unknown job is 404.
    assert client.get("/api/browse", params={"path": "/etc"}).status_code == 403
    assert client.get("/api/browse", params={"path": "../../.."}).status_code == 403
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_upload_job_sse_and_download(tmp_path):
    src = _make_png(tmp_path / "sample.png")
    client = TestClient(create_app(tmp_path))

    # 1) Upload
    with src.open("rb") as fh:
        up = client.post("/api/upload", files={"files": ("sample.png", fh, "image/png")})
    assert up.status_code == 200
    targets = up.json()["targets"]
    assert targets and targets[0].startswith(".uploads/")

    # 2) Create job: media optimization only (no model downloads), CPU.
    body = {
        "mode": "files",
        "targets": targets,
        "remove_background": False,
        "optimize_media": True,
        "image_format": "png",
        "video_format": "webm",
        "compression": "balanced",
        "use_gpu": False,
        "workers": 1,
        "skip_existing": False,
        "output_dir": "output",
    }
    job_id = client.post("/api/jobs", json=body).json()["job_id"]

    # 3) Drain the SSE stream until the job finishes.
    seen = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                event = json.loads(line[6:])
                seen.append(event["type"])
                if event["type"] in ("done", "error"):
                    break
    assert "done" in seen and "error" not in seen

    # 4) Final state + outputs + zip + thumbnail.
    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "completed"
    assert state["outputs"]

    zip_resp = client.get(f"/api/jobs/{job_id}/download.zip")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"

    thumb = client.get("/api/file", params={"path": state["outputs"][0], "thumb": "true"})
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/png"
