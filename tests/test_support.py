"""Tests for recommendation heuristics, caches, the web API surface and CLI text output."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from conftest import make_image

from aforge_flow import cli
from aforge_flow.utils import recommend as rec
from aforge_flow.utils.deduplication import DuplicateDetector
from aforge_flow.utils.embedding_cache import EmbeddingCache


def hw(**overrides) -> rec.HardwareInfo:
    base = dict(
        physical_cores=8,
        logical_cores=16,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        has_cuda=True,
        gpu_name="RTX",
        total_vram_gb=12.0,
        free_vram_gb=10.0,
        has_av1_nvenc=True,
    )
    return rec.HardwareInfo(**{**base, **overrides})


def work(images=0, videos=0, largest=0) -> rec.WorkloadInfo:
    return rec.WorkloadInfo(images, videos, largest * videos, largest)


# recommendation heuristics


@pytest.mark.parametrize(
    "hardware, fmt, use_gpu, note",
    [
        (hw(has_cuda=False), "webm", False, "No CUDA GPU"),
        (hw(), "mp4", False, "encodes on CPU"),
        (hw(has_av1_nvenc=False), "webm", False, "av1_nvenc is unavailable"),
        (hw(), "webm", True, "GPU AV1 NVENC"),
    ],
)
def test_gpu_decision(hardware, fmt, use_gpu, note) -> None:
    r = rec.recommend_settings(hardware, work(images=1, videos=1), fmt)
    assert r.use_gpu is use_gpu
    assert any(note in n for n in r.notes)


def test_gpu_video_workers_are_small() -> None:
    assert rec.recommend_settings(hw(), work(videos=2), "webm").workers == rec.GPU_VIDEO_WORKERS
    mixed = rec.recommend_settings(hw(), work(images=2, videos=2), "webm")
    assert mixed.workers == rec.GPU_VIDEO_WORKERS_WITH_IMAGES


def test_cpu_video_workers_are_ram_capped() -> None:
    r = rec.recommend_settings(
        hw(has_cuda=False, physical_cores=32, available_ram_gb=1.0), work(videos=3), "webm"
    )
    assert r.workers == 1
    assert any("Reduced workers" in w for w in r.warnings)


def test_image_only_and_large_video_and_low_vram_warnings() -> None:
    assert rec.recommend_settings(hw(), work(images=10), "webm").workers >= 1

    big = rec.recommend_settings(
        hw(has_cuda=False, physical_cores=32, available_ram_gb=64),
        work(videos=4, largest=rec.LARGE_VIDEO_WARN_BYTES),
        "webm",
    )
    assert any("Largest video" in w for w in big.warnings)

    low = rec.recommend_settings(hw(free_vram_gb=0.5), work(images=1), "webm", full_pipeline=True)
    assert any("VRAM free" in w for w in low.warnings)


def test_scan_workload_counts_media(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png")
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 100)
    w = rec.scan_workload(tmp_path)
    assert (w.num_images, w.num_videos, w.largest_video_bytes) == (1, 1, 100)
    assert rec.scan_workload(tmp_path / "missing").num_images == 0


def test_detect_hardware_degrades_without_cuda(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = rec.detect_hardware()
    assert info.physical_cores >= 1 and info.has_cuda is False


def test_build_command_line_and_as_dict() -> None:
    r = rec.Recommendation(use_gpu=False, workers=3)
    line = rec.build_command_line("process", "in", "out", "webm", r)
    assert line.startswith("aforge process") and "--cpu" in line and "--workers 3" in line
    payload = rec.as_dict(hw(), work(images=1), r)
    assert payload["recommendation"]["workers"] == 3


# caches


def test_embedding_cache_batch_api_and_dim_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "e.sqlite"
    with EmbeddingCache(db, "m", dim=4) as cache:
        cache.put_many({"a": np.ones(4, np.float32), "b": np.zeros(4, np.float32)})
        assert set(cache.get_many(["a", "b", "zz"])) == {"a", "b"}
        assert cache.get_many([]) == {}
        assert cache.count() == 2
    with EmbeddingCache(db, "m", dim=8) as wrong_dim:
        assert wrong_dim.get("a") is None
        assert wrong_dim.get_many(["a"]) == {}


def test_duplicate_detector_flags_identical_content(tmp_path: Path) -> None:
    a = make_image(tmp_path / "a.png", seed=1)
    b = tmp_path / "b.png"
    b.write_bytes(a.read_bytes())
    detector = DuplicateDetector(tmp_path / "hashes")
    assert not detector.is_duplicate(a)
    assert detector.is_duplicate(b)
    assert detector.register_file(a) == detector.get_file_hash(b)


# CLI text output and error paths


def invoke(*args: str):
    return CliRunner().invoke(cli.main, list(args))


def test_recommend_text_output(workdir: Path) -> None:
    make_image(workdir / "assets" / "a.png")
    result = invoke("recommend", "-i", "assets")
    assert result.exit_code == 0, result.output
    assert "PERFORMANCE RECOMMENDATION" in result.output
    assert "Suggested command:" in result.output


def test_auto_tune_overrides_workers(workdir: Path, monkeypatch) -> None:
    make_image(workdir / "assets" / "a.png")
    configs: list = []

    class Recording:
        def __init__(self, config) -> None:
            configs.append(config)

        def setup(self) -> None: ...
        def process_directory(self, **_kwargs) -> None: ...
        def print_summary(self) -> None: ...

    monkeypatch.setattr(cli, "_apply_auto_tune", lambda *a, **k: (False, 2))
    monkeypatch.setattr(cli, "AssetPipeline", Recording)
    result = invoke("optimize", "-i", "assets", "-o", "output", "--gpu", "--auto-tune")
    assert result.exit_code == 0, result.output
    assert configs[0].max_workers == 2
    assert configs[0].use_gpu is False


def test_every_bundled_model_id_is_pinned() -> None:
    from aforge_flow.config import HF_REVISIONS, Config, hf_revision
    from aforge_flow.processors.caption_naming import MODEL_ID
    from aforge_flow.processors.creature_classification import _ENCODER_REGISTRY

    bundled = {str(spec["hf_id"]) for spec in _ENCODER_REGISTRY.values()}
    bundled |= {
        MODEL_ID,
        Config.model_fields["caption_vlm_model"].default,
        Config.model_fields["vlm_model"].default,
    }
    assert bundled <= HF_REVISIONS.keys()
    assert all(re.fullmatch(r"[0-9a-f]{40}", hf_revision(hf_id)) for hf_id in bundled)
    assert hf_revision("someone/custom-model") == "main"


@pytest.mark.parametrize(
    "command",
    [
        ["process", "--no-background-removal"],
        ["remove-background"],
        ["optimize"],
        ["classify"],
        ["caption-rename"],
    ],
)
def test_pipeline_errors_propagate(workdir: Path, monkeypatch, command) -> None:
    make_image(workdir / "assets" / "a.png")

    class Exploding:
        def __init__(self, config) -> None: ...
        def setup(self) -> None: ...
        def process_directory(self, **_k) -> None:
            raise RuntimeError("disk full")

    monkeypatch.setattr(cli, "AssetPipeline", Exploding)
    result = invoke(*command, "-i", "assets", "-o", "output", "--cpu")
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def test_web_command_starts_uvicorn(workdir: Path, monkeypatch) -> None:
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    result = invoke("web", "--port", "8123")
    assert result.exit_code == 0, result.output
    assert seen == {"host": "127.0.0.1", "port": 8123, "log_level": "info"}


# web API surface


@pytest.fixture
def client(workdir: Path):
    from fastapi.testclient import TestClient

    from aforge_flow.web import create_app

    make_image(workdir / "assets" / "a.png", seed=1)
    make_image(workdir / "assets" / "b.png", seed=2)
    (workdir / "assets" / "notes.txt").write_text("x")
    (workdir / "assets" / "sub").mkdir()
    return TestClient(create_app(workdir), base_url="http://127.0.0.1:8000")


def run_job(client, **body) -> dict:
    job_id = client.post(
        "/api/jobs", json={"remove_background": False, "use_gpu": False, **body}
    ).json()["job_id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_browse_lists_dirs_and_supported_files_only(client) -> None:
    entries = client.get("/api/browse", params={"path": "assets"}).json()["entries"]
    names = [e["name"] for e in entries]
    assert names == ["sub", "a.png", "b.png"]
    assert client.get("/api/browse", params={"path": "assets/a.png"}).status_code == 400


def test_file_thumbnail_and_bad_image(client, workdir: Path) -> None:
    resp = client.get("/api/file", params={"path": "assets/a.png", "thumb": True})
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    (workdir / "assets" / "broken.png").write_bytes(b"nope")
    assert (
        client.get("/api/file", params={"path": "assets/broken.png", "thumb": True}).status_code
        == 415
    )
    assert client.get("/api/file", params={"path": "assets/missing.png"}).status_code == 404


def test_files_job_streams_events_and_zips_outputs(client) -> None:
    state = run_job(
        client, mode="files", targets=["assets/a.png", "assets/b.png"], image_format="png"
    )
    assert state["status"] == "completed", state
    assert len(state["outputs"]) == 2

    job_id = state["id"]
    events = client.get(f"/api/jobs/{job_id}/events").text
    assert '"type": "start"' in events and '"type": "done"' in events

    zipped = client.get(f"/api/jobs/{job_id}/download.zip")
    assert zipped.status_code == 200 and zipped.headers["content-type"] == "application/zip"


def test_job_error_paths(client) -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404
    assert client.get("/api/jobs/nope/download.zip").status_code == 404
    assert (
        client.post("/api/jobs", json={"targets": ["assets"], "compression": "zip"}).status_code
        == 400
    )
    assert (
        client.post("/api/jobs", json={"targets": ["assets"], "category": "dragonish"}).status_code
        == 400
    )
    assert (
        client.post("/api/jobs", json={"mode": "files", "targets": ["assets"]}).status_code == 400
    )
    assert client.post("/api/jobs", json={"targets": ["assets/a.png"]}).status_code == 400


def test_empty_job_has_nothing_to_zip(client, workdir: Path) -> None:
    (workdir / "empty").mkdir()
    state = run_job(client, targets=["empty"])
    assert state["status"] == "completed"
    assert client.get(f"/api/jobs/{state['id']}/download.zip").status_code == 404


def test_download_requires_existing_media(client) -> None:
    assert client.get("/api/download", params={"path": "assets/missing.png"}).status_code == 404
    assert json.loads(client.get("/api/health").text)["status"] == "ok"
