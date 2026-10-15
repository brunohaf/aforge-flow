"""End-to-end tests for every ``aforge`` workflow, driven through the real CLI.

Models are faked at the inference seam (see ``conftest.py``); file discovery,
path mirroring, naming, chaining, encoding and CLI wiring all run for real.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import ClassVar, cast

import pytest
from click.testing import CliRunner
from conftest import make_image
from PIL import Image

from aforge_flow import cli
from aforge_flow.config import RembgModel


def alpha_at(image: Image.Image, xy: tuple[int, int]) -> int:
    """Alpha value of one pixel (0 = fully transparent)."""
    return cast(int, image.getchannel("A").getpixel(xy))


def run(*args: str) -> str:
    """Invoke ``aforge <args>`` in-process and return stdout; fail loudly on error."""
    result = CliRunner().invoke(cli.main, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture
def assets(workdir: Path) -> Path:
    """``assets/`` with two top-level images and one nested one."""
    root = workdir / "assets"
    make_image(root / "Balhannoth.png", seed=1)
    make_image(root / "Brain in a Jar.png", seed=2)
    make_image(root / "items" / "Longsword.png", seed=3)
    return root


def test_optimize_png_lossless_is_pixel_exact(assets: Path, workdir: Path) -> None:
    run(
        "optimize",
        "-i",
        "assets",
        "-o",
        "output",
        "--image-format",
        "png",
        "--compression",
        "lossless",
        "--cpu",
    )

    src = Image.open(assets / "Balhannoth.png")
    out = Image.open(workdir / "output" / "Balhannoth_optimized.png")
    assert out.mode == "RGBA", "alpha channel must survive"
    assert out.tobytes() == src.tobytes()


def test_optimize_mirrors_subdirectories(assets: Path, workdir: Path) -> None:
    run("optimize", "-i", "assets", "-o", "output", "--cpu")

    assert (workdir / "output" / "items" / "Longsword_optimized.webp").is_file()
    assert len(list((workdir / "output").rglob("*.webp"))) == 3


def test_optimize_jpeg_flattens_alpha(assets: Path, workdir: Path) -> None:
    run("optimize", "-i", "assets", "-o", "output", "--image-format", "jpeg", "--cpu")

    out = Image.open(workdir / "output" / "Balhannoth_optimized.jpeg")
    assert out.mode == "RGB"


def test_optimize_skip_existing_does_not_rewrite(assets: Path, workdir: Path) -> None:
    run("optimize", "-i", "assets", "-o", "output", "--image-format", "png", "--cpu")
    target = workdir / "output" / "Balhannoth_optimized.png"
    first = target.stat().st_mtime_ns

    time.sleep(0.05)
    run("optimize", "-i", "assets", "-o", "output", "--image-format", "png", "--cpu")

    assert target.stat().st_mtime_ns == first


def test_remove_background_writes_transparent_png(
    assets: Path, workdir: Path, fake_rembg: dict
) -> None:
    run("remove-background", "-i", "assets", "-o", "output", "--cpu")

    out = Image.open(workdir / "output" / "Balhannoth_nobg.png")
    assert out.mode == "RGBA"
    assert alpha_at(out, (0, 0)) == 0, "masked region must be transparent"
    assert alpha_at(out, (out.width - 1, 0)) == 255
    assert (workdir / "output" / "items" / "Longsword_nobg.png").is_file()


@pytest.mark.parametrize(
    "category, model",
    [
        ("item", RembgModel.BIREFNET_DIS),
        ("portrait", RembgModel.BIREFNET_PORTRAIT),
        ("creature", RembgModel.BRIA_RMBG),
    ],
)
def test_remove_background_routes_model_by_category(
    assets: Path, fake_rembg: dict, category: str, model: RembgModel
) -> None:
    run("remove-background", "-i", "assets", "-o", "output", "--cpu", "--category", category)

    assert fake_rembg["models"] == [model.value]


def test_remove_background_explicit_model_overrides_category(
    assets: Path, fake_rembg: dict
) -> None:
    run(
        "remove-background",
        "-i",
        "assets",
        "-o",
        "output",
        "--cpu",
        "--category",
        "item",
        "--model",
        "isnet-anime",
    )

    assert fake_rembg["models"] == ["isnet-anime"]


def test_process_chains_background_removal_into_optimization(
    assets: Path, workdir: Path, fake_rembg: dict
) -> None:
    run("process", "-i", "assets", "-o", "output", "--image-format", "png", "--cpu")

    final = workdir / "output" / "Balhannoth_nobg_optimized.png"
    assert final.is_file()
    assert alpha_at(Image.open(final), (0, 0)) == 0, "transparency must survive optimization"


@pytest.mark.parametrize(
    "in_dir, out_dir, final",
    [
        (".", "output", "output/assets/items/Longsword_nobg_optimized.png"),
        ("assets", "assets/out", "assets/out/items/Longsword_nobg_optimized.png"),
    ],
)
def test_process_with_output_nested_in_input_keeps_one_level(
    assets: Path, workdir: Path, fake_rembg: dict, in_dir: str, out_dir: str, final: str
) -> None:
    run("process", "-i", in_dir, "-o", out_dir, "--image-format", "png", "--cpu")

    assert (workdir / final).is_file()
    assert not list((workdir / out_dir).rglob(f"{Path(out_dir).name}/**/*_optimized.png"))


@pytest.mark.parametrize("alias", ["go", "pipe"])
def test_process_aliases(assets: Path, workdir: Path, fake_rembg: dict, alias: str) -> None:
    run(alias, "-i", "assets", "-o", "output", "--no-background-removal", "--cpu")

    assert (workdir / "output" / "Balhannoth_optimized.webp").is_file()


class _RecordingPipeline:
    """Captures the Config ``process`` builds instead of running anything."""

    configs: ClassVar[list] = []

    def __init__(self, config) -> None:
        _RecordingPipeline.configs.append(config)

    def setup(self) -> None: ...
    def process_directory(self, **_kwargs) -> None: ...
    def print_summary(self) -> None: ...


@pytest.fixture
def recorded(monkeypatch) -> list:
    _RecordingPipeline.configs = []
    monkeypatch.setattr(cli, "AssetPipeline", _RecordingPipeline)
    return _RecordingPipeline.configs


def test_process_tabletop_profile_enables_classify_and_naming(assets: Path, recorded: list) -> None:
    run("process", "-i", "assets", "--profile", "tabletop")

    assert recorded[0].classification_enabled is True
    assert recorded[0].caption_rename_enabled is True


def test_process_explicit_flag_beats_profile(assets: Path, recorded: list) -> None:
    run("process", "-i", "assets", "--profile", "tabletop", "--no-classify")

    assert recorded[0].classification_enabled is False
    assert recorded[0].caption_rename_enabled is True


def test_process_default_profile_is_opt_in(assets: Path, recorded: list) -> None:
    run("process", "-i", "assets")

    assert recorded[0].classification_enabled is False
    assert recorded[0].caption_rename_enabled is False


def test_rename_keeps_meaningful_names_without_loading_a_model(
    assets: Path, workdir: Path, monkeypatch
) -> None:
    from aforge_flow.processors import caption_naming

    def no_model(*_args, **_kwargs):
        raise AssertionError("captioning model must not load for meaningful names")

    monkeypatch.setattr(caption_naming, "_make_backend", no_model)
    run("rename", "-i", "assets", "-o", "output", "--cpu")

    out = workdir / "output"
    assert (out / "balhannoth-1.png").is_file()
    assert (out / "brain-jar-1.png").is_file()
    assert (out / "items" / "longsword-1.png").is_file()


def test_rename_captions_junk_names(workdir: Path, monkeypatch) -> None:
    from aforge_flow.processors import caption_naming

    class FakeBackend:
        name = "fake"
        slug_prefix = ""

        def caption(self, _image) -> str:
            return "beholder"

    monkeypatch.setattr(caption_naming, "_make_backend", lambda *_a, **_k: FakeBackend())
    make_image(workdir / "assets" / "900297781783770851.png", seed=7)

    run("caption-rename", "-i", "assets", "-o", "output", "--cpu")

    assert (workdir / "output" / "beholder-1.png").is_file()


def test_classify_writes_sidecars(assets: Path, patched_encoder: dict) -> None:
    run("classify", "-i", "assets", "-o", "output", "--cpu")

    sidecars = sorted(assets.rglob("*.classification.json"))
    assert len(sidecars) == 3, "one sidecar per image, next to the input"
    payload = json.loads((assets / "Balhannoth.png.classification.json").read_text("utf-8"))
    assert {"asset_type", "confidence"} <= payload.keys()
    assert 0.0 <= payload["confidence"] <= 1.0


def test_recommend_json_reports_workload(assets: Path) -> None:
    payload = json.loads(run("recommend", "-i", "assets", "--json"))

    assert payload["command_line"].startswith("aforge optimize")
    assert payload["workload"]["num_images"] == 3


def test_info_prints_version() -> None:
    from aforge_flow import __version__

    assert __version__ in run("info")


def test_help_lists_every_command() -> None:
    out = run("-h")
    for command in (
        "process",
        "remove-background",
        "optimize",
        "caption-rename",
        "classify",
        "recommend",
        "info",
        "web",
    ):
        assert command in out


@pytest.fixture
def client(assets: Path, workdir: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aforge_flow.web import create_app

    return TestClient(create_app(workdir), base_url="http://127.0.0.1:8000")


def test_web_health_and_options(client) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    assert "rembg_models" in client.get("/api/options").json()


def test_web_browse_rejects_path_traversal(client) -> None:
    assert client.get("/api/browse", params={"path": "assets"}).status_code == 200
    assert client.get("/api/browse", params={"path": "../"}).status_code == 403


def test_web_job_optimizes_a_directory(client, workdir: Path) -> None:
    job = client.post(
        "/api/jobs",
        json={
            "mode": "directory",
            "targets": ["assets"],
            "output_dir": "output",
            "remove_background": False,
            "image_format": "png",
            "use_gpu": False,
        },
    ).json()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job['job_id']}").json()
        if state["status"] != "running":
            break
        time.sleep(0.05)

    assert state["status"] == "completed", state
    assert (workdir / "output" / "Balhannoth_optimized.png").is_file()


def test_web_refuses_non_media_files(client, workdir: Path) -> None:
    (workdir / ".env").write_text("SECRET=1", encoding="utf-8")
    for route in ("/api/file", "/api/download"):
        assert client.get(route, params={"path": ".env"}).status_code == 403
        assert client.get(route, params={"path": "assets/Balhannoth.png"}).status_code == 200


def test_web_rejects_foreign_host_header(client) -> None:
    resp = client.get("/api/health", headers={"Host": "rebind.evil.example:8000"})
    assert resp.status_code == 400


def test_web_rejects_cross_site_origin(client) -> None:
    assert client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"Origin": "http://localhost:5173"}).status_code == 200


def test_web_upload_rejects_unsupported_type_without_creating_dirs(client, workdir: Path) -> None:
    resp = client.post("/api/upload", files={"files": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400
    assert not (workdir / ".uploads").exists()


def test_web_upload_stores_media_under_uploads(client, workdir: Path) -> None:
    png = (workdir / "assets" / "Balhannoth.png").read_bytes()
    resp = client.post("/api/upload", files={"files": ("../../x.png", png, "image/png")})
    assert resp.status_code == 200
    (target,) = resp.json()["targets"]
    assert Path(target).parts[0] == ".uploads"
    assert (workdir / target).read_bytes() == png


def test_web_upload_caps_file_count(client, monkeypatch) -> None:
    from aforge_flow.web.routes import upload

    monkeypatch.setattr(upload, "MAX_FILES", 1)
    files = [("files", (f"{i}.png", b"x", "image/png")) for i in range(2)]
    assert client.post("/api/upload", files=files).status_code == 413


def test_web_upload_size_cap_removes_partial_batch(client, workdir: Path, monkeypatch) -> None:
    from aforge_flow.web.routes import upload

    monkeypatch.setattr(upload, "MAX_FILE_BYTES", 4)
    files = [
        ("files", ("a.png", b"ok", "image/png")),
        ("files", ("b.png", b"too big", "image/png")),
    ]
    assert client.post("/api/upload", files=files).status_code == 413
    assert not any((workdir / ".uploads").rglob("*.png"))


def test_web_job_rejects_workers_above_cpu_count(client) -> None:
    import os

    resp = client.post(
        "/api/jobs", json={"targets": ["assets"], "workers": (os.cpu_count() or 1) + 1}
    )
    assert resp.status_code == 422
