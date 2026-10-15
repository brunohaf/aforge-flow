"""Unit tests for processor internals: edge refinement, OOM recovery, video encoding."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aforge_flow.config import CompressionLevel, Config, MediaFormat, RembgModel
from aforge_flow.processors import background_removal as br
from aforge_flow.processors import media_optimization as mo


def rgba(alpha: np.ndarray, rgb=(200, 50, 50)) -> Image.Image:
    h, w = alpha.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        use_gpu=False,
    )


# background removal: pure helpers


def test_is_oom_error_matches_known_markers() -> None:
    assert br._is_oom_error(RuntimeError("CUDA out of memory"))
    assert not br._is_oom_error(ValueError("bad shape"))


def test_resize_for_inference_only_downscales_large_images() -> None:
    small = Image.new("RGB", (100, 50))
    assert br.resize_for_inference(small, max_dim=200) == (small, (100, 50))

    resized, original = br.resize_for_inference(Image.new("RGB", (400, 200)), max_dim=200)
    assert original == (400, 200)
    assert resized.size == (200, 100)


def test_composite_mask_upscales_alpha_onto_original() -> None:
    original = Image.new("RGB", (8, 8), (10, 20, 30))
    mask = Image.new("L", (4, 4), 128)
    out = br.composite_mask(original, mask)
    assert out.mode == "RGBA"
    assert out.size == (8, 8)
    assert out.getpixel((0, 0))[:3] == (10, 20, 30)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_sharpen_alpha_pushes_values_away_from_the_middle(strength: float) -> None:
    alpha = np.array([[10, 100, 128, 160, 250]], dtype=np.uint8)
    out = np.array(br.sharpen_alpha(rgba(alpha), strength=strength).getchannel("A"))
    if strength == 0.0:
        assert (out == alpha).all()
    else:
        assert out[0, 0] <= alpha[0, 0] and out[0, -1] >= alpha[0, -1]
        assert out[0, 1] < 100 < 160 < out[0, 3]


def test_decontaminate_edges_recolours_fringe_from_opaque_neighbours() -> None:
    alpha = np.array([[255, 255, 120, 0]], dtype=np.uint8)
    img = rgba(alpha)
    arr = np.array(img)
    arr[0, 2, :3] = (255, 255, 255)
    out = np.array(br.decontaminate_edges(Image.fromarray(arr, "RGBA"), radius=2))
    assert tuple(out[0, 2, :3]) != (255, 255, 255)


def test_decontaminate_edges_is_noop_without_fringe() -> None:
    img = rgba(np.full((3, 3), 255, dtype=np.uint8))
    assert br.decontaminate_edges(img) is img


def test_decontaminate_edges_wip_handles_white_bleed() -> None:
    alpha = np.zeros((5, 5), dtype=np.uint8)
    alpha[1:4, 1:4] = 255
    arr = np.array(rgba(alpha))
    arr[1, 1, :3] = (255, 255, 255)
    out = br.decontaminate_edges_wip(Image.fromarray(arr, "RGBA"), radius=1)
    assert out.mode == "RGBA"


def test_erode_alpha_trims_the_border() -> None:
    alpha = np.zeros((7, 7), dtype=np.uint8)
    alpha[1:6, 1:6] = 255
    out = np.array(br.erode_alpha(rgba(alpha), pixels=1).getchannel("A"))
    assert out[1, 1] == 0 and out[3, 3] == 255


def test_apply_alpha_matting_returns_full_size_rgba(monkeypatch) -> None:
    monkeypatch.setattr(br, "MAX_INFERENCE_DIM", 64)
    alpha = np.zeros((96, 96), dtype=np.uint8)
    alpha[24:72, 24:72] = 255
    original = Image.new("RGB", (96, 96), (90, 90, 90))
    out = br.apply_alpha_matting(original, rgba(alpha), erode_structure_size=3)
    assert out.size == (96, 96)
    assert out.mode == "RGBA"


# background removal: processor behaviour


class FakeSession:
    pass


@pytest.fixture
def processor(config: Config, monkeypatch) -> br.BackgroundRemovalProcessor:
    monkeypatch.setattr(
        br.BackgroundRemovalProcessor,
        "_create_session",
        lambda self, force_cpu=False: FakeSession(),
    )
    return br.BackgroundRemovalProcessor(config)


def test_build_providers_cpu_and_gpu(processor: br.BackgroundRemovalProcessor) -> None:
    assert processor._build_providers(force_cpu=True) == ["CPUExecutionProvider"]

    processor.use_gpu = True
    processor.config.gpu_mem_limit_gb = 2
    (name, opts), cpu = processor._build_providers(force_cpu=False)
    assert name == "CUDAExecutionProvider" and cpu == "CPUExecutionProvider"
    assert opts["gpu_mem_limit"] == 2 * 1024**3


def test_create_session_propagates_rembg_errors(config: Config, monkeypatch) -> None:
    import rembg

    def boom(*_a, **_k):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(rembg, "new_session", boom)
    with pytest.raises(RuntimeError, match="model download failed"):
        br.BackgroundRemovalProcessor(config)


def test_create_session_passes_model_and_providers(config: Config, monkeypatch) -> None:
    import rembg

    seen = {}

    def fake_new_session(model, providers):
        seen.update(model=model, providers=providers)
        return FakeSession()

    monkeypatch.setattr(rembg, "new_session", fake_new_session)
    config.rembg_model = RembgModel.U2NETP
    br.BackgroundRemovalProcessor(config)
    assert seen == {"model": "u2netp", "providers": ["CPUExecutionProvider"]}


def test_cpu_session_is_cached(processor: br.BackgroundRemovalProcessor) -> None:
    assert processor._get_cpu_session() is processor._get_cpu_session()


def test_flush_gpu_memory_drops_the_session(processor: br.BackgroundRemovalProcessor) -> None:
    processor._flush_gpu_memory()
    assert processor.session is None


def _source(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    return path


def test_oom_rebuilds_session_then_falls_back_to_cpu(processor, config, monkeypatch) -> None:
    monkeypatch.setattr(br.time, "sleep", lambda _s: None)
    calls = []

    def segment(self, source, remove_fn, session):
        calls.append(session)
        if len(calls) <= 2:
            raise RuntimeError("CUDA out of memory")
        return source.convert("RGBA")

    monkeypatch.setattr(br.BackgroundRemovalProcessor, "_segment", segment)
    src = _source(config.input_dir / "a.png")
    result = processor.process(src)

    assert result.success, result.error
    assert result.metadata["device"] == "cpu_fallback"
    assert len(calls) == 3


def test_oom_without_cpu_fallback_fails(processor, config, monkeypatch) -> None:
    monkeypatch.setattr(br.time, "sleep", lambda _s: None)
    config.cpu_fallback_on_oom = False

    def segment(self, *_a):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(br.BackgroundRemovalProcessor, "_segment", segment)
    result = processor.process(_source(config.input_dir / "a.png"))
    assert not result.success and "after OOM recovery" in result.error


def test_non_oom_error_fails_immediately(processor, config, monkeypatch) -> None:
    def segment(self, *_a):
        raise ValueError("bad input")

    monkeypatch.setattr(br.BackgroundRemovalProcessor, "_segment", segment)
    result = processor.process(_source(config.input_dir / "a.png"))
    assert not result.success and "bad input" in result.error


def test_process_skips_existing_and_rejects_bad_files(processor, config) -> None:
    src = _source(config.input_dir / "a.png")
    existing = config.output_dir / "a_nobg.png"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"x")
    assert processor.process(src).metadata == {"skipped": True}

    broken = config.input_dir / "broken.png"
    broken.write_bytes(b"not an image")
    assert not processor.process(broken).success
    assert not processor.process(config.input_dir / "missing.png").success


def test_segment_composites_smaller_masks(processor) -> None:
    source = Image.new("RGB", (8, 8))
    out = processor._segment(source, lambda img, session: Image.new("RGBA", (4, 4)), FakeSession())
    assert out.size == (8, 8)


def test_refine_edges_runs_every_enabled_stage(processor, monkeypatch) -> None:
    stages = []
    monkeypatch.setattr(br, "apply_alpha_matting", lambda s, m: stages.append("matting") or m)
    monkeypatch.setattr(br, "sharpen_alpha", lambda m, strength: stages.append("sharpen") or m)
    monkeypatch.setattr(br, "decontaminate_edges", lambda m, radius: stages.append("decon") or m)
    monkeypatch.setattr(br, "erode_alpha", lambda m, pixels: stages.append("erode") or m)
    cfg = processor.config
    cfg.alpha_matting, cfg.edge_decontamination_radius, cfg.erode_alpha = True, 1, True
    mask = Image.new("RGBA", (2, 2))
    assert processor._refine_edges(mask, mask) is mask
    assert stages == ["matting", "sharpen", "decon", "erode"]


def test_refine_edges_survives_matting_failure(processor, monkeypatch) -> None:
    def fail(*_a):
        raise ImportError("pymatting missing")

    monkeypatch.setattr(br, "apply_alpha_matting", fail)
    processor.config.alpha_matting = True
    processor.config.alpha_edge_sharpening = 0
    mask = Image.new("RGBA", (2, 2))
    assert processor._refine_edges(mask, mask) is mask


def test_save_failure_is_reported(processor, config, tmp_path: Path) -> None:
    src = _source(config.input_dir / "a.png")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    result = processor._save(src, Image.new("RGBA", (2, 2)), blocker / "sub" / "out.png")
    assert not result.success and "Failed to save" in result.error


# media optimization: video


@pytest.fixture
def video(config: Config) -> Path:
    path = config.input_dir / "clip.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _ffmpeg_ok(cmd, **_kwargs):
    Path(cmd[-1]).write_bytes(b"\x01" * 16)
    return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.mark.parametrize(
    "fmt, gpu, nvenc, encoder, audio",
    [
        (MediaFormat.WEBM, True, True, "av1_nvenc", "libopus"),
        (MediaFormat.WEBM, True, False, "libvpx-vp9", "libopus"),
        (MediaFormat.WEBM, False, True, "libvpx-vp9", "libopus"),
        (MediaFormat.MP4, True, True, "libx264", "aac"),
    ],
)
def test_video_encoder_selection_and_command(
    config: Config, video: Path, monkeypatch, fmt, gpu, nvenc, encoder, audio
) -> None:
    monkeypatch.setattr(mo, "_encoder_available", lambda name: nvenc)
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _ffmpeg_ok(cmd, **kwargs)

    monkeypatch.setattr(mo.subprocess, "run", run)
    config.use_gpu = gpu
    proc = mo.MediaOptimizationProcessor(config, video_format=fmt)
    result = proc.process(video)

    assert result.success, result.error
    assert result.output_path.suffix == f".{fmt.value}"
    cmd = seen["cmd"]
    assert cmd[cmd.index("-c:v") + 1] == encoder
    assert cmd[cmd.index("-c:a") + 1] == audio


@pytest.mark.parametrize(
    "error, message",
    [
        (subprocess.TimeoutExpired("ffmpeg", 1), "timeout"),
        (subprocess.CalledProcessError(1, "ffmpeg", stderr="boom"), "FFmpeg error: boom"),
        (OSError("no ffmpeg"), "Unexpected error"),
    ],
)
def test_video_failures_are_reported(config, video, monkeypatch, error, message) -> None:
    def run(*_a, **_k):
        raise error

    monkeypatch.setattr(mo.subprocess, "run", run)
    monkeypatch.setattr(mo, "_encoder_available", lambda name: False)
    result = mo.MediaOptimizationProcessor(config, video_format=MediaFormat.MKV).process(video)
    assert not result.success and message in result.error


def test_video_skip_existing(config, video, monkeypatch) -> None:
    out = config.output_dir / "clip_optimized.webm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"done")
    monkeypatch.setattr(mo.subprocess, "run", lambda *a, **k: pytest.fail("must not encode"))
    result = mo.MediaOptimizationProcessor(config).process(video)
    assert result.metadata == {"skipped": True}


def test_encoder_probe_parses_ffmpeg_output(monkeypatch) -> None:
    mo._encoder_available.cache_clear()
    monkeypatch.setattr(
        mo.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, " V..... av1_nvenc  NVIDIA AV1", ""),
    )
    assert mo._encoder_available("av1_nvenc")
    mo._encoder_available.cache_clear()

    def missing(*_a, **_k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(mo.subprocess, "run", missing)
    assert not mo._encoder_available("av1_nvenc")
    mo._encoder_available.cache_clear()


def test_image_skip_existing_and_failures(config: Config, monkeypatch) -> None:
    src = config.input_dir / "a.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4)).save(src)
    config.compression_level = CompressionLevel.AGGRESSIVE
    proc = mo.MediaOptimizationProcessor(config)
    assert proc.process(src).success
    assert proc.process(src).metadata == {"skipped": True}

    broken = config.input_dir / "broken.png"
    broken.write_bytes(b"nope")
    assert "Image processing failed" in proc.process(broken).error
    assert not proc.process(config.input_dir / "missing.png").success
