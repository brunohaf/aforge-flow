"""Hardware-aware recommendation of pipeline performance settings.

Inspects the host machine (CPU, RAM, GPU/VRAM, NVENC support) and the target
input directory (image vs video mix, sizes) and recommends safe, performant
values for ``--workers`` and ``--gpu/--cpu`` without risking OOM or encoder
session exhaustion.

These functions are pure data-gatherers/calculators — they do not print. The CLI
layer is responsible for rendering. See the ``recommend`` command and the
``--auto-tune`` flag in ``cli.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from aforge_flow.utils.validators import (
    get_files_recursively,
    is_image,
    is_video,
)

# libvpx-vp9 is already multithreaded (we pass ``-row-mt 1``), so running many
# workers oversubscribes the CPU. Allot roughly one worker per 4 logical cores.
CPU_CORES_PER_VP9_WORKER = 4
MAX_CPU_VIDEO_WORKERS = 4

# Rough peak RAM per concurrent 1080p VP9 encode, used to cap workers so a big
# batch does not exhaust system memory.
VP9_RAM_PER_WORKER_GB = 1.5
RAM_BUDGET_FRACTION = 0.7  # only plan to use this fraction of *available* RAM

# Image (Pillow/webp) work is light; allow more parallelism but stay bounded.
MAX_IMAGE_WORKERS = 8

# NVENC is a single fixed-function block shared by all jobs, so extra workers do
# not increase encode throughput — they only add contention and VRAM pressure.
GPU_VIDEO_WORKERS = 1
GPU_VIDEO_WORKERS_WITH_IMAGES = 2  # overlap CPU image work with GPU encodes

# Full-pipeline ML models (rembg + CLIP/SigLIP) need GPU memory headroom.
LOW_VRAM_WARN_GB = 3.0

# Warn about big single files driving long, memory-heavy encodes.
LARGE_VIDEO_WARN_BYTES = 500 * 1024 * 1024


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@dataclass
class HardwareInfo:
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    has_cuda: bool
    gpu_name: str | None
    total_vram_gb: float | None
    free_vram_gb: float | None
    has_av1_nvenc: bool


@dataclass
class WorkloadInfo:
    num_images: int
    num_videos: int
    total_video_bytes: int
    largest_video_bytes: int


@dataclass
class Recommendation:
    use_gpu: bool
    workers: int
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareInfo:
    """Probe CPU, RAM, GPU/VRAM, and NVENC AV1 support.

    Degrades gracefully when psutil/torch/CUDA are unavailable.
    """
    import psutil

    # Imported lazily to avoid a utils <-> processors import cycle.
    from aforge_flow.processors.media_optimization import _encoder_available

    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    has_cuda = False
    gpu_name: str | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024**3)
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_vram_gb = free_bytes / (1024**3)
            except Exception:
                free_vram_gb = None
    except Exception:
        pass

    return HardwareInfo(
        physical_cores=physical,
        logical_cores=logical,
        total_ram_gb=round(total_ram_gb, 1),
        available_ram_gb=round(available_ram_gb, 1),
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        total_vram_gb=round(total_vram_gb, 1) if total_vram_gb is not None else None,
        free_vram_gb=round(free_vram_gb, 1) if free_vram_gb is not None else None,
        has_av1_nvenc=_encoder_available("av1_nvenc"),
    )


def scan_workload(input_dir: Path) -> WorkloadInfo:
    """Count images/videos and total/largest video size under ``input_dir``."""
    num_images = 0
    num_videos = 0
    total_video_bytes = 0
    largest_video_bytes = 0

    if input_dir.exists():
        for path in get_files_recursively(input_dir, supported_only=True):
            if is_image(path):
                num_images += 1
            elif is_video(path):
                num_videos += 1
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                total_video_bytes += size
                largest_video_bytes = max(largest_video_bytes, size)

    return WorkloadInfo(
        num_images=num_images,
        num_videos=num_videos,
        total_video_bytes=total_video_bytes,
        largest_video_bytes=largest_video_bytes,
    )


def _ram_capped(workers: int, available_ram_gb: float) -> int:
    budget = available_ram_gb * RAM_BUDGET_FRACTION
    ram_limit = max(1, int(budget // VP9_RAM_PER_WORKER_GB))
    return min(workers, ram_limit)


def recommend_settings(
    hw: HardwareInfo,
    workload: WorkloadInfo,
    video_format: str,
    full_pipeline: bool = False,
) -> Recommendation:
    """Recommend ``use_gpu`` and ``workers`` from hardware + workload.

    GPU acceleration today only benefits the WebM path (av1_nvenc); mp4/mkv are
    still CPU libx264, so ``use_gpu`` is reported False (with a note) for them.
    """
    notes: list[str] = []
    warnings: list[str] = []

    has_images = workload.num_images > 0
    has_videos = workload.num_videos > 0
    is_webm = video_format == "webm"

    if not hw.has_cuda:
        use_gpu = False
        notes.append("No CUDA GPU detected — using CPU encoders.")
    elif not is_webm:
        use_gpu = False
        notes.append(
            f"--video-format {video_format} encodes on CPU (libx264) today; "
            "--gpu only accelerates the WebM (av1_nvenc) path. Use --video-format "
            "webm to engage the GPU."
        )
    elif not hw.has_av1_nvenc:
        use_gpu = False
        notes.append(
            "GPU present but av1_nvenc is unavailable in this ffmpeg build — "
            "falling back to CPU libvpx-vp9 for WebM."
        )
    else:
        use_gpu = True
        notes.append(f"GPU AV1 NVENC ({hw.gpu_name}) available for WebM encoding.")

    if has_videos and use_gpu:
        workers = GPU_VIDEO_WORKERS_WITH_IMAGES if has_images else GPU_VIDEO_WORKERS
        notes.append(
            "NVENC is a single hardware block, so video encoding does not scale "
            f"with workers — {workers} keeps it fed without contention."
        )
    elif has_videos:
        workers = _clamp(hw.physical_cores // CPU_CORES_PER_VP9_WORKER, 1, MAX_CPU_VIDEO_WORKERS)
        capped = _ram_capped(workers, hw.available_ram_gb)
        if capped < workers:
            warnings.append(
                f"Reduced workers {workers}->{capped}: only "
                f"{hw.available_ram_gb:.1f} GB RAM available "
                f"(~{VP9_RAM_PER_WORKER_GB} GB/worker for CPU video encode)."
            )
            workers = capped
        notes.append(
            "CPU video encoding is multithreaded; fewer workers with more threads "
            "each beats many contending workers."
        )
    else:
        workers = _clamp(hw.physical_cores, 1, MAX_IMAGE_WORKERS)
        workers = _ram_capped(workers, hw.available_ram_gb)

    if full_pipeline and hw.has_cuda and hw.free_vram_gb is not None:
        if hw.free_vram_gb < LOW_VRAM_WARN_GB:
            warnings.append(
                f"Only {hw.free_vram_gb:.1f} GB VRAM free; the full pipeline loads "
                "rembg + CLIP models on the GPU. Consider fewer workers or --cpu for "
                "the ML steps to avoid CUDA OOM."
            )

    if has_videos and workers > 1 and workload.largest_video_bytes >= LARGE_VIDEO_WARN_BYTES:
        warnings.append(
            f"Largest video is {workload.largest_video_bytes / (1024**3):.1f} GB; "
            "with multiple workers these large encodes run concurrently and raise "
            "peak memory use."
        )

    return Recommendation(use_gpu=use_gpu, workers=workers, notes=notes, warnings=warnings)


def build_command_line(
    command: str,
    input_dir: str,
    output_dir: str,
    video_format: str,
    rec: Recommendation,
) -> str:
    """Render a ready-to-paste aforge command reflecting the recommendation."""
    gpu_flag = "--gpu" if rec.use_gpu else "--cpu"
    return (
        f"aforge {command} -i {input_dir} -o {output_dir} "
        f"--video-format {video_format} {gpu_flag} --workers {rec.workers}"
    )


def as_dict(hw: HardwareInfo, workload: WorkloadInfo, rec: Recommendation) -> dict:
    """Combine all three records into a JSON-serializable dict."""
    return {
        "hardware": asdict(hw),
        "workload": asdict(workload),
        "recommendation": asdict(rec),
    }
