"""Shared fixtures: CPU-only fakes for the heavy models plus image helpers.

Nothing here downloads weights or touches a GPU. The fakes replace exactly the
seam where a model would run, so everything around it (path mirroring, file
naming, chaining, sidecars, CLI wiring) is exercised for real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aforge_flow.processors import background_removal as br
from aforge_flow.processors import creature_classification as cc

DIM = 16


def make_image(path: Path, seed: int = 0, size=(64, 48), mode: str = "RGBA") -> Path:
    """Write a seeded-noise image so each seed has a distinct perceptual hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    channels = len(mode)
    arr = rng.integers(0, 256, size=(size[1], size[0], channels), dtype=np.uint8)
    Image.fromarray(arr, mode).save(path)
    return path


@pytest.fixture
def fake_rembg(monkeypatch):
    """Skip rembg sessions/inference; the "mask" makes the left half transparent.

    Returns a dict that records the model name each processor was built with.
    """
    seen: dict = {"models": []}

    def fake_create_session(self, force_cpu: bool = False):
        seen["models"].append(self.model_name)
        return object()

    def fake_segment(self, source, remove_fn, session):
        out = source.convert("RGBA")
        alpha = Image.new("L", out.size, 255)
        alpha.paste(0, (0, 0, out.size[0] // 2, out.size[1]))
        out.putalpha(alpha)
        return out

    monkeypatch.setattr(br.BackgroundRemovalProcessor, "_create_session", fake_create_session)
    monkeypatch.setattr(br.BackgroundRemovalProcessor, "_segment", fake_segment)
    return seen


class FakeEncoder:
    """Deterministic stand-in for :class:`ClipEncoder` (no model, no GPU).

    Text/image embeddings are derived from a stable hash of their content, so
    classification is reproducible. ``image_calls`` records how many images
    were actually embedded, letting tests assert the cache fast path.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        self.hf_id = "fake/encoder"
        self.dim = DIM
        self.image_calls = 0

    @property
    def model_key(self) -> str:
        return f"{self.hf_id}|d{self.dim}"

    @staticmethod
    def _vec(seed_text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed_text)) % (2**32))
        return cc.l2_normalize(rng.standard_normal(DIM).astype(np.float32))

    def encode_texts(self, texts):
        return np.vstack([self._vec("t:" + t) for t in texts])

    def encode_images(self, images):
        self.image_calls += len(images)
        # Seed by pixel bytes so identical pictures get identical vectors.
        return np.vstack([self._vec("i:" + im.tobytes().hex()[:32]) for im in images])


@pytest.fixture
def patched_encoder(monkeypatch):
    """Replace ClipEncoder with FakeEncoder and hand back the instance."""
    holder = {}

    def factory(*args, **kwargs):
        enc = FakeEncoder(*args, **kwargs)
        holder["enc"] = enc
        return enc

    monkeypatch.setattr(cc, "ClipEncoder", factory)
    return holder


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run inside ``tmp_path`` so CLI defaults (``.cache``, ``assets``) stay isolated."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
