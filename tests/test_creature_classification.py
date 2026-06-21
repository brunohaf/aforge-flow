"""Unit tests for the creature classification processor.

These tests run entirely on CPU with **no model downloads**: the heavy
:class:`ClipEncoder` is replaced by a deterministic fake, so they exercise the
taxonomy, the math (normalisation / softmax matching), the persistent
embedding cache, the cache-hit fast path, the sidecar writer and the
``ProcessorResult`` contract without touching a GPU or HuggingFace.

Testing strategy
----------------
* **Pure functions** (`l2_normalize`, `classify_against`, `build_prototypes`)
  are tested directly with hand-built vectors — fast and exact.
* **EmbeddingCache** is tested against a real on-disk SQLite file in a
  ``tmp_path`` to catch serialisation / dim-mismatch / namespacing bugs.
* **The processor** is tested with a fake encoder injected via monkeypatch.
  The fake counts ``encode_images`` calls so we can assert the precompute pass
  makes ``process()`` do zero further GPU work (the core throughput claim).
* A separate, ``@pytest.mark.gpu`` integration test (skipped by default) is
  provided as the template for a real end-to-end check on a machine with the
  model + GPU available.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from aforge_flow.config import ClipModelChoice, Config
from aforge_flow.processors import creature_classification as cc
from aforge_flow.taxonomy import (
    ASSET_TYPE_NORMALIZATION,
    CREATURE_CATEGORY_PROMPTS,
    asset_type_labels,
    creature_category_labels,
)
from aforge_flow.utils.embedding_cache import EmbeddingCache
from PIL import Image

# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #

DIM = 16


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
def config(tmp_path) -> Config:
    return Config(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        classification_enabled=True,
        clip_model=ClipModelChoice.SIGLIP2,
        use_gpu=False,
        classification_sidecar=True,
        vlm_fallback_enabled=False,
    )


def _make_image(path, color=(120, 30, 200), size=(64, 64)):
    """Solid-colour image. Note: solid fills all share a dHash of 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _make_distinct_image(path, seed, size=(64, 64)):
    """Image with seeded random content so its perceptual hash is unique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #

def test_taxonomy_has_14_creature_types():
    assert len(creature_category_labels()) == 14
    assert "Undead" in creature_category_labels()
    assert "Dragon" in creature_category_labels()


def test_every_class_has_prompts():
    for lp in CREATURE_CATEGORY_PROMPTS:
        assert lp.prompts, f"{lp.label} has no prompts"
        # Prompts must be descriptions, not the bare label.
        assert all(p.lower() != lp.label.lower() for p in lp.prompts)


def test_asset_type_normalization_targets_two_buckets():
    assert set(ASSET_TYPE_NORMALIZATION.values()) == {"Portrait", "Token"}
    for label in asset_type_labels():
        assert label in ASSET_TYPE_NORMALIZATION


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #

def test_l2_normalize_unit_length():
    x = np.array([[3.0, 4.0]])
    out = cc.l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-6)


def test_l2_normalize_handles_zero_vector():
    out = cc.l2_normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(out))


def test_classify_against_picks_nearest_prototype():
    protos = cc.l2_normalize(np.eye(3))
    labels = ["a", "b", "c"]
    vec = cc.l2_normalize(np.array([0.1, 0.9, 0.2]))
    label, conf, margin, probs = cc.classify_against(vec, labels, protos, 0.01)
    assert label == "b"
    assert 0.0 <= conf <= 1.0
    assert margin >= 0.0
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)


def test_build_prototypes_shapes():
    enc = FakeEncoder()
    labels, protos = cc.build_prototypes(enc, CREATURE_CATEGORY_PROMPTS)
    assert protos.shape == (14, DIM)
    np.testing.assert_allclose(np.linalg.norm(protos, axis=1), np.ones(14), atol=1e-5)


def test_siglip_text_encoding_uses_dynamic_padding():
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeProcessor:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return FakeInputs(input_ids="fake")

    class FakeFeatures:
        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.eye(2, DIM, dtype=np.float32)

    class FakeModel:
        @staticmethod
        def get_text_features(**_inputs):
            return FakeFeatures()

    enc = cc.ClipEncoder.__new__(cc.ClipEncoder)
    enc.processor = FakeProcessor()
    enc.model = FakeModel()
    enc._torch = FakeTorch()
    enc.device = "cpu"
    enc.is_siglip = True

    out = enc.encode_texts(["short prompt", "a slightly longer prompt"])

    assert out.shape == (2, DIM)
    assert enc.processor.kwargs["padding"] is True
    assert enc.processor.kwargs["truncation"] is True


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #

def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite", model_key="m1", dim=DIM)
    vec = np.arange(DIM, dtype=np.float32)
    cache.put("k1", vec)
    got = cache.get("k1")
    assert got is not None
    np.testing.assert_allclose(got, vec, atol=1e-2)  # float16 storage tolerance
    assert cache.get("missing") is None
    assert cache.count() == 1
    cache.close()


def test_embedding_cache_namespaced_by_model(tmp_path):
    p = tmp_path / "emb.sqlite"
    EmbeddingCache(p, "m1", DIM).put("k", np.ones(DIM, dtype=np.float32))
    # A different model_key must not see m1's vectors.
    other = EmbeddingCache(p, "m2", DIM)
    assert other.get("k") is None
    other.close()


def test_embedding_cache_persists_across_instances(tmp_path):
    p = tmp_path / "emb.sqlite"
    c1 = EmbeddingCache(p, "m1", DIM)
    c1.put("k", np.ones(DIM, dtype=np.float32))
    c1.close()
    c2 = EmbeddingCache(p, "m1", DIM)
    assert c2.get("k") is not None
    c2.close()


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #

def test_process_returns_valid_classification(patched_encoder, config):
    img = _make_image(config.input_dir / "mystery.png")
    proc = cc.CreatureClassificationProcessor(config)

    result = proc.process(img)

    assert result.success
    assert result.output_path == img  # pass-through, pixels untouched
    meta = result.metadata
    assert meta["asset_type"] in {"Portrait", "Token"}
    assert meta["category"] in creature_category_labels()
    assert 0.0 <= meta["confidence"] <= 1.0
    assert meta["source"] == "clip"
    assert meta["model"] == "siglip2"


def test_sidecar_written_with_required_schema(patched_encoder, config):
    img = _make_image(config.input_dir / "mystery.png")
    proc = cc.CreatureClassificationProcessor(config)
    proc.process(img)

    sidecar = img.with_suffix(img.suffix + ".classification.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert set(payload) == {"asset_type", "category", "confidence", "model", "source"}


def test_precompute_makes_process_a_cache_hit(patched_encoder, config):
    paths = [_make_distinct_image(config.input_dir / f"img_{i}.png", seed=i) for i in range(5)]
    proc = cc.CreatureClassificationProcessor(config)
    enc = patched_encoder["enc"]

    computed = proc.precompute_embeddings(paths)
    assert computed == 5
    assert enc.image_calls == 5

    # Per-file processing must not embed any image again.
    for p in paths:
        proc.process(p)
    assert enc.image_calls == 5


def test_replicas_share_one_embedding(patched_encoder, config):
    # Two byte-identical pictures => one perceptual hash => one encode call.
    a = _make_image(config.input_dir / "a.png", color=(10, 20, 30))
    b = _make_image(config.input_dir / "b.png", color=(10, 20, 30))
    proc = cc.CreatureClassificationProcessor(config)
    enc = patched_encoder["enc"]

    computed = proc.precompute_embeddings([a, b])
    assert computed == 1
    assert enc.image_calls == 1


def test_unsupported_file_cannot_be_processed(patched_encoder, config):
    proc = cc.CreatureClassificationProcessor(config)
    assert not proc.can_process(config.input_dir / "notes.txt")


def test_missing_file_returns_failure(patched_encoder, config):
    proc = cc.CreatureClassificationProcessor(config)
    result = proc.process(config.input_dir / "does_not_exist.png")
    assert not result.success
    assert result.error


@pytest.mark.gpu
@pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS"),
    reason="real model + CUDA; set RUN_GPU_TESTS=1 to run",
)
def test_real_encoder_end_to_end(config):  # pragma: no cover - needs GPU + weights
    """Template integration test — run with: RUN_GPU_TESTS=1 pytest -m gpu.

    Requires the SigLIP 2 weights and a CUDA device; deselected by default.
    """
    config.use_gpu = True
    img = _make_image(config.input_dir / "dragon.png")
    proc = cc.CreatureClassificationProcessor(config)
    result = proc.process(img)
    assert result.success
    assert result.metadata["category"] in creature_category_labels()
