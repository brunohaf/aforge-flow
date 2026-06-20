"""Creature classification processor (zero-shot, CLIP-family).

Classifies fantasy artwork purely from **image content** — file names,
folders and metadata are treated as unreliable and never consulted.  Two
levels are produced per asset:

* **Level 1 — asset type:** normalised to ``"Portrait"`` or ``"Token"``.
* **Level 2 — creature category:** one of the 14 official D&D creature types.

Both levels use the same recipe: a CLIP-family vision encoder embeds the
image once; the label set is represented by *prompt-ensembled* text
prototypes (see :mod:`aforge_flow.taxonomy`); classification is the
nearest prototype by cosine similarity.  Image embeddings are cached in a
persistent vector store keyed by picture content, so a second run (or a
re-run after tweaking thresholds/prompts) does **zero** GPU work.

When the top prediction is not confident enough, an optional Vision-Language
Model (Qwen2.5-VL by default) is consulted for that single image only.

Design notes
------------
* The processor is a *pass-through*: it does not modify pixels.  Its
  ``ProcessorResult.output_path`` equals the input path so the pipeline chain
  continues unchanged; the classification is attached to ``metadata`` and,
  optionally, written as a ``<image>.classification.json`` sidecar.
* Throughput comes from :meth:`precompute_embeddings`, which batches the whole
  input set through the encoder before the per-file pass.  ``process()`` then
  only does a cached lookup + a cheap matmul (+ rare VLM fallback).

Recommended models (trade-offs)
-------------------------------
=====================  =======  ===================================  ==========================
Encoder                ~VRAM    Strength for fantasy art             When to pick
=====================  =======  ===================================  ==========================
SigLIP 2 so400m/384    ~1.5 GB  Best zero-shot accuracy, robust to   **Default.** Best quality
(``siglip2``)                   stylised art, strong text tower      that still fits a 4070.
SigLIP so400m/384      ~1.3 GB  Very strong, slightly below v2       If siglip2 weights absent.
OpenAI CLIP ViT-L/14   ~1.2 GB  Ubiquitous, weaker on long prompts   Baseline / comparison.
MetaCLIP-2 / EVA-CLIP  ~2-4 GB  Competitive, larger                  Accuracy headroom on 12 GB.
=====================  =======  ===================================  ==========================

Fallback VLM: **Qwen2.5-VL-7B-Instruct** (4-bit) gives the best
taxonomy reasoning on a 4070 for the uncertain tail.  Florence-2 is far
cheaper but its closed task vocabulary makes it weak at the abstract D&D
type distinction; InternVL2.5 is excellent but heavier.  Because the
fallback only runs on the < ~10% uncertain cases, the 7B model's cost is
amortised away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import structlog
from PIL import Image

from aforge_flow.config import ClipModelChoice, Config
from aforge_flow.processors.base import BaseProcessor, ProcessorResult
from aforge_flow.processors.caption_naming import perceptual_hash
from aforge_flow.taxonomy import (
    ASSET_TYPE_NORMALIZATION,
    ASSET_TYPE_PROMPTS,
    CREATURE_CATEGORY_PROMPTS,
    LabelPrompts,
)
from aforge_flow.utils import is_image, validate_file
from aforge_flow.utils.embedding_cache import EmbeddingCache

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = structlog.get_logger(__name__)


# HF model ids per encoder choice. ``is_siglip`` controls processor padding
# and the matching convention (sigmoid-trained vs softmax-trained towers).
_ENCODER_REGISTRY: Dict[ClipModelChoice, Dict[str, object]] = {
    ClipModelChoice.SIGLIP2: {
        "hf_id": "google/siglip2-so400m-patch14-384",
        "is_siglip": True,
    },
    ClipModelChoice.SIGLIP: {
        "hf_id": "google/siglip-so400m-patch14-384",
        "is_siglip": True,
    },
    ClipModelChoice.CLIP: {
        "hf_id": "openai/clip-vit-large-patch14",
        "is_siglip": False,
    },
    ClipModelChoice.METACLIP2: {
        "hf_id": "facebook/metaclip-2-worldwide-huge-quickgelu",
        "is_siglip": False,
    },
}


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise *x* along *axis* (safe against zero vectors)."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


class ClipEncoder:
    """Thin wrapper over a HuggingFace CLIP-family model.

    Exposes two operations the classifier needs — batched image embedding and
    batched text embedding — returning L2-normalised ``float32`` numpy arrays.
    The model is loaded lazily and lives on the configured device.
    """

    def __init__(self, choice: ClipModelChoice, device: str, cache_dir: Optional[Path] = None):
        """Load the encoder selected by *choice* onto *device*.

        Args:
            choice: Which CLIP-family encoder to use.
            device: ``"cuda"`` or ``"cpu"``.
            cache_dir: Optional HF cache directory override.

        Raises:
            ImportError: If ``transformers``/``torch`` are not installed.
        """
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "transformers and torch are required for creature classification — "
                "install them with: pip install transformers torch"
            ) from exc

        spec = _ENCODER_REGISTRY[choice]
        self.choice = choice
        self.hf_id: str = str(spec["hf_id"])
        self.is_siglip: bool = bool(spec["is_siglip"])
        self.device = device
        self._torch = torch

        logger.info("loading_clip_encoder", model=self.hf_id, device=device)
        kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
        self.model = AutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, **kwargs
        ).to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **kwargs)

        # Probe embedding dim with a 1x1 dummy so the cache can validate it.
        with torch.no_grad():
            dummy = Image.new("RGB", (16, 16))
            feats = self._image_features([dummy])
        self.dim = int(feats.shape[1])
        logger.info("clip_encoder_loaded", model=self.hf_id, dim=self.dim)

    @property
    def model_key(self) -> str:
        """Stable cache namespace: HF id + dim (survives prompt edits)."""
        return f"{self.hf_id}|d{self.dim}"

    def _image_features(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return feats.float().cpu().numpy()

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images.

        Args:
            images: RGB PIL images.

        Returns:
            ``(N, dim)`` L2-normalised ``float32`` array.
        """
        with self._torch.no_grad():
            feats = self._image_features(images)
        return l2_normalize(feats)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text prompts.

        Args:
            texts: Natural-language prompts.

        Returns:
            ``(N, dim)`` L2-normalised ``float32`` array.
        """
        torch = self._torch
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return l2_normalize(feats.float().cpu().numpy())


def build_prototypes(encoder: ClipEncoder, classes: List[LabelPrompts]) -> Tuple[List[str], np.ndarray]:
    """Embed and average each class's prompt ensemble into one prototype.

    Args:
        encoder: Loaded :class:`ClipEncoder`.
        classes: Ordered list of :class:`LabelPrompts`.

    Returns:
        ``(labels, prototypes)`` where ``labels`` are the canonical labels and
        ``prototypes`` is an ``(num_classes, dim)`` L2-normalised array.
    """
    labels: List[str] = []
    vectors: List[np.ndarray] = []
    for lp in classes:
        emb = encoder.encode_texts(lp.prompts)  # (k, dim), already normalised
        proto = l2_normalize(emb.mean(axis=0, keepdims=True))[0]  # average then renormalise
        labels.append(lp.label)
        vectors.append(proto)
    return labels, np.vstack(vectors)


def classify_against(
    image_vec: np.ndarray,
    labels: List[str],
    prototypes: np.ndarray,
    temperature: float,
) -> Tuple[str, float, float, np.ndarray]:
    """Score one image embedding against prototypes.

    Confidence is a temperature-scaled softmax over the candidate classes —
    a *relative* probability that is comparable across images and works for
    both CLIP (softmax-trained) and SigLIP (sigmoid-trained) towers, since we
    only ever rank within a fixed candidate set.

    Args:
        image_vec: ``(dim,)`` L2-normalised image embedding.
        labels: Class labels in prototype order.
        prototypes: ``(num_classes, dim)`` normalised prototypes.
        temperature: Softmax temperature; lower = sharper. ~0.01 mirrors CLIP's
            learned logit scale of ~100.

    Returns:
        ``(label, confidence, margin, probs)`` where ``margin`` is
        ``top1 - top2`` probability and ``probs`` is the full distribution.
    """
    sims = prototypes @ image_vec  # cosine, since both normalised
    logits = sims / max(temperature, 1e-6)
    logits -= logits.max()
    exp = np.exp(logits)
    probs = exp / exp.sum()
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else float(probs[top])
    return labels[top], float(probs[top]), margin, probs


class CreatureClassificationProcessor(BaseProcessor):
    """Zero-shot D&D creature classifier built on a CLIP-family encoder.

    Follows the :class:`BaseProcessor` contract.  Loads the encoder and builds
    text prototypes on init; classifies in :meth:`process` using a cached image
    embedding (computed on demand, or in bulk via :meth:`precompute_embeddings`).

    Recognised (all optional) config attributes — see
    :class:`~aforge_flow.config.Config`:
    ``clip_model``, ``classification_confidence_threshold``,
    ``classification_margin_threshold``, ``classification_temperature``,
    ``classification_batch_size``, ``classification_sidecar``,
    ``vlm_fallback_enabled``, ``vlm_model``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.device = "cuda" if config.use_gpu else "cpu"

        self._choice: ClipModelChoice = getattr(
            config, "clip_model", ClipModelChoice.SIGLIP2
        )
        self._conf_threshold: float = getattr(
            config, "classification_confidence_threshold", 0.85
        )
        self._margin_threshold: float = getattr(
            config, "classification_margin_threshold", 0.10
        )
        self._temperature: float = getattr(config, "classification_temperature", 0.01)
        self._batch_size: int = getattr(config, "classification_batch_size", 32)
        self._write_sidecar: bool = getattr(config, "classification_sidecar", True)
        self._vlm_enabled: bool = getattr(config, "vlm_fallback_enabled", False)
        self._vlm_model: str = getattr(
            config, "vlm_model", "Qwen/Qwen2.5-VL-7B-Instruct"
        )

        self.encoder = ClipEncoder(self._choice, self.device, cache_dir=config.cache_dir)

        # Prototypes are cheap — build them once up front.
        self._type_labels, self._type_protos = build_prototypes(
            self.encoder, ASSET_TYPE_PROMPTS
        )
        self._cat_labels, self._cat_protos = build_prototypes(
            self.encoder, CREATURE_CATEGORY_PROMPTS
        )

        self.cache = EmbeddingCache(
            config.cache_dir / "clip_embeddings.sqlite",
            model_key=self.encoder.model_key,
            dim=self.encoder.dim,
        )

        self._vlm = None  # lazily constructed on first uncertain case
        logger.info(
            "creature_classifier_initialized",
            model=self.encoder.hf_id,
            cached_vectors=self.cache.count(),
        )

    # ----- BaseProcessor contract ----------------------------------------- #

    def can_process(self, input_path: Path) -> bool:
        """Return ``True`` for supported image formats."""
        return is_image(input_path)

    def process(self, input_path: Path) -> ProcessorResult:
        """Classify *input_path* and attach the result as metadata.

        The image is never modified; ``output_path`` is the input path so the
        pipeline chain continues.  If ``classification_sidecar`` is set, the
        structured result is also written to ``<image>.classification.json``.

        Args:
            input_path: Path to the source image.

        Returns:
            A :class:`ProcessorResult` whose ``metadata`` contains
            ``asset_type``, ``category``, ``confidence``, ``model`` and
            ``source`` (``"clip"`` or ``"vlm"``), plus per-level detail.
        """
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            return ProcessorResult(input_path, success=False, error=error_msg)
        try:
            image = Image.open(input_path).convert("RGB")
        except Exception as exc:
            return ProcessorResult(
                input_path, success=False, error=f"Failed to open image: {exc}"
            )

        try:
            image_vec = self._embedding_for(image)
        except Exception as exc:
            logger.exception("embedding_failed", input=str(input_path))
            return ProcessorResult(
                input_path, success=False, error=f"Embedding failed: {exc}"
            )

        result_meta = self._classify_vector(image_vec)

        # Confidence workflow: accept directly, else optionally consult the VLM.
        uncertain = (
            result_meta["confidence"] < self._conf_threshold
            or result_meta["category_margin"] < self._margin_threshold
        )
        if uncertain and self._vlm_enabled:
            try:
                vlm_meta = self._vlm_classify(image, image_vec)
                if vlm_meta is not None:
                    result_meta.update(vlm_meta)
                    result_meta["source"] = "vlm"
            except Exception as exc:  # never fail the asset on a fallback error
                logger.warning("vlm_fallback_failed", input=str(input_path), error=str(exc))

        result_meta["uncertain"] = uncertain
        result_meta["model"] = self._choice.value

        if self._write_sidecar:
            self._write_sidecar_file(input_path, result_meta)

        logger.info(
            "classified",
            input=str(input_path),
            asset_type=result_meta["asset_type"],
            category=result_meta["category"],
            confidence=round(result_meta["confidence"], 3),
            source=result_meta["source"],
        )
        return ProcessorResult(
            input_path,
            output_path=input_path,  # pass-through: pixels unchanged
            success=True,
            metadata=result_meta,
        )

    # ----- Batch throughput ------------------------------------------------ #

    def precompute_embeddings(self, paths: List[Path]) -> int:
        """Embed every not-yet-cached picture in *paths* in GPU batches.

        Call this once before the per-file pass to turn ``process()`` into a
        pure cache lookup.  Replicas (same perceptual hash) are encoded once.

        Args:
            paths: Candidate image paths.

        Returns:
            Number of *new* embeddings computed and cached.
        """
        # Map content key -> first path that has it (dedupes replicas).
        pending: Dict[str, Path] = {}
        for p in paths:
            if not self.can_process(p):
                continue
            try:
                with Image.open(p) as im:
                    key = self._content_key(im.convert("RGB"))
            except Exception:
                continue
            if self.cache.get(key) is None and key not in pending:
                pending[key] = p

        if not pending:
            logger.info("precompute_embeddings_all_cached", total=len(paths))
            return 0

        keys = list(pending.keys())
        computed = 0
        for start in range(0, len(keys), self._batch_size):
            chunk = keys[start : start + self._batch_size]
            images = [Image.open(pending[k]).convert("RGB") for k in chunk]
            vecs = self.encoder.encode_images(images)
            self.cache.put_many({k: vecs[i] for i, k in enumerate(chunk)})
            computed += len(chunk)
            logger.info("precompute_progress", done=computed, total=len(keys))
        return computed

    # ----- internals ------------------------------------------------------- #

    @staticmethod
    def _content_key(image: Image.Image) -> str:
        """Perceptual-hash content key (replicas share an embedding)."""
        return f"{perceptual_hash(image):016x}"

    def _embedding_for(self, image: Image.Image) -> np.ndarray:
        """Return the cached embedding for *image*, computing it on a miss."""
        key = self._content_key(image)
        vec = self.cache.get(key)
        if vec is None:
            vec = self.encoder.encode_images([image])[0]
            self.cache.put(key, vec)
        return l2_normalize(vec)

    def _classify_vector(self, image_vec: np.ndarray) -> dict:
        """Run both classification levels against the cached prototypes."""
        type_label, type_conf, type_margin, _ = classify_against(
            image_vec, self._type_labels, self._type_protos, self._temperature
        )
        cat_label, cat_conf, cat_margin, _ = classify_against(
            image_vec, self._cat_labels, self._cat_protos, self._temperature
        )
        asset_type = ASSET_TYPE_NORMALIZATION.get(type_label, "Portrait")
        return {
            "asset_type": asset_type,
            "asset_type_raw": type_label,
            "asset_type_confidence": round(type_conf, 4),
            "category": cat_label,
            "confidence": round(cat_conf, 4),
            "category_margin": round(cat_margin, 4),
            "asset_type_margin": round(type_margin, 4),
            "source": "clip",
        }

    def _write_sidecar_file(self, input_path: Path, meta: dict) -> None:
        """Write the structured classification next to the image as JSON."""
        sidecar = input_path.with_suffix(input_path.suffix + ".classification.json")
        payload = {
            "asset_type": meta["asset_type"],
            "category": meta["category"],
            "confidence": meta["confidence"],
            "model": meta["model"],
            "source": meta["source"],
        }
        try:
            sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # sidecar is best-effort
            logger.warning("sidecar_write_failed", path=str(sidecar), error=str(exc))

    def _vlm_classify(self, image: Image.Image, image_vec: np.ndarray) -> Optional[dict]:
        """Consult the VLM for a single uncertain image.

        Lazily constructs the VLM on first use.  Returns a metadata patch with
        the refined ``category``/``asset_type`` and a confidence of 1.0 (the VLM
        gives a hard label), or ``None`` if the VLM is unavailable / off.
        """
        if self._vlm is None:
            from aforge_flow.processors.creature_classification import (  # local import
                QwenVLClassifier,
            )

            self._vlm = QwenVLClassifier(
                self._vlm_model, self.device, cache_dir=self.config.cache_dir
            )
        category = self._vlm.classify_category(image, self._cat_labels)
        if category is None:
            return None
        return {
            "category": category,
            "confidence": 1.0,
            "category_margin": 1.0,
        }


class QwenVLClassifier:
    """Optional Qwen2.5-VL fallback for the uncertain tail.

    Asks the model to pick exactly one label from the candidate set and parses
    the reply.  Loaded lazily and only when ``vlm_fallback_enabled`` is set, so
    its VRAM/latency cost is never paid on the confident majority.
    """

    def __init__(self, hf_id: str, device: str, cache_dir: Optional[Path] = None):
        """Load the VLM (4-bit if ``bitsandbytes`` is available on CUDA).

        Args:
            hf_id: HuggingFace model id, e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``.
            device: ``"cuda"`` or ``"cpu"``.
            cache_dir: Optional HF cache override.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        kwargs: dict = {"cache_dir": str(cache_dir)} if cache_dir else {}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            try:  # optional 4-bit to fit a 7B model on a 12 GB 4070
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                logger.info("bitsandbytes_unavailable_loading_fp16")
        logger.info("loading_vlm", model=hf_id, device=device)
        self.processor = AutoProcessor.from_pretrained(hf_id, **({"cache_dir": str(cache_dir)} if cache_dir else {}))
        self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def classify_category(self, image: Image.Image, labels: List[str]) -> Optional[str]:
        """Return the VLM-chosen creature category, or ``None`` if unparseable.

        Args:
            image: RGB image to classify.
            labels: Allowed category labels.

        Returns:
            One of *labels*, matched case-insensitively against the reply.
        """
        options = ", ".join(labels)
        prompt = (
            "You are classifying fantasy RPG artwork by Dungeons & Dragons creature "
            f"type. Choose the single best matching type from this list: {options}. "
            "Answer with only the one word from the list."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        reply = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip().lower()
        for label in labels:
            if label.lower() in reply:
                return label
        return None
