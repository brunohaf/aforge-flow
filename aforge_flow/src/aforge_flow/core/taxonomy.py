"""Pluggable taxonomy profiles and CLIP prompt engineering.

This module is the single source of truth for *what* the
:class:`~aforge_flow.processors.creature_classification.CreatureClassificationProcessor`
classifies against, and for the soft name hint the
:class:`~aforge_flow.processors.caption_naming.CaptionRenameProcessor` feeds to
its captioning backend. It is deliberately model-agnostic: it only produces
label sets and natural-language prompts. The encoder/​matching logic lives in
the processors.

Why prompt engineering matters for CLIP-family models
-----------------------------------------------------
CLIP / SigLIP map an image and a piece of text into a shared embedding space
and score them by cosine similarity. The text tower was trained on billions of
*web captions*, not on a controlled vocabulary. So a bare label like ``"Fiend"``
is out-of-distribution and noisy, while a rich description ("a demonic infernal
fantasy creature from the lower planes, with horns and fire") lands where the
model learned from real captions. The standard remedy used here is **prompt
ensembling**: several phrasings per class are embedded, L2-normalised and
averaged into one robust *class prototype*.

Pluggable domains (profiles)
----------------------------
The taxonomy is **data-driven**: each *profile* is a JSON file under
``core/profiles/`` carrying the category labels + prompts, the asset-type
formats + normalisation, an optional captioning instruction/prefix, and an
optional name bank. The default profile is ``dnd`` (Dungeons & Dragons), which
reproduces the original hard-coded taxonomy byte-for-byte. Ship others such as
``mundane`` (everyday photos) and ``call_of_cthulhu`` (Lovecraftian horror), or
point :func:`load_profile` at any external ``<name>.json``.

The module-level constants/functions (:data:`CREATURE_CATEGORY_PROMPTS`,
:data:`ASSET_TYPE_PROMPTS`, :data:`ASSET_TYPE_NORMALIZATION`,
:func:`creature_category_labels`, :func:`asset_type_labels`,
:func:`load_creature_names`, :func:`creature_name_prompt_block`) are kept for
backward compatibility and resolve to the **default ``dnd`` profile**.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

PROFILES_DIR = Path(__file__).with_name("profiles")
"""Directory holding the built-in profile JSON files (and their name banks)."""

DEFAULT_PROFILE = "dnd"
"""Profile used when none is selected — preserves the historical D&D behaviour."""


@dataclass(frozen=True)
class LabelPrompts:
    """A single class: its canonical label and the prompts that describe it.

    Attributes:
        label: Canonical machine-readable label, e.g. ``"Undead"``. This is what
            ends up in the structured output — never the raw prompt.
        prompts: One or more semantically rich natural-language descriptions.
            These are embedded and averaged into a class prototype. Order is
            irrelevant; aim for 3-6 complementary phrasings per class.
    """

    label: str
    prompts: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaxonomyProfile:
    """A complete, named classification/naming domain loaded from JSON.

    Attributes:
        name: Profile identifier (e.g. ``"dnd"``).
        categories: The level-2 classes (creature/subject types) to classify into.
        asset_types: The level-1 visual-format classes (e.g. portrait vs token).
        asset_type_normalization: Maps a fine-grained asset-type label to the
            normalised bucket required by the output schema.
        name_bank: Optional canonical-name vocabulary keyed by category label,
            used as a soft hint for the caption backend.
        caption_instruction: Optional instruction for the Qwen caption backend.
        caption_prefix: Optional conditional prefix for the BLIP caption backend.
    """

    name: str
    categories: List[LabelPrompts]
    asset_types: List[LabelPrompts] = field(default_factory=list)
    asset_type_normalization: Dict[str, str] = field(default_factory=dict)
    name_bank: Dict[str, List[str]] = field(default_factory=dict)
    caption_instruction: str = ""
    caption_prefix: str = ""

    def category_labels(self) -> List[str]:
        """Return the category labels in prototype order."""
        return [lp.label for lp in self.categories]

    def asset_type_labels(self) -> List[str]:
        """Return the fine-grained asset-type labels in prototype order."""
        return [lp.label for lp in self.asset_types]

    def name_prompt_block(self) -> str:
        """Render the name bank as a compact, type-grouped block for an LLM prompt.

        Produces one line per category, e.g.::

            Aberration: mind-flayer, beholder, gibbering-mouther, ...
            Beast: dire-wolf, giant-spider, ...
        """
        return "\n".join(
            f"{label}: {', '.join(vals)}" for label, vals in self.name_bank.items()
        )


def available_profiles() -> List[str]:
    """List the built-in profile names shipped under ``core/profiles/``."""
    return sorted(
        p.stem
        for p in PROFILES_DIR.glob("*.json")
        if not p.name.endswith(".names.json")
    )


@lru_cache(maxsize=None)
def load_profile(name_or_path: str = DEFAULT_PROFILE) -> TaxonomyProfile:
    """Load a taxonomy profile by built-in name or external JSON path.

    Resolution order:
      1. A path ending in ``.json`` that exists on disk (a custom profile).
      2. A built-in name resolved to ``core/profiles/<name>.json``.

    A sibling ``<name>.names.json`` (or an embedded ``name_bank`` key) supplies
    the optional name bank. Name-bank keys are validated against the profile's
    own category labels so they can't silently drift.

    Args:
        name_or_path: Built-in profile name (e.g. ``"dnd"``) or a path to a
            custom profile JSON.

    Returns:
        The parsed :class:`TaxonomyProfile`. Cached; call
        ``load_profile.cache_clear()`` to force a reload after editing a file.

    Raises:
        ValueError: If the name is unknown, or the name bank references a
            category the profile does not define.
    """
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" and candidate.exists():
        path = candidate
        names_path = candidate.with_name(f"{candidate.stem}.names.json")
    else:
        path = PROFILES_DIR / f"{name_or_path}.json"
        names_path = PROFILES_DIR / f"{name_or_path}.names.json"
        if not path.exists():
            raise ValueError(
                f"Unknown taxonomy profile {name_or_path!r}; built-ins are "
                f"{available_profiles()}, or pass a path to a .json profile."
            )

    data = json.loads(path.read_text(encoding="utf-8"))

    categories = [LabelPrompts(c["label"], list(c["prompts"])) for c in data["categories"]]
    asset_types = [
        LabelPrompts(a["label"], list(a["prompts"])) for a in data.get("asset_types", [])
    ]

    name_bank: Dict[str, List[str]] = {}
    if names_path.exists():
        raw = json.loads(names_path.read_text(encoding="utf-8"))
        name_bank = {k: v for k, v in raw.items() if not k.startswith("_")}
    elif "name_bank" in data:
        name_bank = {k: v for k, v in data["name_bank"].items() if not k.startswith("_")}

    valid = {c.label for c in categories}
    unknown = set(name_bank) - valid
    if unknown:
        raise ValueError(
            f"profile {name_or_path!r} name bank has unknown categories "
            f"{sorted(unknown)}; keys must be among {sorted(valid)}"
        )

    return TaxonomyProfile(
        name=str(data.get("name", Path(name_or_path).stem)),
        categories=categories,
        asset_types=asset_types,
        asset_type_normalization=dict(data.get("asset_type_normalization", {})),
        name_bank=name_bank,
        caption_instruction=str(data.get("caption_instruction", "")),
        caption_prefix=str(data.get("caption_prefix", "")),
    )


# --------------------------------------------------------------------------- #
# Backward-compatible surface — resolves to the default ``dnd`` profile so any
# existing caller (and the test-suite) keeps working unchanged.
# --------------------------------------------------------------------------- #

_DEFAULT = load_profile(DEFAULT_PROFILE)

ASSET_TYPE_PROMPTS: List[LabelPrompts] = _DEFAULT.asset_types
CREATURE_CATEGORY_PROMPTS: List[LabelPrompts] = _DEFAULT.categories
ASSET_TYPE_NORMALIZATION: Dict[str, str] = _DEFAULT.asset_type_normalization


def asset_type_labels() -> List[str]:
    """Return the default profile's fine-grained asset-type labels."""
    return _DEFAULT.asset_type_labels()


def creature_category_labels() -> List[str]:
    """Return the default profile's category labels (the 14 D&D creature types)."""
    return _DEFAULT.category_labels()


def load_creature_names() -> Dict[str, List[str]]:
    """Return the default (D&D) profile's creature name bank."""
    return _DEFAULT.name_bank


def creature_name_prompt_block(names: Dict[str, List[str]] | None = None) -> str:
    """Render a name-bank block; defaults to the D&D profile's name bank."""
    if names is None:
        return _DEFAULT.name_prompt_block()
    return "\n".join(f"{label}: {', '.join(vals)}" for label, vals in names.items())
