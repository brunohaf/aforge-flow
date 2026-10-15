"""D&D creature taxonomy and CLIP prompt engineering.

This module is the single source of truth for *what* the
:class:`~aforge_flow.processors.creature_classification.CreatureClassificationProcessor`
classifies against.  It is deliberately model-agnostic: it only produces
label sets and natural-language prompts.  The encoder/​matching logic lives
in the processor.

Why prompt engineering matters for CLIP-family models
-----------------------------------------------------
CLIP / SigLIP map an image and a piece of text into a shared embedding
space and score them by cosine similarity.  The text tower was trained on
billions of *web captions*, not on a controlled vocabulary.  Two
consequences follow:

1. **A bare label is out-of-distribution.**  The token sequence ``"Fiend"``
   almost never appears as a standalone web caption, so its embedding is
   noisy and sits far from how artwork is actually described online.  A
   sentence such as *"a demonic infernal fantasy creature from the lower
   planes, with horns and fire"* lands squarely in the part of the space
   the model learned from real captions, so the image→text similarity is
   far more reliable.

2. **Ambiguous words collide.**  ``"Construct"`` overlaps with grammar /
   programming senses; ``"Beast"`` with the fairy-tale character; ``"Fey"``
   with surnames.  Embedding a *description* of the visual concept ("a
   natural animal such as a wolf, bear, or giant spider") disambiguates the
   sense we want.

The remedy used here is the standard CLIP recipe of **prompt ensembling**:
for every class we keep several rich, complementary templates, embed all of
them, L2-normalise, and average into one *class prototype* vector.  The
averaged prototype is more robust than any single phrasing and costs
nothing at inference time (text embeddings are computed once and cached).

The dataclasses below carry both the machine label (returned in the
``ProcessorResult`` metadata) and the human prompt list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class LabelPrompts:
    """A single class: its canonical label and the prompts that describe it.

    Attributes:
        label: Canonical machine-readable label, e.g. ``"Undead"``.  This is
            what ends up in the structured output — never the raw prompt.
        prompts: One or more semantically rich natural-language descriptions.
            These are embedded and averaged into a class prototype.  Order is
            irrelevant; aim for 3-6 complementary phrasings per class.
    """

    label: str
    prompts: list[str] = field(default_factory=list)


# Level 1 — Asset type
#
# We do not ask CLIP for "Portrait" / "Token" directly.  Those words are
# overloaded (a "token" is also a coin / a software token).  Instead we
# describe the *visual format* with several phrasings each, then map the
# winning fine-grained format back to the two normalised buckets below.

ASSET_TYPE_NORMALIZATION: dict[str, str] = {
    "battle_token": "Token",
    "portrait": "Portrait",
    "character_illustration": "Portrait",
    "creature_illustration": "Portrait",
}
"""Maps the fine-grained format label CLIP selects to the two normalised
asset types required by the output schema (``"Portrait"`` / ``"Token"``)."""


ASSET_TYPE_PROMPTS: list[LabelPrompts] = [
    LabelPrompts(
        "battle_token",
        [
            "a top-down circular battle map token for a tabletop role-playing game",
            "a round VTT token seen from directly above, with a circular border ring",
            "a top-down overhead view of a miniature on a circular base",
            "a virtual tabletop creature token, centered, viewed from the top",
        ],
    ),
    LabelPrompts(
        "portrait",
        [
            "a fantasy character portrait showing the head and shoulders",
            "a close-up bust portrait of a fantasy character facing the viewer",
            "a framed headshot illustration of a single fantasy character",
        ],
    ),
    LabelPrompts(
        "character_illustration",
        [
            "a full-body fantasy character illustration in a dynamic pose",
            "a detailed digital painting of a standing fantasy adventurer",
            "concept art of a fantasy character shown from head to toe",
        ],
    ),
    LabelPrompts(
        "creature_illustration",
        [
            "a full-body illustration of a fantasy monster or creature",
            "a detailed digital painting of a fantasy beast in a natural pose",
            "concept art of a large fantasy creature shown in full",
        ],
    ),
]


# Level 2 — D&D creature category (the 14 official creature types)
#
# Each prompt describes the *visual archetype* of the type as it appears in
# fantasy art, not the rules definition.  Several phrasings per class are
# ensembled into one prototype vector.

CREATURE_CATEGORY_PROMPTS: list[LabelPrompts] = [
    LabelPrompts(
        "Aberration",
        [
            "an alien aberration with tentacles, many eyes, and an unnatural otherworldly anatomy",
            "a grotesque eldritch creature with a mass of tentacles and asymmetric mouths",
            "a mind flayer style aberration, a beholder, or a writhing cosmic horror",
        ],
    ),
    LabelPrompts(
        "Beast",
        [
            "a natural animal such as a wolf, bear, giant spider, or eagle in a fantasy setting",
            "an ordinary mundane animal or wild beast, biologically plausible, not magical",
            "a realistic creature like a big cat, boar, snake, or bird of prey",
        ],
    ),
    LabelPrompts(
        "Celestial",
        [
            "a radiant celestial being such as an angel with feathered wings and a glowing halo",
            "a divine holy creature wreathed in golden light from the upper planes",
            "a benevolent heavenly messenger glowing with sacred light",
        ],
    ),
    LabelPrompts(
        "Construct",
        [
            "an artificial construct such as an animated armor, golem, or clockwork automaton",
            "a non-living creature built from stone, metal, clay, or gears, magically animated",
            "a mechanical robot-like fantasy automaton with no organic flesh",
        ],
    ),
    LabelPrompts(
        "Dragon",
        [
            "a large draconic fantasy creature with scales, leathery wings, and reptilian features",
            "a dragon or wyvern with a long neck, horns, claws, and a scaled serpentine body",
            "a winged fire-breathing reptilian dragon perched or in flight",
        ],
    ),
    LabelPrompts(
        "Elemental",
        [
            "an elemental creature made of living fire, water, earth, or swirling air",
            "a being composed entirely of one element, such as a fire elemental or earth elemental",
            "a humanoid-shaped mass of flame, stone, water, or wind with no solid flesh",
        ],
    ),
    LabelPrompts(
        "Fey",
        [
            "a whimsical fey creature such as a fairy, pixie, dryad, or satyr with insect or plant features",
            "a delicate magical sprite from the feywild with butterfly wings and a nature theme",
            "an enchanting woodland fey being tied to nature and the seasons",
        ],
    ),
    LabelPrompts(
        "Fiend",
        [
            "a demonic or infernal fantasy creature from the lower planes, with horns, fangs, and fire",
            "a devil or demon with red skin, bat wings, horns, and a menacing hellish aura",
            "an evil fiend wreathed in brimstone and flame from the abyss or the nine hells",
        ],
    ),
    LabelPrompts(
        "Giant",
        [
            "a giant humanoid of enormous size such as an ogre, troll, hill giant, or frost giant",
            "a towering brutish giant dwarfing normal people, wielding a huge club or boulder",
            "a massive humanoid colossus, crude and oversized",
        ],
    ),
    LabelPrompts(
        "Humanoid",
        [
            "a humanoid person of a fantasy ancestry such as a human, elf, dwarf, orc, or goblin",
            "a person-sized fantasy character: an adventurer, soldier, mage, rogue, or villager",
            "a bipedal humanoid with two arms and two legs wearing armor or clothing",
        ],
    ),
    LabelPrompts(
        "Monstrosity",
        [
            "a monstrous hybrid creature such as a manticore, griffon, chimera, owlbear, or hydra",
            "an unnatural monster combining features of several animals into one fearsome beast",
            "a fantastical predatory monster that is neither a normal animal nor a dragon",
        ],
    ),
    LabelPrompts(
        "Ooze",
        [
            "an amorphous gelatinous ooze, slime, or jelly with no fixed shape",
            "a blob of translucent acidic slime oozing across the ground",
            "a shapeless creature made of viscous goo, such as a gelatinous cube or black pudding",
        ],
    ),
    LabelPrompts(
        "Plant",
        [
            "a plant creature such as an animated tree, walking shrub, or carnivorous vine monster",
            "a living plant monster covered in bark, leaves, vines, fungi, or flowers",
            "a treant, shambling mound, or vegetal creature rooted in nature",
        ],
    ),
    LabelPrompts(
        "Undead",
        [
            "a fantasy undead creature animated by necromancy, such as a skeleton, zombie, or lich",
            "a rotting corpse, ghostly spirit, vampire, or skeletal warrior risen from death",
            "a decaying or spectral undead being with exposed bone, pale flesh, or a ghostly form",
        ],
    ),
]


def asset_type_labels() -> list[str]:
    """Return the fine-grained asset-type labels in prototype order."""
    return [lp.label for lp in ASSET_TYPE_PROMPTS]


def creature_category_labels() -> list[str]:
    """Return the 14 canonical creature-category labels in prototype order."""
    return [lp.label for lp in CREATURE_CATEGORY_PROMPTS]


# Creature name bank
#
# A curated vocabulary of canonical creature *names*, grouped by the same 14
# creature-category labels above.  Unlike the prompts (which describe a class
# for CLIP), these are concrete monster-manual names used as a soft hint for
# the Qwen caption backend, so generated filenames prefer real creatures
# (``mind-flayer``) over invented phrasings.  The data lives in an editable
# JSON file next to this module so non-developers can extend it.

CREATURE_NAMES_PATH = Path(__file__).with_name("creature_names.json")
"""Location of the editable name-bank JSON, shipped alongside this module."""


@lru_cache(maxsize=1)
def load_creature_names() -> dict[str, list[str]]:
    """Load the creature name bank, keyed by canonical creature-category label.

    The JSON may carry a leading ``_comment`` key (and any other underscore-
    prefixed metadata), which is ignored.  Every remaining key is validated
    against :func:`creature_category_labels` so the bank can't silently drift
    from the taxonomy.

    Returns:
        Mapping of creature-category label -> list of slug-formatted names.
        Result is cached; call :func:`load_creature_names.cache_clear` to
        force a reload after editing the file.

    Raises:
        ValueError: If the file contains a key that is not one of the 14
            canonical creature-category labels.
    """
    raw = json.loads(CREATURE_NAMES_PATH.read_text(encoding="utf-8"))
    names = {k: v for k, v in raw.items() if not k.startswith("_")}

    valid = set(creature_category_labels())
    unknown = set(names) - valid
    if unknown:
        raise ValueError(
            f"creature_names.json has unknown creature types {sorted(unknown)}; "
            f"keys must be one of {sorted(valid)}"
        )
    return names


def creature_name_prompt_block(names: dict[str, list[str]] | None = None) -> str:
    """Render the name bank as a compact, type-grouped block for an LLM prompt.

    Produces one line per creature type, e.g.::

        Aberration: mind-flayer, beholder, gibbering-mouther, ...
        Beast: dire-wolf, giant-spider, ...

    Args:
        names: Name bank to render; defaults to :func:`load_creature_names`.

    Returns:
        A newline-joined string suitable for embedding in a caption prompt.
    """
    names = names or load_creature_names()
    return "\n".join(f"{label}: {', '.join(vals)}" for label, vals in names.items())
