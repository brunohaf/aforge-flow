"""Tests for source-name preservation in the caption-rename processor."""

import pytest

from aforge_flow.processors.caption_naming import (
    _looks_like_id,
    simplify_caption,
    source_name_slug,
)


@pytest.mark.parametrize(
    "stem, expected",
    [
        # Meaningful hand-authored names survive (pipeline suffixes stripped).
        ("Balhannoth_nobg_optimized", "balhannoth"),
        ("Aberrant Spirit_nobg_optimized", "aberrant-spirit"),
        ("Neh-thalggu_nobg_optimized", "neh-thalggu"),
        ("Giant Space Hamster_nobg_optimized", "giant-space-hamster"),
        # Filler words ("in", "a") are dropped.
        ("Brain in a Jar_nobg_optimized", "brain-jar"),
        # Descriptive text is recovered even behind a numeric ID prefix.
        (
            "900297781783671446_seheran. mutated human flesh_nobg_optimized",
            "seheran-mutated-human-flesh",
        ),
    ],
)
def test_meaningful_names_preserved(stem, expected):
    assert source_name_slug(stem) == expected


@pytest.mark.parametrize(
    "stem",
    [
        "900297781783770851_nobg_optimized",  # pure numeric ID
        "af49b6a5f78957e448ce5b9d29a29369_nobg_optimized",  # hex hash
        "nobg_optimized",  # only pipeline noise
        "",
    ],
)
def test_junk_names_fall_through(stem):
    """Non-descriptive names return None so the caller captions instead."""
    assert source_name_slug(stem) is None


def test_max_words_caps_slug():
    stem = "one two three four five six seven_nobg"
    assert source_name_slug(stem, max_words=3) == "one-two-three"


def test_looks_like_id():
    assert _looks_like_id("900297781783671446")
    assert _looks_like_id("af49b6a5f78957e448ce5b9d29a29369")
    assert not _looks_like_id("balhannoth")
    assert not _looks_like_id("12345")  # too short to be an ID


def test_source_slug_does_not_apply_creature_synonyms():
    """Unlike caption slugs, source names are taken at face value."""
    # 'rat' would become 'wererat' via the caption synonym map...
    assert "wererat" in simplify_caption("a giant rat")
    # ...but a hand-named source file keeps the literal word.
    assert source_name_slug("rat_nobg_optimized") == "rat"
