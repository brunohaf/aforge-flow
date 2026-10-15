"""Named presets that bundle option defaults for a use case.

A profile only changes *defaults*: it is applied through Click's ``default_map``,
so any flag the user passes explicitly on the command line still wins.

Example::

    aforge process --profile tabletop                 # classify + caption-rename on
    aforge process --profile tabletop --no-classify   # explicit flag overrides it
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import click


@dataclass(frozen=True)
class Profile:
    """A named bundle of ``process`` option defaults.

    Attributes:
        name: Value accepted by ``--profile``.
        summary: One-line description shown in ``--help``.
        defaults: Click parameter name -> default value. Keys must match the
            Python parameter names of the command (e.g. ``classification``,
            not ``--classify``).
    """

    name: str
    summary: str
    defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


PROFILES: Mapping[str, Profile] = MappingProxyType(
    {
        "default": Profile(
            name="default",
            summary="Generic assets: background removal + media optimization.",
        ),
        "tabletop": Profile(
            name="tabletop",
            summary="Tabletop RPG / VTT art: adds D&D creature classification and "
            "creature-aware file naming.",
            defaults=MappingProxyType(
                {
                    "classification": True,
                    "caption_rename": True,
                }
            ),
        ),
    }
)


def apply_profile(ctx: click.Context, param: click.Parameter, value: str | None) -> str:
    """Click callback for ``--profile``: merge the profile into ``ctx.default_map``.

    Must be attached to an ``is_eager=True`` option so it runs before the
    options it provides defaults for are resolved.
    """
    name = value or "default"
    profile = PROFILES[name]
    ctx.default_map = {**(ctx.default_map or {}), **profile.defaults}
    return name


def profile_help() -> str:
    """Help text for ``--profile`` listing every profile and its summary."""
    lines = "; ".join(f"'{p.name}': {p.summary}" for p in PROFILES.values())
    return f"Preset of option defaults (explicit flags still win). {lines}"
