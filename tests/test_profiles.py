"""Tests for ``--profile`` default resolution (explicit flags must win)."""

import click
import pytest
from click.testing import CliRunner

from aforge_flow.profiles import PROFILES, apply_profile


@click.command()
@click.option(
    "--profile",
    type=click.Choice(list(PROFILES)),
    default="default",
    is_eager=True,
    callback=apply_profile,
)
@click.option("--classify/--no-classify", "classification", default=False)
@click.option("--caption-rename/--no-caption-rename", "caption_rename", default=False)
def _probe(profile: str, classification: bool, caption_rename: bool) -> None:
    click.echo(f"{profile} {classification} {caption_rename}")


def _run(*args: str) -> str:
    result = CliRunner().invoke(_probe, list(args))
    assert result.exit_code == 0, result.output
    return result.output.strip()


def test_default_profile_keeps_command_defaults() -> None:
    assert _run() == "default False False"


def test_tabletop_profile_enables_classification_and_naming() -> None:
    assert _run("--profile", "tabletop") == "tabletop True True"


def test_explicit_flag_overrides_profile_regardless_of_order() -> None:
    assert _run("--no-classify", "--profile", "tabletop") == "tabletop False True"
    assert _run("--profile", "tabletop", "--no-caption-rename") == "tabletop True False"


def test_unknown_profile_is_rejected() -> None:
    result = CliRunner().invoke(_probe, ["--profile", "nope"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


@pytest.mark.parametrize("name", list(PROFILES))
def test_profile_defaults_are_immutable(name: str) -> None:
    with pytest.raises(TypeError):
        PROFILES[name].defaults["classification"] = False  # type: ignore[index]
