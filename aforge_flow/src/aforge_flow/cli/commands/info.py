"""``aforge info`` — show version, PyTorch, and CUDA/GPU details."""

import click

from aforge_flow import __version__


@click.command()
def info() -> None:
    """Display system and configuration information."""
    import torch

    click.echo("\n" + "=" * 60)
    click.echo("AFORGE FLOW (aforge) - SYSTEM INFORMATION")
    click.echo("=" * 60)
    click.echo(f"Version:          {__version__}")
    click.echo(f"PyTorch:          {torch.__version__}")
    click.echo(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA device:      {torch.cuda.get_device_name(0)}")
        click.echo(f"GPU memory:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    click.echo("=" * 60 + "\n")
