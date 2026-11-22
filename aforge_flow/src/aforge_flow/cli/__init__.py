"""The ``aforge`` command-line interface (Click).

The command group is assembled in :mod:`aforge_flow.cli.main`; each subcommand
lives in its own module under :mod:`aforge_flow.cli.commands`. The console-script
entry point remains ``aforge_flow.cli:main``.
"""

from aforge_flow.cli.main import main

__all__ = ["main"]
