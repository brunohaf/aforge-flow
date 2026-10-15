"""Shell-agnostic helpers for the Makefile (stdlib only).

Usage: python scripts/make_helpers.py <help|mkdir|rm|doctor|ruleset> [args...]
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_TARGET = re.compile(r"^([a-zA-Z0-9_.-]+):[^=]*?##\s*(.+)$")
_SECTION = re.compile(r"^##@\s*(.+)$")
_VENDORED = {".venv", "node_modules", ".git"}


def cmd_help(makefile: str = "Makefile") -> int:
    """Print ``target  description`` for every ``target: ## description`` line."""
    print("Usage: make <target> [VAR=value ...]")
    for line in Path(makefile).read_text(encoding="utf-8").splitlines():
        if m := _SECTION.match(line):
            print(f"\n{m.group(1)}")
        elif m := _TARGET.match(line):
            print(f"  {m.group(1):<20} {m.group(2)}")
    print("\nVariables: IN=assets OUT=output IMAGE_FORMAT=webp COMPRESSION=balanced")
    print("           CATEGORY=general DEVICE=--gpu ARGS='<extra aforge flags>'")
    return 0


def cmd_mkdir(*dirs: str) -> int:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"ready: {d}/")
    return 0


def _expand(pattern: str) -> list[Path]:
    if not any(c in pattern for c in "*?["):
        return [Path(pattern)]
    return sorted(p for p in Path(".").glob(pattern) if not _VENDORED.intersection(p.parts[:-1]))


def cmd_rm(*patterns: str) -> int:
    """Remove files/dirs inside the repo; globs skip .venv, node_modules and .git."""
    root = Path.cwd().resolve()
    for pattern in patterns:
        for p in _expand(pattern):
            target = p.resolve()
            if root not in target.parents:
                print(f"refusing to remove {p}: outside {root}", file=sys.stderr)
                return 1
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                else:
                    continue
            except PermissionError as exc:
                print(f"cannot remove {p}: {exc} (is a process still using it?)", file=sys.stderr)
                return 1
            print(f"removed: {p}")
    return 0


def _torch_status(venv_python: Path) -> tuple[bool, str]:
    probe = "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
    try:
        out = subprocess.run(  # nosec B603: fixed probe against the local venv interpreter
            [str(venv_python.resolve()), "-c", probe], check=False, capture_output=True, text=True
        )
    except OSError as exc:
        return False, str(exc)
    if out.returncode == 0:
        return True, out.stdout.strip()
    lines = (out.stderr or out.stdout).strip().splitlines()
    return False, lines[-1] if lines else f"exit code {out.returncode}"


def cmd_doctor() -> int:
    """Report required/optional tools; non-zero exit only if a required one is missing."""
    checks = [
        ("uv", True, "https://docs.astral.sh/uv/getting-started/installation/"),
        ("ffmpeg", False, "needed for video only: https://ffmpeg.org/download.html"),
        ("nvidia-smi", False, "NVIDIA driver; without it everything runs on CPU"),
        ("direnv", False, "optional auto-activation: https://direnv.net"),
        ("npm", False, "only to build the web UI frontend"),
    ]
    missing_required = False
    for tool, required, hint in checks:
        path = shutil.which(tool)
        status = "ok" if path else ("MISSING" if required else "absent")
        print(f"  [{status:^7}] {tool:<11} {path or hint}")
        missing_required |= required and not path

    venv_python = Path(
        ".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python"
    )
    if venv_python.exists():
        ok, detail = _torch_status(venv_python)
        print(f"  [{'ok' if ok else 'FAIL':^7}] {'torch':<11} {detail}")
    else:
        print(f"  [{'absent':^7}] {'.venv':<11} run: make deps-install")
    return 1 if missing_required else 0


def cmd_ruleset(path: str = ".github/rulesets/main.json") -> int:
    """Create or update the repository ruleset defined in *path* (needs ``gh`` auth)."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    gh = shutil.which("gh")
    if not gh:
        print("gh CLI not found: https://cli.github.com", file=sys.stderr)
        return 1
    listing = subprocess.run(  # nosec B603: gh resolved via shutil.which, fixed arguments
        [gh, "api", "repos/{owner}/{repo}/rulesets"], check=False, capture_output=True, text=True
    )
    if listing.returncode != 0:
        print(listing.stderr.strip(), file=sys.stderr)
        return listing.returncode
    existing = next(
        (r["id"] for r in json.loads(listing.stdout) if r["name"] == spec["name"]), None
    )
    method, endpoint = (
        ("PUT", f"repos/{{owner}}/{{repo}}/rulesets/{existing}")
        if existing
        else ("POST", "repos/{owner}/{repo}/rulesets")
    )
    applied = subprocess.run(  # nosec B603: gh resolved via shutil.which, repo-controlled path
        [gh, "api", "-X", method, endpoint, "--input", path], check=False
    )
    if applied.returncode == 0:
        print(f"ruleset '{spec['name']}' {'updated' if existing else 'created'}")
    return applied.returncode


COMMANDS: dict[str, Callable[..., int]] = {
    "help": cmd_help,
    "mkdir": cmd_mkdir,
    "rm": cmd_rm,
    "doctor": cmd_doctor,
    "ruleset": cmd_ruleset,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](*sys.argv[2:]))
