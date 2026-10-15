# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `--profile` option for `process`, with a `tabletop` profile that turns on D&D classification and creature-aware naming.
- Makefile with setup, pipeline, web UI, quality and cleanup targets (`make help`).
- uv project setup with `uv.lock`, CUDA 12.6 PyTorch wheels on Windows/Linux, and a direnv `.envrc`.
- CI with an 85% coverage gate, security (bandit, pip-audit, gitleaks, dependency review), CodeQL and reviewdog workflows, Dependabot, and issue/PR templates.
- Label-driven releases (`bump:*` labels, action-bumpr). The version is derived from git tags via setuptools-scm.
- Hugging Face downloads pinned to reviewed commits (`HF_REVISIONS`).
- Branch ruleset as code (`.github/rulesets/main.json`, `make protect-main`).
- End-to-end tests for every CLI workflow and for the web API.
- Web UI hardening: a Host allowlist, an Origin check, media-only file serving and upload limits.

### Changed
- **Python 3.11+ is required** (rembg >= 2.0.76 needs it). Migration: install Python 3.11+, or let uv install it (`make deps-install`).
- **Conda support removed** (`environment.yml`). Migration: use `make deps-install` / `uv sync`. `pip install -e ".[classify,web]"` also still works.
- **transformers 5.x** (`>=5.10,<6`) for the `classify` extra. It clears the PYSEC-2026-2288/2289/3929 advisories. The model code now passes `dtype=` instead of `torch_dtype=`. Run `make tests-gpu` to validate real models on your hardware.
- An explicit `--model` now overrides the `--category` routing. Before, it was silently ignored.

### Fixed
- `process` now runs the optimization step after background removal. Before, the intermediate file's path broke output mirroring.
- Worker exceptions are recorded as failures instead of being silently dropped.
- The `video` extra can be installed again: the `ffmpeg-python` lower bound pointed at a version that was never released.

### Removed
- Stale package copies from earlier renames (`asset_foundry`, `vtt_asset_forge`, `aforge_foundry`), machine-specific shell scripts and `docs/reference.md`.
