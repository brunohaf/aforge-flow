# aforge-flow Roadmap

Monorepo restructure and pluggable taxonomy. Status was read from the working tree, not from test runs. Items marked **verify** have not been checked.

## Goal

One professional `uv` monorepo with three projects and a domain-pluggable taxonomy, with zero regression against the current D&D behavior.

| Project | Path | Role | Entry point |
|---|---|---|---|
| CLI + pipeline | `aforge_flow/` | Processing core and `aforge` CLI | `aforge` |
| API | `aforge_flow_api/` | FastAPI backend (extracted from `web/`) | `aforge-api` |
| UI | `aforge_flow_ui/` | React/Vite frontend | `npm run dev` |

Data and runtime folders (`assets/`, `output/`, `.cache/`, `scripts/`) stay at the root and are not touched by code moves.

## Phases

### 1. Split into three projects (done, uncommitted)
- Move config, pipeline and taxonomy into `aforge_flow/core/`.
- Split the 864-line `cli.py` into `cli/main.py`, `cli/_shared.py` and `cli/commands/*`.
- Extract the API from `aforge_flow/web/` into `aforge_flow_api/` (`core`, `schemas`, `services`, `routers`).
- Move the frontend from `web/` to `aforge_flow_ui/`.
- Drop `aforge web` from the CLI. The API ships its own `aforge-api` script.

Exit: no references to `aforge_flow.web` remain, and the old `src/`, `web/` and `tests/` at the root are gone.

### 2. uv workspace
- Root `pyproject.toml` is a virtual workspace with shared ruff, black, mypy and pytest config.
- Members are `aforge_flow` and `aforge_flow_api`. The API depends on `aforge-flow` through `[tool.uv.sources]`.
- CLI-only install: `uv sync --package aforge-flow`. Full install: `uv sync`.
- `core/profiles/*.json` is declared as package-data.
- Optional PyTorch `[[tool.uv.index]]` is documented, not pinned.

Exit: both installs resolve, and the CLI-only environment has no fastapi.

### 3. Pluggable taxonomy
- `TaxonomyProfile` and `load_profile(name | path)` live in `core/taxonomy.py`, with `lru_cache`.
- D&D data moves verbatim into `dnd.json` and `dnd.names.json`. The public constants and functions keep working from the default `dnd` profile.
- `Config.taxonomy_profile` defaults to `"dnd"`. `--taxonomy` is available on `classify`, `caption-rename` and `process`.
- Ship `mundane` and `call_of_cthulhu` as examples.

Exit: `load_profile("dnd")` equals the old constants, and the existing tests pass unchanged.

### 4. `rename` improvements
- **Prompt:** replace the generic captioning bias with specific fantasy names. The prompt comes from the active profile (`caption_instruction`, `caption_prefix`), with a better default as fallback.
- **Videos:** `can_process` accepts images or videos. A video is copied as-is to `output_dir / relative_path` with `shutil.copy2`, respects `skip_existing`, and is not captioned.

Exit: `aforge rename` on a temp folder with one image and one `.webm` renames the image and copies the `.webm` byte-identical. **Verify** that the default caption no longer yields names like "armored-fighter".

### 5. Docs
- Root `README.md`: pipeline overview, CLI/API/UI diagram, uv setup, the two run modes (CLI-only and web UI), why D&D names exist and how to switch taxonomy.
- One README per project: `aforge_flow` (commands, flags, profiles), `aforge_flow_api` (endpoints, SSE, `--root` sandbox, `AFORGE_FRONTEND_DIST`), `aforge_flow_ui` (dev proxy, prod build, i18n, theme).

### 6. Release
- Rename the repo folder from `vtt-asset-forge` to `aforge-flow`.
- Commit, then run the full verification below.

## Verification checklist

1. `uv sync` and `uv sync --package aforge-flow` both succeed.
2. Imports work for `aforge_flow.core.*`, `aforge_flow.cli` and `aforge_flow_api`.
3. `uv run pytest aforge_flow/tests -m "not gpu"` passes (baseline: 30 tests).
4. `uv run --package aforge-flow-api pytest aforge_flow_api/tests` passes (options, browse `/etc` returns 403, upload to job to SSE to zip, thumbnail).
5. `uv run aforge-api --root .` answers `/api/health`, `/api/options` and `/api/browse?path=assets`.
6. `npm run build` in `aforge_flow_ui/` produces `dist/`. With `AFORGE_FRONTEND_DIST` set, `GET /` serves the SPA and `/api/*` stays reachable.
7. `uv run aforge --help` has no `web` command. `aforge process --taxonomy mundane --help` works.
8. Orphan-reference grep is clean, except the compat re-exports.

Pipeline and API checks write to temp directories, never to `output/`.

## Risks

- The first `uv sync` is heavy (torch, onnxruntime). Fallback: editable install in the current interpreter.
- Profile JSON missing from the built package. Test `load_profile` from an installed wheel, not only from source.
- Compat re-exports in `aforge_flow/__init__.py` and `core/__init__.py` are the only guard for the public API. Do not remove them without a migration note.

## Differences from the original plan

- The frontend folder is `aforge_flow_ui/`, not `frontend/`.
- Update the `frontend/` paths in the verification steps and READMEs to match.
