# Contributing

Thanks for your interest in aforge-flow. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/brunohaf/aforge-flow.git
cd aforge-flow
make deps-install      # uv sync --frozen --all-extras
make doctor
```

You need [uv](https://docs.astral.sh/uv/). GNU make and [direnv](https://direnv.net) are optional.

## Workflow

1. Branch from `main`: `feat/<topic>`, `fix/<topic>`, `refactor/<topic>`, `docs/<topic>`.
2. Keep changes focused. If you spot an unrelated bug, open an issue for it instead of fixing it in the same PR.
3. Add or update tests. `make tests` must pass on CPU without downloading models.
   Fake heavy models at the inference boundary (see `tests/conftest.py`).
   Put real-model tests behind `@pytest.mark.gpu`.
4. Run `make check` (lint, format check, tests with 85% coverage) before opening the PR.
   Resolve every review thread; the branch ruleset blocks merging otherwise.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) messages:
   `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.

## Releases

Add one label to the PR: `bump:major`, `bump:minor` or `bump:patch`. When it is merged, the
`release` workflow tags `vX.Y.Z`, moves `vX` and `vX.Y`, and publishes a GitHub release. The
package version is read from the tag, so never edit a version number by hand.

## Dependencies

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. After you
change them, run `uv lock` and commit both files. CI fails if `uv lock --check` fails.

## Adding a profile

Profiles are plain data in `src/aforge_flow/profiles.py`. Add an entry to
`PROFILES` whose `defaults` keys are `process` parameter names, then add a test
in `tests/test_workflows.py`.
