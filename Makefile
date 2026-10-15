# Recipes are plain commands so they run under sh and cmd.exe alike; anything
# shell-specific lives in scripts/make_helpers.py (stdlib only).

.DEFAULT_GOAL := help
.SUFFIXES:
MAKEFLAGS += --no-builtin-rules --warn-undefined-variables

UV           ?= uv
OS           ?=
ifeq ($(OS),Windows_NT)
PYTHON       ?= python
else
PYTHON       ?= python3
endif
IN           ?= assets
OUT          ?= output
IMAGE_FORMAT ?= webp
VIDEO_FORMAT ?= webm
COMPRESSION  ?= balanced
CATEGORY     ?= general
DEVICE       ?= --gpu
HOST         ?= 127.0.0.1
PORT         ?= 8000
EXTRAS       ?= --all-extras
ARGS         ?=

# --no-sync: runs never add/remove packages; deps-install owns the environment.
RUN    := $(UV) run --no-sync
AFORGE := $(RUN) aforge
# --isolated: the helper must not run from .venv, or distclean would delete its own interpreter.
HELPER := $(UV) run --isolated --no-project --quiet python scripts/make_helpers.py

.PHONY: help doctor deps-install deps-install-core deps-update direnv protect-main dirs \
        optimize optimize-lossless remove-bg process tabletop rename classify \
        recommend info benchmark ui web web-ui-install web-ui-build web-ui-dev \
        tests tests-gpu coverage lint format format-check typecheck check \
        clean clean-output distclean

##@ Setup
help: ## Show this help
	@$(HELPER) help $(firstword $(MAKEFILE_LIST))

doctor: ## Check uv, ffmpeg, GPU driver, direnv and the torch build in .venv
	@$(PYTHON) scripts/make_helpers.py doctor

deps-install: ## Create/refresh .venv from uv.lock with all extras
	$(UV) sync --frozen $(EXTRAS)

deps-install-core: ## Install only core + dev deps (no classify/web/video)
	$(UV) sync --frozen --extra dev

deps-update: ## Re-resolve uv.lock to the newest allowed versions
	$(UV) lock --upgrade

direnv: ## Trust .envrc so direnv auto-activates .venv on cd
	direnv allow .

protect-main: ## Apply .github/rulesets/main.json (resolved threads + green checks); repo must be public or Pro
	@$(PYTHON) scripts/make_helpers.py ruleset

dirs: ## Create the IN and OUT folders (default: assets/ and output/)
	@$(HELPER) mkdir $(IN) $(OUT)

##@ Pipeline   (vars: IN OUT IMAGE_FORMAT COMPRESSION CATEGORY DEVICE ARGS)
optimize: dirs ## Transcode images/video in IN to IMAGE_FORMAT/VIDEO_FORMAT
	$(AFORGE) optimize -i $(IN) -o $(OUT) --image-format $(IMAGE_FORMAT) --video-format $(VIDEO_FORMAT) --compression $(COMPRESSION) $(DEVICE) $(ARGS)

optimize-lossless: dirs ## Pixel-exact PNG recompression (keeps transparency)
	$(AFORGE) optimize -i $(IN) -o $(OUT) --image-format png --compression lossless $(ARGS)

remove-bg: dirs ## Strip backgrounds; model picked by CATEGORY
	$(AFORGE) remove-background -i $(IN) -o $(OUT) --category $(CATEGORY) $(DEVICE) $(ARGS)

process: dirs ## Background removal, then optimization to IMAGE_FORMAT
	$(AFORGE) process -i $(IN) -o $(OUT) --category $(CATEGORY) --image-format $(IMAGE_FORMAT) --compression $(COMPRESSION) $(DEVICE) $(ARGS)

tabletop: dirs ## process + D&D classification + creature-aware naming
	$(AFORGE) process --profile tabletop -i $(IN) -o $(OUT) --category $(CATEGORY) --image-format $(IMAGE_FORMAT) $(DEVICE) $(ARGS)

rename: dirs ## Give files meaningful names (captions only junk names)
	$(AFORGE) caption-rename -i $(IN) -o $(OUT) $(DEVICE) $(ARGS)

classify: dirs ## Tag images with the D&D creature taxonomy (JSON sidecars)
	$(AFORGE) classify -i $(IN) -o $(OUT) $(DEVICE) $(ARGS)

recommend: dirs ## Suggest --workers/--gpu for this machine and IN
	$(AFORGE) recommend -i $(IN) -o $(OUT) $(ARGS)

info: ## Show version, PyTorch and CUDA details
	$(AFORGE) info

benchmark: ## Classifier throughput benchmark on IN (classify extra)
	$(RUN) python benchmarks/benchmark_classifier.py throughput -i $(IN) $(ARGS)

##@ Web UI   (work in progress; a refactor is planned)
ui: web-ui-install web-ui-build web ## Install, build and serve the UI

web: ## Serve the API + built UI on HOST:PORT, confined to the repo
	$(AFORGE) web --host $(HOST) --port $(PORT) --root . $(ARGS)

web-ui-install: ## Install frontend deps from package-lock.json
	npm --prefix web ci

web-ui-build: ## Build the frontend into web/dist
	npm --prefix web run build

web-ui-dev: ## Vite dev server with hot reload (run make web alongside)
	npm --prefix web run dev

##@ Quality
tests: ## CPU test suite (models are faked, nothing downloads)
	$(RUN) pytest $(ARGS)

tests-gpu: export RUN_GPU_TESTS := 1
tests-gpu: ## Only the real-model GPU tests (downloads weights)
	$(RUN) pytest -m gpu $(ARGS)

coverage: ## Tests with coverage; fails under 85% (pyproject [tool.coverage])
	$(RUN) pytest --cov $(ARGS)

lint: ## Ruff lint
	$(RUN) ruff check src tests scripts benchmarks

format: ## Format with black and apply ruff fixes
	$(RUN) black src tests scripts benchmarks
	$(RUN) ruff check --fix src tests scripts benchmarks

format-check: ## Fail if formatting is off
	$(RUN) black --check src tests scripts benchmarks

typecheck: ## mypy over the package (not yet clean; not part of check)
	$(RUN) mypy src/aforge_flow

check: lint format-check coverage ## lint + format-check + tests with the 85% gate (CI)

##@ Cleanup
clean: ## Remove caches and build artefacts
	@$(HELPER) rm .pytest_cache .mypy_cache .ruff_cache .coverage build dist "src/*.egg-info" "**/__pycache__"

clean-output: ## Delete OUT and the cache dir (forces a full re-run)
	@$(HELPER) rm $(OUT) .cache

distclean: clean clean-output ## Also remove .venv, web/node_modules and web/dist
	@$(HELPER) rm .venv web/node_modules web/dist
