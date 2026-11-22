# aforge-flow-api — FastAPI service

A local web service that wraps the [`aforge-flow`](../aforge_flow/README.md)
pipeline: browse or upload assets, set options, run background removal + media
optimization, watch **live progress over SSE**, and download results. It serves
the built [`aforge_flow_ui`](../aforge_flow_ui/README.md) SPA and the JSON API from one
origin.

> Architecturally the API depends on the pipeline, never the reverse. Run it
> with the `aforge-api` console script (the old `aforge web` command is gone).

## Run

```bash
uv sync                              # installs aforge-flow-api + aforge-flow
uv run aforge-api --root .           # http://127.0.0.1:8000
```

`--root` is the **only** directory the file browser, uploads and downloads can
touch — every path is resolved and confined to it (see the sandbox below).

| Option | Default | Meaning |
|--------|---------|---------|
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | `8000` | Port |
| `--root` | `.` | Sandbox root for browse/upload/download |
| `--log-level` | `info` | Uvicorn log level |

Or with uvicorn directly (root via `AFORGE_ROOT`, default cwd):

```bash
uv run uvicorn aforge_flow_api.main:app --host 127.0.0.1 --port 8000
```

To serve the SPA at `/`, build it first (`cd aforge_flow_ui && npm run build`). The
dist location defaults to `../aforge_flow_ui/dist`; override with
**`AFORGE_FRONTEND_DIST`** when deploying elsewhere. Without a build, only the
JSON API under `/api` is exposed.

## Endpoints

| Method & path | Purpose |
|---------------|---------|
| `GET /api/health` | Liveness + version + active root |
| `GET /api/options` | Enum choices + defaults (drives the form) |
| `GET /api/browse?path=…` | List sub-folders / supported files (sandboxed) |
| `POST /api/upload` | Upload 1..N files into `<root>/.uploads/<uuid>/` |
| `POST /api/jobs` | Start a job (directory or explicit files) |
| `GET /api/jobs/{id}` | Job state, summary, outputs |
| `GET /api/jobs/{id}/events` | **SSE** stream: `start` · `file_done` · `done` / `error` |
| `GET /api/file?path=…&thumb=` | Serve an image/video (optional thumbnail) for preview |
| `GET /api/download?path=…` | Download a single output |
| `GET /api/jobs/{id}/download.zip` | Zip of all of a job's outputs |

### Quick curl

```bash
curl -s localhost:8000/api/health
curl -s "localhost:8000/api/options" | jq '.image_formats, .video_formats'
curl -s "localhost:8000/api/browse?path=assets" | jq '.entries[].name'
# escaping the sandbox is refused:
curl -s -o /dev/null -w '%{http_code}\n' "localhost:8000/api/browse?path=/etc"   # 403
```

### Path sandbox

`core/security.py:safe_resolve()` resolves every user path (relative or
absolute, symlinks included) and returns **403** unless it stays under `--root`.
Uploads land in `<root>/.uploads/<uuid>/` so they're inside the sandbox and
usable as job targets.

## Layout

```
aforge_flow_api/
├── pyproject.toml            # script aforge-api; depends on aforge-flow (uv workspace)
├── src/aforge_flow_api/
│   ├── main.py               # create_app(root) + module-level `app`
│   ├── cli.py                # the `aforge-api` launcher
│   ├── core/                 # settings.py (root, aforge_flow_ui dist) · security.py (sandbox)
│   ├── schemas/jobs.py       # JobRequest / JobCreated / UploadResponse
│   ├── services/             # job_manager.py (threaded jobs + SSE log) · config_builder.py
│   └── routers/              # options · browse · jobs · files · upload · download
└── tests/test_api_flow.py    # TestClient flow (CPU, no model downloads)
```

Jobs run in a background thread; the `JobManager` keeps an ordered event log so
an SSE client that connects (or reconnects) mid-run replays every event with no
duplicates.

## Tests

```bash
uv run --package aforge-flow-api pytest aforge_flow_api/tests
```

Covers `/api/options`, the sandbox (403), and the full
upload → job → SSE → zip + thumbnail flow in media-optimization-only mode (no
GPU, no model downloads, all writes to a temp dir).
