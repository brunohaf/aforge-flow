# AForge Flow — Web UI

A local browser interface for the `aforge-flow` pipeline: browse/upload assets,
set options visually, run background removal + media optimization, watch live
progress and download results. Dark/light theme and pt-BR / en-US.

## Run

**Backend** (FastAPI) — from the repo root:

```bash
pip install -e ".[web]"
aforge web --root .            # http://127.0.0.1:8000
```

`--root` is the only directory the file browser / uploads / downloads can touch.

### Production (single server)

```bash
cd web && npm install && npm run build   # outputs web/dist/
aforge web --root .                        # serves the built SPA at /
```

### Development (hot reload)

Two terminals:

```bash
aforge web --root .          # backend on :8000
cd web && npm run dev        # Vite on :5173, proxies /api -> :8000
```

Open http://localhost:5173.
