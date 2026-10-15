# aforge-flow Web UI

> **Work in progress.** A refactor is planned, so expect breaking changes. See the [main README](../README.md).

A local browser interface for the pipeline. You can browse or upload assets, set options,
run background removal and media optimization, watch progress live and download the results.
It has a dark and a light theme and supports pt-BR and en-US.

## Run

From the repo root:

```bash
make ui            # npm ci + build + serve the built UI at http://127.0.0.1:8000
```

### Development (hot reload)

Use two terminals:

```bash
make web           # backend on :8000
make web-ui-dev    # Vite on :5173, proxies /api -> :8000
```

Then open http://localhost:5173.

## Security

The API has **no authentication**. It only accepts loopback `Host` and `Origin` headers, and it
serves only supported media files from inside `--root`. The Makefile passes the repo as `--root`.
Point `--root` at a dedicated folder, and do not bind `--host` to a public interface.
