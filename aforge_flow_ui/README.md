# frontend — AForge Flow web UI

A React + Vite single-page app for the [`aforge-flow-api`](../aforge_flow_api/README.md)
service: browse/upload assets, set options visually, run the pipeline, watch
live progress with before/after previews, and download results. Includes a
**dark/light** theme and **pt-BR / en-US** i18n.

## Develop (hot reload)

Run the API and the Vite dev server side by side — Vite proxies `/api` → the
backend, so they're same-origin:

```bash
# terminal 1 — backend (repo root)
uv run aforge-api --root .            # :8000

# terminal 2 — frontend (this folder)
npm install
npm run dev                           # http://localhost:5173
```

## Build (served by the API)

```bash
npm install
npm run build                         # → aforge_flow_ui/dist/
uv run aforge-api --root .            # serves dist/ at / (and the API at /api)
```

The API looks for the build at `../aforge_flow_ui/dist` by default; override with the
`AFORGE_FRONTEND_DIST` env var when deploying elsewhere.

## Structure

```
frontend/
├── index.html                 # Tailwind via CDN (darkMode: 'class'); pre-paint theme
├── vite.config.ts             # dev proxy /api → 127.0.0.1:8000
├── tsconfig.json
└── src/
    ├── main.tsx  App.tsx
    ├── lib/api.ts             # fetch / EventSource / upload wrappers + types
    ├── i18n/                  # index.tsx (provider + useI18n) · locales.ts (pt-BR / en-US)
    ├── hooks/useTheme.ts      # dark/light toggle (localStorage + prefers-color-scheme)
    ├── components/            # Tab · ThemeToggle · LanguageSelect (shared)
    └── features/
        ├── source/            # FileBrowser · Uploader
        ├── options/           # OptionsForm (rendered from /api/options)
        └── progress/          # ProgressPanel (SSE, previews, downloads)
```

## Notes

- **Tailwind** is loaded via CDN in `index.html` (no PostCSS build step); the
  theme class is applied pre-paint to avoid a flash.
- **State** is plain React hooks + `fetch`/`EventSource` — no extra state lib.
- Add a UI string in **both** locales in `src/i18n/locales.ts`.
- Type-check + bundle with `npm run build` (`tsc && vite build`).
