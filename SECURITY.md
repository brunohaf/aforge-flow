# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through
[GitHub Security Advisories](https://github.com/brunohaf/aforge-flow/security/advisories/new).
You can expect an acknowledgement within 7 days.

## Web UI threat model

`aforge web` is a local tool with **no authentication**. Its defenses:

- It binds to `127.0.0.1` by default.
- Requests are rejected unless the `Host` header is a loopback name or the configured `--host`.
  This blocks DNS rebinding.
- Requests that carry an `Origin` other than `http://localhost|127.0.0.1|[::1]:<port>` are
  rejected. This blocks cross-site requests, including multipart uploads, which skip CORS preflight.
- Every path is resolved, symlinks included, and must stay inside `--root`.
- `/api/file` and `/api/download` serve only supported image and video formats, so files such as
  `.env`, keys or `.git/` content are never returned.
- Uploads are limited to supported formats, at most 500 files per request and 2 GiB per file.

Known limits: any local process can still reach the port, and finished jobs stay in memory until
the server stops. Point `--root` at a folder that contains nothing sensitive. **Never** bind
`--host` to a public interface on an untrusted network.

## Other surfaces

- Model weights are downloaded from Hugging Face and the rembg model hosts on first use.
- Untrusted input files are decoded by Pillow and FFmpeg (FFmpeg is called with an argument list,
  never through a shell).

## Supported versions

While the project is in alpha, only the latest commit on `main` receives fixes.
