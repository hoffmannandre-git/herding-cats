# Changelog

## 0.3.0 — initial public release

First release of **herding-cats** (formerly developed as `localcrew`).

- Four-role crew runtime (orchestrator → thinker → fetcher → finalist) on local Ollama
- Docker Compose as the supported run path (`api` and `test` profiles)
- HTTP API (`/health`, `/crew/run`, `/crew/step`, `/crew/stream`, sessions)
- Built-in tools + filesystem `ls` / `cat` bound to `./data`
- Prompt packs in English and German; prompts stay as Markdown
- Fast test suite + optional slow grounding tests against Ollama
- CI: ruff + `pytest -m "not slow"` on push/PR
