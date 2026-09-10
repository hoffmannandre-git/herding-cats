# Contributing

Thanks for taking a look. Small, focused PRs are welcome.

## How to run the project

The supported path is Docker Compose. From a clone:

```bash
cp .env.example .env
docker compose --profile api up -d --build
curl -fsS http://localhost:18000/health
```

Run the fast checks the same way CI does:

```bash
docker compose --profile test run --rm test
```

That runs `ruff` and `pytest -m "not slow"` inside the test container.
Slow tests that need a live Ollama model are opt-in:

```bash
# set HERDING_CATS_PIPELINE_MODE=all in .env, then:
docker compose --profile test run --rm test
```

## Working on the code without Docker

Useful if you want a fast edit/test loop on the host. You still need
Ollama somewhere (Compose’s `ollama` service on port `11434` is fine).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"

ruff check .
pytest -q -m "not slow"
python scripts/check_prompts_isolated.py
```

Set `HERDING_CATS_OLLAMA_URL=http://localhost:11434` when you run
anything that talks to the model.

## Project conventions

- **Prompts stay in Markdown.** Role text lives under
  `src/herding_cats/prompts/` (and `prompts_de/` for German). Don’t paste
  multi-line “You are the Orchestrator…” strings into Python. The check
  script above enforces that.
- **Docs match Docker.** User-facing README examples should use Compose.
  Host `pip` / venv steps belong in this file.
- **Env vars** use the `HERDING_CATS_*` prefix. Older `LOCALCREW_*` names
  still work as a fallback for one release; prefer the new prefix in new
  code.
- **Grounding language.** Call the harness “grounding tests.” Don’t market
  the project as hallucination-proof or hallucination-tolerant.

## Pull requests

1. One concern per PR when you can.
2. Keep the fast suite green (`not slow`).
3. If you change public behavior or the HTTP API, update the README.
4. Don’t commit secrets, `.env`, or local scratch files.

CI (`.github/workflows/ci.yml`) runs on every push and pull request:
`ruff check`, the prompt-isolation script, and `pytest -m "not slow"`.
Jobs that need a live model are not required in CI.

## Questions

Open an issue if something in the README or this file is wrong or unclear.
