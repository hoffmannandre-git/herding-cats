"""``python -m herding_cats`` entrypoint + console scripts.

Three subcommands are exposed:

* ``api``     — start the FastAPI HTTP server (default if no subcommand).
* ``run``     — one-shot question via the crew.
* ``version`` — print the installed version and exit.

The HTTP server (``api``) is the default because the docker image's
``CMD`` is just ``herding-cats-api``. Running ``python -m herding_cats``
without a subcommand also starts the server, which is what container
operators expect.

Environment variables:

* ``HERDING_CATS_OLLAMA_URL``  Ollama endpoint (default ``http://localhost:11434``).
* ``HERDING_CATS_MODEL``       Model name (default ``llama3.1:8b``).
* ``HERDING_CATS_HOST``        Bind host for the API (default ``0.0.0.0``).
* ``HERDING_CATS_PORT``        Bind port for the API (default ``18000``).
* ``HERDING_CATS_DATA_DIR``    Filesystem tools root (default ``./data``).
* ``HERDING_CATS_MAX_LOOP``    Override the crew loop cap.
* ``HERDING_CATS_MAX_FETCH``   Override the fetch-turn cap.
* ``HERDING_CATS_MAX_THINK``   Override the think-turn cap.
* ``HERDING_CATS_ORCHESTRATOR_MODE``  ``rules`` (default) or ``llm``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from herding_cats import __version__
from herding_cats.helpers.env import get_env


def _crew_from_env(tools: str = "builtin+filesystem") -> Any:
    """Build a default `Crew` from environment variables.

    ``tools`` selects which tool specs to wire up:

    * ``builtin``       — only the 5 general tools from ``tools_builtin``.
    * ``filesystem``    — only ``ls`` / ``cat`` from ``tools_filesystem``.
    * ``builtin+filesystem`` (default) — both.
    """
    # Imports are deferred so the module loads even if FastAPI is not
    # installed (only `api` needs uvicorn).
    from herding_cats import Crew, OllamaClient
    from herding_cats.tools_builtin import builtin_tools
    from herding_cats.tools_filesystem import (
        default_data_dir,
        filesystem_tools,
    )

    tool_specs: list[Any] = []
    if "builtin" in tools:
        tool_specs.extend(builtin_tools())
    if "filesystem" in tools:
        tool_specs.extend(filesystem_tools(root=default_data_dir()))

    return Crew(
        ollama=OllamaClient(
            base_url=get_env("OLLAMA_URL"),
        ),
        tool_specs=tool_specs,
    )


def cmd_api(_args: argparse.Namespace) -> int:
    """Start the FastAPI HTTP server."""
    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError:
        print(
            "FastAPI / uvicorn not installed. Install with:\n"
            "  pip install herding-cats[server]",
            file=sys.stderr,
        )
        return 1

    # Probe Ollama once before binding so we fail fast with a clear
    # message instead of starting the server and then 502-ing forever.
    from herding_cats.server import create_app  # type: ignore[import-not-found]

    crew = _crew_from_env()
    app = create_app(crew=crew)
    # Env lookup order: HERDING_CATS_* -> LOCALCREW_* (legacy) -> default.
    host = get_env("HOST") or "0.0.0.0"
    port = int(get_env("PORT") or "18000")
    print(
        f"herding_cats {__version__} API listening on http://{host}:{port}",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """One-shot: build a crew and answer a single question."""
    from herding_cats import CrewRunner

    crew = _crew_from_env(tools=args.tools)
    runner = CrewRunner(crew)
    answer = runner.run(args.question)
    if args.json:
        print(json.dumps({"answer": answer, "model": crew.model}))
    else:
        print(answer)
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"herding_cats {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="herding_cats",
        description=(
            "Multi-agent LLM crew runtime backed by a local Ollama instance."
        ),
    )
    sub = p.add_subparsers(dest="command")

    p_api = sub.add_parser("api", help="start the FastAPI HTTP server (default)")
    p_api.set_defaults(func=cmd_api)

    p_run = sub.add_parser("run", help="answer a single question and exit")
    p_run.add_argument("question", help="the question to ask the crew")
    p_run.add_argument(
        "--tools",
        choices=("builtin", "filesystem", "builtin+filesystem"),
        default=get_env("TOOLS") or "builtin+filesystem",
        help="which tool registry to wire up (default: builtin+filesystem)",
    )
    p_run.add_argument(
        "--json",
        action="store_true",
        help="emit the answer as JSON {answer, model}",
    )
    p_run.set_defaults(func=cmd_run)

    p_version = sub.add_parser("version", help="print the installed version")
    p_version.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # Default to `api` for the docker container ergonomics.
        args = parser.parse_args(["api"])
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
