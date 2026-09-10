"""Filesystem tools demo.

The `ls` and `cat` tools are bound to a directory so the LLM can locate
files in your project's data folder. By default that's `./data` next to
the cwd, but you can override it with `HERDING_CATS_DATA_DIR`.

Run:
    PYTHONPATH=src python examples/filesystem_demo.py

This works whether Ollama is local or in the docker-compose stack — the
filesystem tools only read from disk, the LLM is just there to decide
which file to read.
"""

from __future__ import annotations

import sys

from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.tools_builtin import builtin_tools
from herding_cats.tools_filesystem import default_data_dir, filesystem_tools


def main() -> None:
    root = default_data_dir()
    print(f"Filesystem tools are bound to: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"Created {root} (it was empty).")

    ollama = OllamaClient()
    if not ollama.health():
        print(
            "Ollama not reachable. Start it with `docker compose up -d`.",
            file=sys.stderr,
        )
        sys.exit(1)

    crew = Crew(
        ollama=ollama,
        tool_specs=builtin_tools() + filesystem_tools(root=root),
    )
    runner = CrewRunner(crew)

    for q in [
        "What files are in the data directory?",
        "Show me the contents of any .md file you find.",
        "What's the largest file under data/?",
    ]:
        print(f"\nQ: {q}")
        try:
            print(f"A: {runner.run(q)}")
        except Exception as e:
            print(f"(error: {e})")


if __name__ == "__main__":
    main()
