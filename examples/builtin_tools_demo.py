"""Built-in tool registry demo.

The crew is configured with `herding_cats.tools_builtin.builtin_tools()`,
which ships five ready-to-use tools with pydantic schemas:

* get_current_time — wall-clock in a timezone
* date_now        — today's date and weekday
* calculator      — sandboxed arithmetic
* echo            — repeat text N times
* web_search      — STUB; returns empty + a note

Run:
    PYTHONPATH=src python examples/builtin_tools_demo.py
"""

from __future__ import annotations

import sys

from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.tools_builtin import builtin_tools


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        print("Ollama not reachable. Start it with `docker compose up -d`.", file=sys.stderr)
        sys.exit(1)

    crew = Crew(
        ollama=ollama,
        tool_specs=builtin_tools(),
    )
    runner = CrewRunner(crew)

    for q in [
        "What time is it in Berlin?",
        "What is 17 * 23?",
        "What's today's date?",
    ]:
        print(f"\nQ: {q}")
        print(f"A: {runner.run(q)}")


if __name__ == "__main__":
    main()
