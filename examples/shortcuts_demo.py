"""Shortcuts example: fast-path known query shapes.

When a regex pattern matches, the runner calls the handler directly and
bypasses the crew entirely — no LLM call, no orchestrator, no fetcher.

Run:
    PYTHONPATH=src python examples/shortcuts_demo.py
"""

from __future__ import annotations

import sys

from herding_cats import Crew, CrewRunner, OllamaClient

# A fake "contacts" store — would normally be a database query.
CONTACTS = [
    {"name": "Alice", "role": "engineer"},
    {"name": "Bob", "role": "designer"},
    {"name": "Carol", "role": "PM"},
]


def list_contacts(_input: dict) -> str:
    """Shortcut handler: list all contacts in a fixed shape."""
    return "\n".join(f"- {c['name']} ({c['role']})" for c in CONTACTS)


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        print("Ollama not reachable. Start it with `docker compose up -d`.", file=sys.stderr)
        sys.exit(1)

    crew = Crew(
        ollama=ollama,
        shortcuts={
            r"^(list|show)\s+(all\s+)?contacts\??$": list_contacts,
        },
    )
    runner = CrewRunner(crew)

    # This will be answered without a single LLM call.
    answer = runner.run("list all contacts")
    print("Shortcut answer:")
    print(answer)

    # This goes through the regular crew.
    answer = runner.run("What is the meaning of life?")
    print("\nRegular crew answer:")
    print(answer)


if __name__ == "__main__":
    main()
