"""weather_user.py — a multi-tool demo.

Two tools: one synchronous (returns data immediately), one async
(simulates an HTTP fetch). The crew treats them the same way.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from herding_cats import Crew, CrewRunner, OllamaClient


def get_weather(input: dict) -> dict:
    """Return canned weather for a city. Sync tool."""
    city = input.get("city", "Unknown")
    return {
        "city": city,
        "temp_c": 17,
        "summary": "overcast",
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


async def lookup_user(input: dict) -> dict:
    """Pretend to look up a user. Async tool."""
    await asyncio.sleep(0.05)
    user_id = input.get("id", "anon")
    return {"id": user_id, "name": "Demo User", "role": "admin"}


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        raise SystemExit(f"Ollama not reachable at {ollama.base_url}")

    crew = Crew(
        ollama=ollama,
        tools={
            "get_weather": get_weather,
            "lookup_user": lookup_user,
        },
    )
    runner = CrewRunner(crew)

    questions = [
        "What's the weather in Berlin right now?",
        "Who am I, and what's the weather in Tokyo?",
    ]
    for q in questions:
        print(f"\n>>> {q}")
        answer = runner.run(q)
        print(answer)


if __name__ == "__main__":
    main()
