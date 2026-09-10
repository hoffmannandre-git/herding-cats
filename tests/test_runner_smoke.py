"""Slow smoke test: end-to-end crew run against a real Ollama.

Skipped automatically if Ollama is not reachable.

Run explicitly with:
    pytest -m slow
"""

from __future__ import annotations

import pytest

from herding_cats import Crew, CrewRunner, OllamaClient

pytestmark = [pytest.mark.slow]


def test_smalltalk_returns_short_answer(skip_if_no_ollama) -> None:
    crew = Crew(ollama=OllamaClient())
    runner = CrewRunner(crew)
    answer = runner.run("hi")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_crew_with_tool_finds_and_calls_it(skip_if_no_ollama) -> None:
    called: list[dict] = []

    def echo(i: dict) -> dict:
        called.append(i)
        return {"echoed": i}

    crew = Crew(
        ollama=OllamaClient(),
        tools={"echo": echo},
    )
    runner = CrewRunner(crew)
    runner.run("Please call the echo tool with city=Berlin.")
    assert called, "echo tool was never called"


def test_stream_emits_final_event(skip_if_no_ollama) -> None:
    crew = Crew(ollama=OllamaClient())
    runner = CrewRunner(crew)
    events = list(runner.run_stream("hello there"))
    final = [e for e in events if e.kind == "final"]
    assert final, "no final event emitted"
    assert "answer" in final[0].payload
