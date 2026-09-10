"""hello_crew.py — minimal smoke test.

Run with:
    OLLAMA_BASE_URL=http://localhost:11434 python examples/hello_crew.py

Requires a running Ollama daemon and at least one chat model pulled.
Defaults to `llama3.1:8b`; override with HERDING_CATS_MODEL.
"""

from __future__ import annotations

from herding_cats import Crew, CrewRunner, OllamaClient


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        raise SystemExit(
            f"Ollama not reachable at {ollama.base_url}. "
            "Start it with `ollama serve` and pull a model."
        )

    def echo(input: dict) -> dict:
        """Echo the input back — useful as a smoke-test tool."""
        return {"echo": input}

    crew = Crew(
        ollama=ollama,
        tools={"echo": echo},
    )
    runner = CrewRunner(crew)

    for question in [
        "Hello!",
        "What does the echo tool say if I pass city=Boston?",
        "Thanks, bye.",
    ]:
        print(f"\n>>> {question}")
        answer = runner.run(question)
        print(answer)


if __name__ == "__main__":
    main()
