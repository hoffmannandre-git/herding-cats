"""Per-step steering example: drive the crew one orchestrator decision at a time.

This is what an HTTP `/crew/step` endpoint or a UI panel would call.

Run:
    PYTHONPATH=src python examples/step_demo.py
"""

from __future__ import annotations

import json
import sys

from herding_cats import (
    Crew,
    CrewRunner,
    CrewState,
    OllamaClient,
    RunnerEvent,
    StepResult,
    pipeline_status,
)


def print_event(ev: RunnerEvent | None) -> None:
    if ev is None:
        return
    print(f"  [{ev.kind}] {json.dumps(ev.payload, ensure_ascii=False)[:120]}")


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        print("Ollama not reachable. Start it with `docker compose up -d`.", file=sys.stderr)
        sys.exit(1)

    crew = Crew(
        ollama=ollama,
        tools={
            "echo": lambda input: {"echoed": input},
        },
    )
    runner = CrewRunner(crew)

    question = "What's the weather in Berlin?"
    state = CrewState(question=question)

    print(f"Question: {question}\n")
    step_no = 0
    while True:
        step_no += 1
        result: StepResult = runner.step(state)
        state = result.state

        print(f"step {step_no}: orchestrator → {result.decision.next} ({result.decision.reason})")
        print_event(result.event)
        print(f"  pipeline: {pipeline_status(state, result.decision)}")
        print("  trace:")
        for line in state.trace[-3:]:
            print(f"    - {line}")
        print()

        if result.done:
            print(f"\nFinal answer:\n  {state.current.answer}")
            break

        if step_no > 10:
            print("safety brake hit; aborting")
            break


if __name__ == "__main__":
    main()
