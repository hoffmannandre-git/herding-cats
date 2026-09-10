"""stream_demo.py — observe the crew's loop event-by-event.

Useful for UI integration: print one line per event so you can see
orchestrator decisions, tool calls, and the final answer as they happen.
"""

from __future__ import annotations

from herding_cats import Crew, CrewRunner, OllamaClient


def get_weather(input: dict) -> dict:
    return {"city": input.get("city", "?"), "temp_c": 17, "summary": "overcast"}


def main() -> None:
    ollama = OllamaClient()
    if not ollama.health():
        raise SystemExit(f"Ollama not reachable at {ollama.base_url}")

    crew = Crew(
        ollama=ollama,
        tools={"get_weather": get_weather},
    )
    runner = CrewRunner(crew)

    for event in runner.run_stream("What's the weather in Paris?"):
        kind = event.kind
        if kind == "orchestrator_decision":
            print(f"  [orchestrator] → {event.payload['next']}  ({event.payload['reason']})")
        elif kind == "think":
            terms = ", ".join(event.payload.get("search_terms") or [])
            print(f"  [thinker]     notes={event.payload['notes'][:80]!r}  terms=[{terms}]")
        elif kind == "tool_call":
            print(f"  [executor]    call {event.payload['tool']}({event.payload['input']})")
        elif kind == "tool_result":
            err = event.payload.get("error")
            if err:
                print(f"  [executor]    ! error: {err}")
            else:
                print(f"  [executor]    → {event.payload['output']}")
        elif kind == "final":
            print("\n" + event.payload["answer"])
        elif kind == "warning":
            print(f"  [warning]     {event.payload.get('message')}")


if __name__ == "__main__":
    main()
