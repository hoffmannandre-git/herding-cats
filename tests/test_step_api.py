"""Tests for the per-step steering API (Gap 1).

`CrewRunner.step(state)` lets a caller advance the loop one orchestrator
decision at a time. This is the basis of HTTP `/crew/step` endpoints,
UI buttons, and external steering.

These tests don't hit Ollama — they exercise the runner with stub Ollama
clients and tools, and inspect the `StepResult` return value.
"""

from __future__ import annotations

import pytest

from herding_cats import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Crew,
    CrewRunner,
    CrewState,
    OllamaClient,
    StepResult,
)


class _StubOllama(OllamaClient):
    """An OllamaClient that records requests and returns canned responses.

    We override `__init__` so no real HTTP client is created.
    """

    def __init__(self, responses: list[str], *, default: str | None = None) -> None:  # type: ignore[no-untyped-def]
        # Skip the parent __init__ — we don't need a real httpx client.
        self.base_url = "http://stub"
        self.timeout_s = 0.0
        self.calls: list[ChatRequest] = []
        self._responses = list(responses)
        self._default = default

    async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
        self.calls.append(request)
        text = self._responses.pop(0) if self._responses else (self._default or "")
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=text),
            done=True,
        )

    def health(self) -> bool:  # type: ignore[override]
        return True


@pytest.fixture
def stub_crew() -> Crew:
    return Crew(ollama=_StubOllama([]), tools={})  # type: ignore[arg-type]


def test_step_returns_decision_and_event(stub_crew: Crew) -> None:
    runner = CrewRunner(stub_crew)
    state = CrewState(question="What is the weather in Berlin?")

    result = runner.step(state)

    assert isinstance(result, StepResult)
    assert result.decision.next in {"think", "fetch", "final", "stop"}
    # First round on a real question → think.
    assert result.decision.next == "think"
    assert result.event is not None
    assert result.event.kind in {"think", "warning"}
    assert result.done is False
    # State was mutated (round_number incremented, think_used tracked).
    new_state = result.state
    assert new_state.round_number == 1
    assert new_state.think_used == 1
    assert new_state.last_role == "think"


def test_step_state_is_round_trippable_through_json(stub_crew: Crew) -> None:
    """A core requirement of step(): state can be JSON-serialized and resumed."""
    runner = CrewRunner(stub_crew)
    state = CrewState(question="hi")

    first = runner.step(state)
    serialized = first.state.model_dump_json()

    # Resume from a freshly deserialized state.
    rehydrated = CrewState.model_validate_json(serialized)
    assert rehydrated.round_number == first.state.round_number
    assert rehydrated.think_used == first.state.think_used


def test_step_done_flag_when_orchestrator_says_stop(stub_crew: Crew) -> None:
    runner = CrewRunner(stub_crew)
    state = CrewState(question="hi", round_number=20, max_rounds=12)  # way past max

    result = runner.step(state)
    assert result.done is True
    assert result.event is not None
    assert result.event.kind in {"warning", "final"}


def test_step_drives_full_loop_to_completion() -> None:
    """A canned sequence of LLM responses drives the crew through to final.

    The point of this test is that `CrewRunner.step(state)` can be called
    repeatedly to drive the loop from outside; the final state carries
    the answer. We don't care about the *content* of the answer (that's
    the LLM's job) — we only assert that the loop terminates with an
    answer filled in, and the trace captures the run.
    """
    ollama = _StubOllama(
        [
            # 1: thinker turn 1
            '{"notes": "user wants weather", "search_terms": ["weather Berlin"]}',
            # 2: fetcher turn 1
            '{"tool": "get_weather", "input": {"city": "Berlin"}}',
            # Default for further thinker/fetcher/finalist calls.
        ],
        default="stub answer",
    )

    def get_weather(input: dict) -> dict:
        return {"city": input["city"], "summary": "overcast"}

    crew = Crew(ollama=ollama, tools={"get_weather": get_weather})  # type: ignore[arg-type]
    runner = CrewRunner(crew)
    state = CrewState(question="What is the weather in Berlin?")

    steps = 0
    while not (result := runner.step(state)).done:
        state = result.state
        steps += 1
        if steps > 10:
            pytest.fail("crew did not terminate in 10 steps")

    # Loop terminated with an answer in state.
    assert result.state.current.answer, "expected non-empty final answer"
    assert result.event is not None
    assert result.event.kind == "final"
    assert result.event.payload["answer"] == result.state.current.answer
    # Trace captured the orchestration decisions.
    assert any("orchestrator →" in line for line in result.state.trace)
    assert any("fetch" in line for line in result.state.trace)
    # Tool ran and hit was recorded.
    assert result.state.current.hits
    assert result.state.current.hits[0]["tool"] == "get_weather"


def test_pipeline_status_reports_role_progression() -> None:
    state = CrewState(question="hi")
    # Pretend a fetch happened.
    state.think_used = 1
    state.fetch_used = 1
    state.current.hits.append({"tool": "x", "output": "y"})

    from herding_cats.crew.orchestrator import OrchestratorDecision, pipeline_status

    statuses = pipeline_status(state, None)
    assert statuses["orchestrator"] == "complete"
    assert statuses["thinker"] == "complete"
    assert statuses["fetcher"] == "complete"

    # With a `final` decision in flight, finalist is running.
    statuses = pipeline_status(
        state, OrchestratorDecision(next="final", reason="x", mode="rules")
    )
    assert statuses["finalist"] == "running"


def test_pipeline_status_marks_skipped_for_smalltalk() -> None:
    from herding_cats.crew.orchestrator import pipeline_status

    state = CrewState(question="hi")
    state.current.answer = "hello!"  # smalltalk went straight to final
    statuses = pipeline_status(state, None)
    assert statuses["thinker"] == "skipped"
    assert statuses["fetcher"] == "skipped"
    assert statuses["finalist"] == "complete"
