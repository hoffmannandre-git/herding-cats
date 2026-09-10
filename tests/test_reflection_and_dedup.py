"""Tests for Gap 3 (reflection_think_done) and Gap 4 (fetcher dedup retry).

These are pure-logic tests against the rules-mode orchestrator and
the runner's `_do_fetch` dedup path. No Ollama needed.
"""

from __future__ import annotations

from herding_cats import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Crew,
    CrewRunner,
    CrewState,
    OllamaClient,
)
from herding_cats.crew.orchestrator import decide_rules


class _RecordingOllama(OllamaClient):
    def __init__(self, responses: list[str] | None = None) -> None:  # type: ignore[no-untyped-def]
        self.base_url = "http://stub"
        self.timeout_s = 0.0
        self.calls: list[ChatRequest] = []
        self._responses = list(responses or [])

    async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
        self.calls.append(request)
        text = self._responses.pop(0) if self._responses else ""
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=text),
            done=True,
        )

    def health(self) -> bool:  # type: ignore[override]
        return True


# ----- Gap 3: reflection flag -------------------------------------------------


def test_reflection_think_required_once_after_successful_fetch() -> None:
    state = CrewState(question="what about the rate?")
    state.last_role = "fetch"
    state.think_used = 1
    state.fetch_used = 1
    state.current.hits.append({"tool": "x", "output": "y"})
    # Reflection not done → orchestrator should ask for another think pass.
    d = decide_rules(state)
    assert d.next == "think"
    assert "reflect" in d.reason.lower() or "evaluate" in d.reason.lower()


def test_reflection_think_skipped_after_flag_set() -> None:
    state = CrewState(question="what about the rate?")
    state.last_role = "fetch"
    state.think_used = 1
    state.fetch_used = 1
    state.current.hits.append({"tool": "x", "output": "y"})
    state.reflection_think_done = True  # already reflected
    d = decide_rules(state)
    assert d.next == "final"


def test_reflection_think_blocked_when_think_budget_exhausted() -> None:
    state = CrewState(question="what about the rate?")
    state.last_role = "fetch"
    state.think_used = 4  # max
    state.fetch_used = 1
    state.current.hits.append({"tool": "x", "output": "y"})
    d = decide_rules(state)
    assert d.next == "final"


# ----- Gap 4: fetcher dedup retry --------------------------------------------


def test_dedup_retry_records_query_in_state() -> None:
    """After a successful fetch, state.queries_executed should contain the query."""
    ollama = _RecordingOllama(
        responses=[
            # First call: orchestrator is rules-mode, no LLM call.
            # Second call would be the thinker (rules-mode → think first).
            # The orchestrator picks think; the thinker gets a queued response
            # so it produces a search term.
            '{"notes": "weather", "search_terms": ["berlin weather"]}',
            # Then the orchestrator picks fetch; the fetcher gets this response.
            '{"tool": "search", "input": {"query": "berlin weather"}}',
        ]
    )

    def search(input: dict) -> dict:
        return {"results": []}

    crew = Crew(ollama=ollama, tools={"search": search})  # type: ignore[arg-type]
    runner = CrewRunner(crew)
    state = CrewState(question="weather in Berlin")
    # First step: orchestrator → think, thinker runs.
    r1 = runner.step(state)
    # Second step: orchestrator → fetch, fetcher runs.
    r2 = runner.step(r1.state)
    assert r2.state.queries_executed == ["berlin weather"]


def test_dedup_helper_lowercases_for_comparison() -> None:
    from herding_cats.crew.runner import _executed_queries, _extract_query
    from herding_cats.crew.tools import ToolCall

    state = CrewState(question="x")
    state.queries_executed = ["Berlin Weather"]
    assert _executed_queries(state) == {"berlin weather"}

    call = ToolCall(tool="search", input={"query": "BERLIN WEATHER"})
    assert _extract_query(call) == "berlin weather"
    assert _extract_query(call) in _executed_queries(state)


def test_extract_query_returns_none_for_non_search_tool() -> None:
    from herding_cats.crew.runner import _extract_query
    from herding_cats.crew.tools import ToolCall

    call = ToolCall(tool="get_weather", input={"city": "Berlin"})
    assert _extract_query(call) is None


def test_extract_query_returns_none_when_input_empty() -> None:
    from herding_cats.crew.runner import _extract_query
    from herding_cats.crew.tools import ToolCall

    assert _extract_query(None) is None
    assert _extract_query(ToolCall(tool="x", input={})) is None
