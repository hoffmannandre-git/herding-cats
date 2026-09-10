"""Tests for Gap 6 (search-term queue pop) and Gap 7 (state.trace)."""

from __future__ import annotations

from herding_cats.crew.runner import _pop_search_term
from herding_cats.crew.state import CrewState


def test_pop_returns_next_term_and_advances_index() -> None:
    state = CrewState(question="x")
    state.current.search_terms = ["a", "b", "c"]

    assert _pop_search_term(state) == "a"
    assert state.current.search_term_index == 1
    assert _pop_search_term(state) == "b"
    assert state.current.search_term_index == 2
    assert _pop_search_term(state) == "c"
    assert state.current.search_term_index == 3


def test_pop_skips_blank_terms() -> None:
    state = CrewState(question="x")
    state.current.search_terms = ["", "  ", "real", ""]

    assert _pop_search_term(state) == "real"
    assert _pop_search_term(state) is None


def test_pop_returns_none_when_exhausted() -> None:
    state = CrewState(question="x")
    state.current.search_terms = ["only"]
    _pop_search_term(state)
    assert _pop_search_term(state) is None


def test_trace_grows_during_runner_steps() -> None:
    """End-to-end trace log: at minimum, every step adds at least one line."""
    from herding_cats import (
        ChatMessage,
        ChatRequest,
        ChatResponse,
        Crew,
        CrewRunner,
        OllamaClient,
    )

    class _Stub(OllamaClient):
        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            self.base_url = "http://stub"
            self.timeout_s = 0.0
            self.calls: list = []
            self._responses = ['{"notes":"","search_terms":[]}']

        async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
            self.calls.append(request)
            return ChatResponse(
                model=request.model,
                message=ChatMessage(role="assistant", content=self._responses.pop(0)),
                done=True,
            )

        def health(self) -> bool:  # type: ignore[override]
            return True

    crew = Crew(ollama=_Stub())  # type: ignore[arg-type]
    runner = CrewRunner(crew)
    state = CrewState(question="hi")

    r = runner.step(state)
    assert r.state.trace
    assert any("orchestrator" in line for line in r.state.trace)


def test_shortcut_bypasses_crew() -> None:
    """Gap 5: a registered pattern short-circuits the crew entirely."""
    from herding_cats import (
        ChatMessage,
        ChatRequest,
        ChatResponse,
        Crew,
        CrewRunner,
        OllamaClient,
    )

    class _Stub(OllamaClient):
        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            self.base_url = "http://stub"
            self.timeout_s = 0.0
            self.calls: list = []
            self._responses: list[str] = []

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

    crew = Crew(  # type: ignore[arg-type]
        ollama=_Stub(),
        shortcuts={
            r"^what('?s| is) my status\??$": lambda i: "all good!",
        },
    )
    runner = CrewRunner(crew)

    events = list(runner.run_stream("What's my status?"))
    assert not runner.crew.ollama.calls, "shortcut should not call the LLM"
    final = [e for e in events if e.kind == "final"]
    assert final and final[0].payload["answer"] == "all good!"


def test_shortcut_handler_failure_falls_through() -> None:
    """A raising shortcut handler falls through to the regular crew path."""
    from herding_cats import (
        ChatMessage,
        ChatRequest,
        ChatResponse,
        Crew,
        CrewRunner,
        OllamaClient,
    )

    class _Stub(OllamaClient):
        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            self.base_url = "http://stub"
            self.timeout_s = 0.0
            self.calls: list = []
            self._responses: list[str] = ['{"notes":"","search_terms":[]}', "fall-through answer"]
            self._default = '{"notes":"","search_terms":[]}'

        async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
            self.calls.append(request)
            text = (
                self._responses.pop(0)
                if self._responses
                else self._default
            )
            return ChatResponse(
                model=request.model,
                message=ChatMessage(role="assistant", content=text),
                done=True,
            )

        def health(self) -> bool:  # type: ignore[override]
            return True

    def boom(_: dict) -> str:
        raise RuntimeError("nope")

    crew = Crew(  # type: ignore[arg-type]
        ollama=_Stub(),
        shortcuts={r"^list .*": boom},
    )
    runner = CrewRunner(crew)

    events = list(runner.run_stream("list all contacts"))
    # Crew should still complete and call the LLM at least once.
    assert runner.crew.ollama.calls
    final = [e for e in events if e.kind == "final"]
    assert final
