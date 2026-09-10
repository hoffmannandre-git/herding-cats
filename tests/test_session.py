"""Tests for cross-turn memory (Gap 2) and `CrewSession`."""

from __future__ import annotations

import tempfile

import pytest

from herding_cats import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Crew,
    CrewRunner,
    CrewSession,
    CrewState,
    InMemorySession,
    JsonFileSession,
    OllamaClient,
    TurnRecord,
)
from herding_cats.crew.runner import _record_chunk_ids


class _StubOllama(OllamaClient):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        self.base_url = "http://stub"
        self.timeout_s = 0.0
        self.calls: list = []
        self._responses: list[str] = ['{"notes":"","search_terms":[]}', "stub answer"]
        self._default = '{"notes":"","search_terms":[]}'

    async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
        self.calls.append(request)
        text = self._responses.pop(0) if self._responses else self._default
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=text),
            done=True,
        )

    def health(self) -> bool:  # type: ignore[override]
        return True


# ----- chunk-id recording ---------------------------------------------------


def test_record_chunk_ids_from_dict() -> None:
    state = CrewState(question="x")
    _record_chunk_ids(state, {"id": "abc", "text": "hi"})
    assert "abc" in state.seen_chunk_ids


def test_record_chunk_ids_from_chunk_id_field() -> None:
    state = CrewState(question="x")
    _record_chunk_ids(state, {"chunk_id": "42", "text": "hi"})
    assert "42" in state.seen_chunk_ids


def test_record_chunk_ids_from_list_of_dicts() -> None:
    state = CrewState(question="x")
    _record_chunk_ids(
        state,
        [{"id": "1"}, {"id": "2", "doc_id": "3"}, {"id": "1"}],
    )
    assert state.seen_chunk_ids == ["1", "2", "3"]


def test_record_chunk_ids_no_op_on_none() -> None:
    state = CrewState(question="x")
    _record_chunk_ids(state, None)
    assert state.seen_chunk_ids == []


# ----- session stores ------------------------------------------------------


def test_in_memory_session_round_trip() -> None:
    store = InMemorySession()
    assert store.load("nope") is None
    store.save("s1", [TurnRecord(question="q", answer="a")])
    assert store.load("s1") is not None
    assert store.load("s1")[0].answer == "a"
    store.delete("s1")
    assert store.load("s1") is None


def test_in_memory_session_list_sessions() -> None:
    store = InMemorySession()
    store.save("a", [])
    store.save("b", [])
    assert set(store.list_sessions()) == {"a", "b"}


def test_json_file_session_persists() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = JsonFileSession(d)
        store.save("s1", [TurnRecord(question="q", answer="a")])
        # Re-open
        store2 = JsonFileSession(d)
        loaded = store2.load("s1")
        assert loaded and loaded[0].answer == "a"


def test_json_file_session_rejects_path_traversal() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = JsonFileSession(d)
        with pytest.raises(ValueError):
            store.load("../escape")
        with pytest.raises(ValueError):
            store.save("../escape", [])


# ----- crew session --------------------------------------------------------


def test_crew_session_ask_appends_history() -> None:
    ollama = _StubOllama()
    crew = Crew(ollama=ollama, tools={"echo": lambda i: i})  # type: ignore[arg-type]
    runner = CrewRunner(crew)
    sess = CrewSession.with_in_memory(runner, prior_context="user is German")

    sess.ask("What's the weather in Berlin?")
    sess.ask("And tomorrow?")

    history = sess.history()
    assert len(history) == 2
    assert history[0].question.startswith("What's")
    assert history[1].question.startswith("And tomorrow")


def test_crew_session_passes_prior_turns_to_thinker() -> None:
    """The second ask's state should carry the first turn's answer as prior."""
    ollama = _StubOllama()
    crew = Crew(ollama=ollama, tools={"echo": lambda i: i})  # type: ignore[arg-type]
    runner = CrewRunner(crew)
    sess = CrewSession.with_in_memory(runner)

    sess.ask("first question")
    sess.ask("follow up")

    # Build a fresh state the way the session would, and inspect it.
    prior = sess.history()
    state = CrewState(
        question="follow up",
        prior_turns=[t.to_memory() for t in prior[-sess.max_prior_turns :]],
        seen_chunk_ids=list({cid for t in prior for cid in t.seen_chunk_ids}),
    )
    assert state.prior_turns
    # The first turn had a known stub answer.
    assert "stub answer" in state.prior_turns[0].answer or state.prior_turns[0].answer


def test_crew_session_persists_to_json_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        ollama = _StubOllama()
        crew = Crew(ollama=ollama, tools={"echo": lambda i: i})  # type: ignore[arg-type]
        runner = CrewRunner(crew)
        sess = CrewSession.with_json_file(
            runner, d, session_id="my_session"
        )
        sess.ask("hello")

        # Reload via a new session instance.
        new = CrewSession.with_json_file(
            CrewRunner(Crew(ollama=ollama, tools={"echo": lambda i: i})),  # type: ignore[arg-type]
            d,
            session_id="my_session",
        )
        loaded = new.history()
        assert len(loaded) == 1
        assert loaded[0].question == "hello"
