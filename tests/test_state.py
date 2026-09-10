"""Tests for the serializable state types (Gap 1).

`CrewState` and `TurnMemory` must round-trip through JSON so a UI or
HTTP endpoint can send modified state back to the runner.
"""

from __future__ import annotations

import json

from herding_cats.crew.state import CrewState, TurnMemory


def test_crew_state_round_trips_through_json() -> None:
    state = CrewState(
        question="What's the weather in Berlin?",
        max_rounds=8,
        fetch_max=2,
        think_max=3,
    )
    state.append_trace("first line")
    state.append_trace("second line")
    state.current.hits.append({"tool": "get_weather", "output": {"temp_c": 17}})

    # to JSON
    payload = state.model_dump_json()
    assert isinstance(payload, str)

    # and back
    rehydrated = CrewState.model_validate_json(payload)
    assert rehydrated.question == state.question
    assert rehydrated.max_rounds == 8
    assert rehydrated.trace == ["first line", "second line"]
    assert rehydrated.current.hits == [{"tool": "get_weather", "output": {"temp_c": 17}}]


def test_crew_state_uses_dict_like_json() -> None:
    state = CrewState(question="hi")
    body = json.loads(state.model_dump_json())
    assert body["question"] == "hi"
    assert body["max_rounds"] == 12  # default
    assert body["trace"] == []  # default factory


def test_turn_memory_defaults() -> None:
    tm = TurnMemory()
    assert tm.notes == ""
    assert tm.search_terms == []
    assert tm.search_term_index == 0
    assert tm.hits == []
    assert tm.warnings == []
    assert tm.answer == ""


def test_state_can_carry_prior_turns() -> None:
    prior = TurnMemory(answer="It is overcast.", search_terms=["Berlin weather"])
    state = CrewState(
        question="What about tomorrow?",
        prior_turns=[prior],
    )
    assert len(state.prior_turns) == 1
    assert state.prior_turns[0].answer == "It is overcast."


def test_state_seen_chunk_ids_default_empty() -> None:
    state = CrewState(question="hi")
    assert state.seen_chunk_ids == []


def test_state_initializes_trace() -> None:
    state = CrewState(question="hi")
    assert state.trace == []
    state.append_trace("hi")
    assert state.trace == ["hi"]


def test_reflection_flag_starts_false() -> None:
    state = CrewState(question="hi")
    assert state.reflection_think_done is False


def test_queries_executed_starts_empty() -> None:
    state = CrewState(question="hi")
    assert state.queries_executed == []
