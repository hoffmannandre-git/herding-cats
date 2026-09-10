"""Tests for the rules-mode orchestrator.

These run without Ollama. They build `CrewState` instances by hand and
verify the state machine produces the right `next` step.
"""

from __future__ import annotations

from herding_cats.crew.orchestrator import compute_budget, decide_rules
from herding_cats.crew.state import CrewState


def _state(**overrides) -> CrewState:
    s = CrewState(
        question="What is the weather in Berlin?",
        max_rounds=12,
        fetch_max=3,
        think_max=4,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_smalltalk_goes_straight_to_final() -> None:
    s = _state(question="hi")
    d = decide_rules(s)
    assert d.next == "final"
    assert d.mode == "rules"


def test_first_round_thinks() -> None:
    s = _state()
    d = decide_rules(s)
    assert d.next == "think"
    assert d.reason  # has a reason


def test_think_then_fetch() -> None:
    s = _state(last_role="think")
    d = decide_rules(s)
    assert d.next == "fetch"


def test_fetch_with_hits_thinks_again() -> None:
    s = _state(last_role="fetch")
    s.current.hits.append({"tool": "x", "output": "y"})
    d = decide_rules(s)
    assert d.next == "think"


def test_fetch_with_hits_after_reflection_done_goes_final() -> None:
    s = _state(last_role="fetch", reflection_think_done=True)
    s.current.hits.append({"tool": "x", "output": "y"})
    d = decide_rules(s)
    assert d.next == "final"


def test_fetch_with_no_hits_goes_final() -> None:
    s = _state(last_role="fetch")
    d = decide_rules(s)
    assert d.next == "final"


def test_fetch_exhausted_with_no_hits_forces_finish() -> None:
    """Gap 3 enhancement: don't keep searching if budget is gone and nothing came back."""
    s = _state(last_role="fetch", fetch_used=3)
    # No hits; the orchestrator should not pick `fetch` again.
    d = decide_rules(s)
    assert d.next == "final"


def test_fetch_with_hits_but_no_think_budget_goes_final() -> None:
    s = _state(last_role="fetch", think_used=4)
    s.current.hits.append({"tool": "x", "output": "y"})
    d = decide_rules(s)
    assert d.next == "final"


def test_should_finish_forces_final() -> None:
    s = _state(fetch_used=3, think_used=4)
    d = decide_rules(s)
    assert d.next == "final"


def test_compute_budget_pad_is_n_minus_1() -> None:
    s = _state()
    b = compute_budget(s)
    # The orchestrator is told one less than the hard cap.
    assert b.max_rounds == s.max_rounds - 1


def test_budget_summary_includes_counters() -> None:
    s = _state(fetch_used=1, think_used=1, round_number=2)
    b = compute_budget(s)
    summary = b.summary_line()
    assert "fetch 1/" in summary
    assert "think 1/" in summary
