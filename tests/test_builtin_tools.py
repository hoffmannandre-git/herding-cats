"""Tests for the built-in tool registry."""

from __future__ import annotations

import re

import pytest

from herding_cats.tools_builtin import (
    BUILTIN_TOOL_FUNCS,
    CalcInput,
    DateInput,
    EchoInput,
    TimeInput,
    WebSearchInput,
    builtin_tools,
    calculator,
    date_now,
    echo,
    get_current_time,
    web_search_stub,
)

# ----- each tool runs without error and respects its schema -----------------


def test_get_current_time_default_utc() -> None:
    out = get_current_time(TimeInput())
    assert isinstance(out.iso, str)
    assert isinstance(out.epoch_s, int)
    assert out.epoch_s > 0


def test_get_current_time_with_explicit_tz() -> None:
    out = get_current_time(TimeInput(timezone="Europe/Berlin"))
    assert out.iso
    assert "Berlin" in out.timezone or "+" in out.iso or "0" in out.iso


def test_date_now_returns_iso_date() -> None:
    out = date_now(DateInput())
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", out.date)
    assert out.weekday


# ----- calculator -----------------------------------------------------------


def test_calculator_addition() -> None:
    out = calculator(CalcInput(expression="2 + 3 * 4"))
    assert out.value == 14
    assert out.error is None


def test_calculator_math_sqrt() -> None:
    out = calculator(CalcInput(expression="math.sqrt(16)"))
    assert out.value == 4


def test_calculator_rejects_function_calls() -> None:
    out = calculator(CalcInput(expression="open('x')"))
    assert out.error is not None
    assert "not allowed" in out.error


def test_calculator_rejects_empty() -> None:
    out = calculator(CalcInput(expression=""))
    assert out.error == "empty expression"


def test_calculator_handles_division_by_zero_gracefully() -> None:
    out = calculator(CalcInput(expression="1/0"))
    assert out.error is not None
    assert "ZeroDivision" in out.error


# ----- echo -----------------------------------------------------------------


def test_echo_repeats() -> None:
    out = echo(EchoInput(text="hi ", repeat=3))
    assert out.text == "hi hi hi "
    assert out.repeats == 3


def test_echo_clamps_repeat() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        echo(EchoInput(text="x", repeat=99))


# ----- web search stub ------------------------------------------------------


def test_web_search_stub_returns_empty() -> None:
    out = web_search_stub(WebSearchInput(query="anything"))
    assert out.query == "anything"
    assert out.hits == []
    assert "stub" in out.note.lower()


# ----- registry ------------------------------------------------------------


def test_builtin_tools_includes_everything() -> None:
    specs = builtin_tools()
    names = {s.name for s in specs}
    assert {
        "get_current_time",
        "date_now",
        "calculator",
        "echo",
        "web_search",
    } <= names


def test_builtin_tool_funcs_match_registry() -> None:
    specs = {s.name: s for s in builtin_tools()}
    for name, fn in BUILTIN_TOOL_FUNCS.items():
        assert name in specs
        assert specs[name].fn is fn


def test_calculator_uses_explicit_pydantic_types() -> None:
    """The schema we ship must satisfy the runtime executor validation."""
    from herding_cats.crew.executor import Executor
    from herding_cats.crew.tools import ToolCall, ToolRegistry

    reg = ToolRegistry()
    for spec in builtin_tools():
        reg.specs.append(spec)
    ex = Executor(reg)
    out = ex.run(ToolCall(tool="calculator", input={"expression": "2 + 2"}))
    assert out.error is None
    assert out.validation_error is None
    assert out.output["value"] == 4
