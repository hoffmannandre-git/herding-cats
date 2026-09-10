"""Tests for the deterministic executor."""

from __future__ import annotations

import asyncio

import pytest

from herding_cats.crew.executor import Executor
from herding_cats.crew.tools import ToolCall, ToolRegistry


def test_runs_sync_tool() -> None:
    reg = ToolRegistry()
    reg.register("add", lambda i: {"sum": i["a"] + i["b"]}, description="add two ints")
    ex = Executor(reg)
    result = ex.run(ToolCall(tool="add", input={"a": 2, "b": 3}))
    assert result.output == {"sum": 5}
    assert result.error is None


def test_unknown_tool_returns_error_not_exception() -> None:
    reg = ToolRegistry()
    ex = Executor(reg)
    result = ex.run(ToolCall(tool="nope", input={}))
    assert result.error is not None
    assert "unknown tool" in result.error


def test_tool_exception_caught_when_safe() -> None:
    reg = ToolRegistry()

    def boom(_):
        raise RuntimeError("kaboom")

    reg.register("boom", boom)
    ex = Executor(reg, safe=True)
    result = ex.run(ToolCall(tool="boom", input={}))
    assert result.error is not None
    assert "kaboom" in result.error


def test_tool_exception_propagates_when_unsafe() -> None:
    reg = ToolRegistry()

    def boom(_):
        raise RuntimeError("kaboom")

    reg.register("boom", boom)
    ex = Executor(reg, safe=False)
    with pytest.raises(RuntimeError):
        ex.run(ToolCall(tool="boom", input={}))


def test_arun_handles_async_tool() -> None:
    reg = ToolRegistry()

    async def fetch(i):
        return {"ok": True, "city": i["city"]}

    reg.register("fetch", fetch)
    ex = Executor(reg)
    result = asyncio.run(ex.arun(ToolCall(tool="fetch", input={"city": "Berlin"})))
    assert result.output == {"ok": True, "city": "Berlin"}


def test_arun_many_runs_concurrently() -> None:
    import time

    reg = ToolRegistry()

    async def slow(i):
        await asyncio.sleep(0.05)
        return {"i": i}

    reg.register("slow", slow)
    ex = Executor(reg)
    t0 = time.perf_counter()
    results = asyncio.run(
        ex.arun_many([ToolCall(tool="slow", input=i) for i in (1, 2, 3, 4)])
    )
    elapsed = time.perf_counter() - t0
    # 4 sequential 50ms sleeps = 200ms; concurrent should be ~50ms.
    assert elapsed < 0.15
    assert [r.output["i"] for r in results] == [1, 2, 3, 4]


def test_registry_describe_is_prompt_ready() -> None:
    reg = ToolRegistry()
    reg.register("get_weather", lambda i: {}, description="weather", input_hint="{city}")
    desc = reg.describe()
    assert desc == [
        {
            "name": "get_weather",
            "description": "weather",
            "input": "{city}",
            "output": "(any JSON-serializable value)",
        }
    ]
