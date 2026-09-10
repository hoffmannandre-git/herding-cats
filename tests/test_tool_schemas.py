"""Tests for the pydantic-schema-backed tool registry.

ToolSpec now carries optional `input_schema` and `output_schema`; the
registry renders JSON-schema blocks for the fetcher's prompt, and the
executor validates against them.
"""

from __future__ import annotations

from pydantic import BaseModel

from herding_cats.crew.executor import Executor
from herding_cats.crew.tools import ToolCall, ToolRegistry, ToolSpec

# ----- helpers ---------------------------------------------------------------


class CityIn(BaseModel):
    city: str


class WeatherOut(BaseModel):
    temp_c: float
    summary: str


def get_weather(input: CityIn) -> WeatherOut:
    return WeatherOut(temp_c=17.0, summary="overcast")


# ----- tests -----------------------------------------------------------------


def test_tool_spec_accepts_pydantic_schemas() -> None:
    spec = ToolSpec(
        name="get_weather",
        description="weather",
        fn=get_weather,
        input_schema=CityIn,
        output_schema=WeatherOut,
    )
    assert spec.input_schema is CityIn
    assert spec.output_schema is WeatherOut


def test_registry_describe_renders_json_schema_for_inputs() -> None:
    reg = ToolRegistry()
    reg.register(
        "get_weather", get_weather,
        description="weather",
        input_schema=CityIn,
        output_schema=WeatherOut,
    )
    desc = reg.describe()
    assert desc[0]["name"] == "get_weather"
    assert "input" in desc[0]
    inp = desc[0]["input"]
    assert inp["type"] == "object"
    assert "city" in inp["properties"]
    assert "city" in inp["required"]


def test_registry_describe_keeps_hint_string_for_untyped_tools() -> None:
    reg = ToolRegistry()
    reg.register("echo", lambda i: i, description="echo", input_hint="{text}")
    desc = reg.describe()
    assert desc[0]["input"] == "{text}"
    assert desc[0]["output"] == "(any JSON-serializable value)"


def test_executor_validates_input_schema() -> None:
    reg = ToolRegistry()
    reg.register(
        "get_weather", get_weather,
        description="weather",
        input_schema=CityIn,
    )
    ex = Executor(reg)
    # Missing required field "city".
    result = ex.run(ToolCall(tool="get_weather", input={}))
    assert result.error is None
    assert result.validation_error is not None
    assert "input_schema" in result.validation_error


def test_executor_passes_valid_input_through() -> None:
    reg = ToolRegistry()
    reg.register(
        "get_weather", get_weather,
        description="weather",
        input_schema=CityIn,
        output_schema=WeatherOut,
    )
    ex = Executor(reg)
    result = ex.run(ToolCall(tool="get_weather", input={"city": "Berlin"}))
    assert result.validation_error is None
    assert result.error is None
    assert result.output["summary"] == "overcast"


def test_executor_validates_output_schema() -> None:
    def bad_tool(input: CityIn) -> dict:
        return {"wrong_key": "no temp_c"}

    reg = ToolRegistry()
    reg.register(
        "bad", bad_tool,
        description="bad",
        input_schema=CityIn,
        output_schema=WeatherOut,
    )
    ex = Executor(reg)
    result = ex.run(ToolCall(tool="bad", input={"city": "X"}))
    assert result.validation_error is not None
    assert "output_schema" in result.validation_error


def test_crew_picks_tool_specs_over_plain_tools() -> None:
    """If both `tool_specs` and `tools` are set, specs win for that name."""
    from herding_cats import Crew, OllamaClient

    spec = ToolSpec(
        name="x", description="x", fn=lambda i: {"from": "spec"},
        input_schema=CityIn, output_schema=WeatherOut,
    )
    crew = Crew(ollama=OllamaClient(), tool_specs=[spec], tools={"x": lambda i: {"from": "fn"}})
    reg = crew.registry()
    assert reg.by_name("x") is spec
