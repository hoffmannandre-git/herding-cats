"""Fast tests for the MCP adapter (discover_tools, schema generation, crew_with_mcp).

These tests do NOT spawn real subprocesses. They use lightweight stubs
in place of `mcp.ClientSession`. The `mcp` SDK itself is required (it's
in the test extra) because `mcp.py` imports it at module load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

# Skip the whole module if the optional `mcp` SDK isn't available.
pytest.importorskip("mcp")

from herding_cats.crew.runner import Crew
from herding_cats.mcp import (
    McpServerSpec,
    _json_schema_to_model,
    _normalize_call_result,
    apply_tool_overrides,
    build_input_schema_from_json_schema,
    connect_mcp_servers,
    crew_with_mcp,
    discover_tools,
)

# ----- fake MCP types ------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeImageBlock:
    def __init__(self, data: str) -> None:
        self.type = "image"
        self.data = data


class _FakeTool:
    """Stand-in for `mcp.types.Tool`."""

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}
        self.outputSchema = output_schema


class _FakeListToolsResult:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self.tools = tools


class _FakeCallResult:
    def __init__(self, content: list[Any], is_error: bool = False) -> None:
        self.content = content
        self.isError = is_error


# ----- schema generation ---------------------------------------------------


def test_build_input_schema_required_string() -> None:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    Model = build_input_schema_from_json_schema("WeatherInput", schema)
    assert issubclass(Model, BaseModel)
    m = Model(city="Berlin")
    assert m.model_dump() == {"city": "Berlin"}
    with pytest.raises(ValidationError):
        Model()  # missing required


def test_build_input_schema_optional_fields_default_none() -> None:
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string"},
        },
        "required": ["city"],
    }
    Model = build_input_schema_from_json_schema("In", schema)
    m = Model(city="Berlin")
    assert m.units is None


def test_build_input_schema_inferred_types() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "f": {"type": "number"},
            "b": {"type": "boolean"},
            "items": {"type": "array"},
            "meta": {"type": "object"},
        },
        "required": ["n", "f", "b", "items", "meta"],
    }
    Model = build_input_schema_from_json_schema("Mixed", schema)
    m = Model(n=1, f=1.5, b=True, items=[], meta={})
    assert m.model_dump() == {"n": 1, "f": 1.5, "b": True, "items": [], "meta": {}}


def test_build_input_schema_type_list_with_null() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"type": ["string", "null"]}},
    }
    Model = build_input_schema_from_json_schema("Nullable", schema)
    m = Model(x=None)
    assert m.x is None


def test_build_input_schema_any_of_becomes_any() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["x"],
    }
    Model = build_input_schema_from_json_schema("AnyOf", schema)
    m = Model(x="ok")
    m2 = Model(x=42)
    assert m.x == "ok"
    assert m2.x == 42


def test_build_input_schema_ref_becomes_any() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Something"}},
        "required": ["x"],
    }
    Model = build_input_schema_from_json_schema("Ref", schema)
    m = Model(x={"foo": "bar"})
    assert m.x == {"foo": "bar"}


def test_build_input_schema_enum_keeps_declared_type() -> None:
    schema = {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["unit"],
    }
    Model = build_input_schema_from_json_schema("Enum", schema)
    m = Model(unit="c")
    assert m.unit == "c"


def test_build_input_schema_empty_object() -> None:
    schema = {"type": "object"}
    Model = build_input_schema_from_json_schema("Empty", schema)
    m = Model()
    assert m.model_dump() == {}


def test_internal_json_schema_to_model_equivalent_to_public() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    A = _json_schema_to_model("A", schema)
    B = build_input_schema_from_json_schema("B", schema)
    assert A(x="hi").model_dump() == B(x="hi").model_dump()


# ----- _normalize_call_result ---------------------------------------------


def test_normalize_single_text_block_with_json_payload() -> None:
    result = _FakeCallResult([_FakeTextBlock('{"temperature": 22}')])
    assert _normalize_call_result(result) == {"temperature": 22}


def test_normalize_single_text_block_with_plain_text() -> None:
    result = _FakeCallResult([_FakeTextBlock("hello world")])
    assert _normalize_call_result(result) == {"text": "hello world"}


def test_normalize_multiple_text_blocks_first_json_wins() -> None:
    result = _FakeCallResult(
        [_FakeTextBlock("not json"), _FakeTextBlock('{"ok": true}')]
    )
    assert _normalize_call_result(result) == {"ok": True}


def test_normalize_multiple_text_blocks_no_json_returns_list() -> None:
    result = _FakeCallResult(
        [_FakeTextBlock("one"), _FakeTextBlock("two")]
    )
    assert _normalize_call_result(result) == {"text": ["one", "two"]}


def test_normalize_is_error_returns_error_dict() -> None:
    result = _FakeCallResult(
        [_FakeTextBlock("rate limit exceeded")], is_error=True
    )
    assert _normalize_call_result(result) == {"error": "rate limit exceeded"}


def test_normalize_empty_content() -> None:
    result = _FakeCallResult([])
    assert _normalize_call_result(result) == {}


def test_normalize_empty_content_is_error() -> None:
    result = _FakeCallResult([], is_error=True)
    assert _normalize_call_result(result) == {"error": "empty result"}


def test_normalize_mixed_content_types() -> None:
    result = _FakeCallResult(
        [
            _FakeTextBlock('{"summary": "sunny"}'),
            _FakeImageBlock("base64data"),
        ]
    )
    out = _normalize_call_result(result)
    assert out["summary"] == "sunny"
    # Other blocks get grouped under their type name.
    assert out.get("image") == ["base64data"]


# ----- discover_tools: namespace prefixing ---------------------------------


class _FakeSession:
    """Minimal stand-in for `mcp.ClientSession` for `discover_tools` tests."""

    def __init__(self, tools: list[_FakeTool], captured: list | None = None) -> None:
        self._tools = tools
        self._captured = captured if captured is not None else []

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeCallResult:
        self._captured.append((name, arguments))
        return _FakeCallResult([_FakeTextBlock('{"echoed": true}')])


@pytest.mark.asyncio
async def test_discover_tools_returns_one_spec_per_tool() -> None:
    session = _FakeSession(
        [
            _FakeTool("get_weather", "weather desc", {"type": "object"}),
            _FakeTool("get_forecast", "forecast desc", {"type": "object"}),
        ]
    )
    specs = await discover_tools(session)
    assert [s.name for s in specs] == ["get_weather", "get_forecast"]
    assert [s.description for s in specs] == ["weather desc", "forecast desc"]


@pytest.mark.asyncio
async def test_discover_tools_namespace_prefix() -> None:
    session = _FakeSession(
        [_FakeTool("fetch", "f", {"type": "object"})]
    )
    specs = await discover_tools(session, namespace="fetch")
    assert specs[0].name == "fetch_fetch"


@pytest.mark.asyncio
async def test_discover_tools_generates_input_schema() -> None:
    session = _FakeSession(
        [
            _FakeTool(
                "echo",
                "e",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]
    )
    specs = await discover_tools(session)
    assert specs[0].input_schema is not None
    m = specs[0].input_schema(text="hi")
    assert m.model_dump() == {"text": "hi"}


@pytest.mark.asyncio
async def test_discover_tools_generates_output_schema_when_present() -> None:
    session = _FakeSession(
        [
            _FakeTool(
                "w",
                "w",
                {"type": "object"},
                output_schema={
                    "type": "object",
                    "properties": {"temp": {"type": "number"}},
                    "required": ["temp"],
                },
            )
        ]
    )
    specs = await discover_tools(session)
    assert specs[0].output_schema is not None
    assert specs[0].output_schema(temp=1.0).model_dump() == {"temp": 1.0}


@pytest.mark.asyncio
async def test_tool_spec_fn_calls_session_call_tool_with_original_name() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []
    session = _FakeSession(
        [_FakeTool("get_weather", "", {"type": "object"})], captured=captured
    )
    specs = await discover_tools(session, namespace="ns")
    # Simulate the executor invoking the fn.
    out = await specs[0].fn({"city": "Berlin"})
    assert captured == [("get_weather", {"city": "Berlin"})]
    assert out == {"echoed": True}


@pytest.mark.asyncio
async def test_tool_spec_fn_accepts_pydantic_input() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []
    session = _FakeSession(
        [
            _FakeTool(
                "echo",
                "",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
        captured=captured,
    )
    specs = await discover_tools(session)
    model = specs[0].input_schema(text="hi")  # pydantic instance
    await specs[0].fn(model)
    assert captured == [("echo", {"text": "hi"})]


@pytest.mark.asyncio
async def test_tool_spec_fn_normalizes_error_result() -> None:
    class _ErrorSession(_FakeSession):
        async def call_tool(self, name, arguments):
            return _FakeCallResult(
                [_FakeTextBlock("bad input")], is_error=True
            )

    session = _ErrorSession([_FakeTool("x", "", {"type": "object"})])
    specs = await discover_tools(session)
    out = await specs[0].fn({})
    assert out == {"error": "bad input"}


# ----- McpServerSpec validation --------------------------------------------


def test_mcp_server_spec_stdio_requires_command() -> None:
    with pytest.raises(ValueError, match="requires `command`"):
        McpServerSpec(transport="stdio")


def test_mcp_server_spec_sse_requires_url() -> None:
    with pytest.raises(ValueError, match="requires `url`"):
        McpServerSpec(transport="sse")


def test_mcp_server_spec_unknown_transport() -> None:
    with pytest.raises(ValueError, match="unknown transport"):
        McpServerSpec(transport="bogus")  # type: ignore[arg-type]


# ----- connect_mcp_servers: lifecycle --------------------------------------


@pytest.mark.asyncio
async def test_connect_mcp_servers_empty_list() -> None:
    async with connect_mcp_servers([]) as sessions:
        assert sessions == []


# ----- crew_with_mcp: end-to-end without real network ---------------------


@pytest.mark.asyncio
async def test_crew_with_mcp_no_servers_returns_original_crew() -> None:
    from herding_cats.ollama import OllamaClient

    crew = Crew(ollama=OllamaClient())
    async with crew_with_mcp(crew) as ctx:
        assert ctx.crew is crew
        assert ctx.sessions == []


# ----- integration: discover + override + crew_with_mcp shape -------------


@pytest.mark.asyncio
async def test_override_applied_after_discovery(tmp_path: Path) -> None:
    """End-to-end shape: discover_tools → load overrides → apply."""
    session = _FakeSession(
        [_FakeTool("get_weather", "ORIGINAL MCP DESCRIPTION", {"type": "object"})]
    )
    specs = await discover_tools(session)
    # Sanity: original description is in place.
    assert specs[0].description == "ORIGINAL MCP DESCRIPTION"

    override_path = tmp_path / "overrides.yaml"
    override_path.write_text(
        """
tools:
  get_weather:
    description: "Better description for the model."
""",
        encoding="utf-8",
    )
    from herding_cats.mcp import apply_tool_overrides, load_tool_overrides

    overrides = load_tool_overrides(override_path)
    out = apply_tool_overrides(specs, overrides, source=override_path)
    assert out[0].description == "Better description for the model."
    assert "ORIGINAL MCP DESCRIPTION" not in out[0].description


@pytest.mark.asyncio
async def test_override_namespaced_works_through_apply() -> None:
    session = _FakeSession(
        [_FakeTool("get_weather", "orig", {"type": "object"})]
    )
    specs = await discover_tools(session, namespace="fetch")
    assert specs[0].name == "fetch_get_weather"
    overrides = {"get_weather": _make_override("Better.")}
    out = apply_tool_overrides(specs, overrides, source="x.yaml")
    assert out[0].description == "Better."
    assert out[0].name == "fetch_get_weather"


def _make_override(description: str):  # small local helper
    from herding_cats.mcp import ToolOverride

    return ToolOverride(description=description)
