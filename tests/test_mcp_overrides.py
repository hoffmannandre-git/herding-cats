"""Fast tests for the override loader + applier (YAML only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from herding_cats.crew.tools import ToolSpec
from herding_cats.mcp import (
    DESCRIPTION_MAX_CHARS,
    ToolOverride,
    _resolve_dotted_model,
    apply_tool_overrides,
    load_tool_overrides,
)

# ----- helpers --------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str


class _EchoOutput(BaseModel):
    text: str


def _make_spec(name: str, description: str = "orig") -> ToolSpec:
    """A minimal ToolSpec; `fn` doesn't need to run for override tests."""

    async def _fn(_input: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    return ToolSpec(
        name=name,
        description=description,
        fn=_fn,
        input_schema=_EchoInput,
        output_schema=_EchoOutput,
    )


# ----- load_tool_overrides: happy path -------------------------------------


def test_load_minimal_yaml(tmp_path: Path) -> None:
    p = tmp_path / "overrides.yaml"
    p.write_text(
        """
tools:
  get_weather:
    description: "Use this for current weather."
""",
        encoding="utf-8",
    )
    loaded = load_tool_overrides(p)
    assert "get_weather" in loaded
    assert loaded["get_weather"].description == "Use this for current weather."
    assert loaded["get_weather"].triggers == []
    assert loaded["get_weather"].category is None
    assert loaded["get_weather"].input_schema is None


def test_load_yaml_with_triggers_and_category(tmp_path: Path) -> None:
    p = tmp_path / "overrides.yaml"
    p.write_text(
        """
tools:
  get_forecast:
    description: "Multi-day forecasts."
    triggers: ["forecast", "this weekend"]
    category: weather
""",
        encoding="utf-8",
    )
    loaded = load_tool_overrides(p)
    ov = loaded["get_forecast"]
    assert ov.description == "Multi-day forecasts."
    assert ov.triggers == ["forecast", "this weekend"]
    assert ov.category == "weather"


def test_load_yaml_with_defaults(tmp_path: Path) -> None:
    p = tmp_path / "overrides.yaml"
    p.write_text(
        """
defaults:
  category: mcp
  triggers: ["tool"]

tools:
  a:
    description: "Tool A."
  b:
    description: "Tool B."
    category: special
""",
        encoding="utf-8",
    )
    loaded = load_tool_overrides(p)
    assert loaded["a"].category == "mcp"
    assert loaded["a"].triggers == ["tool"]
    assert loaded["b"].category == "special"
    assert loaded["b"].triggers == ["tool"]  # defaults still apply for triggers


def test_load_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_tool_overrides(p) == {}


def test_load_yaml_with_only_defaults(tmp_path: Path) -> None:
    p = tmp_path / "defaults.yaml"
    p.write_text(
        """
defaults:
  category: mcp
""",
        encoding="utf-8",
    )
    assert load_tool_overrides(p) == {}


# ----- load_tool_overrides: validation errors ------------------------------


def test_unknown_field_in_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: "Foo."
    mystery: 42
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown field"):
        load_tool_overrides(p)


def test_non_string_description_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: 42
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="description must be a string"):
        load_tool_overrides(p)


def test_empty_description_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: "   "
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be non-empty"):
        load_tool_overrides(p)


def test_non_list_triggers_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: "Foo."
    triggers: "forecast"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="triggers must be a list"):
        load_tool_overrides(p)


def test_invalid_dotted_schema_path_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: "Foo."
    input_schema: "this.module.does.not.exist.FooInput"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="could not import"):
        load_tool_overrides(p)


def test_dotted_schema_path_not_a_basemodel_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
tools:
  foo:
    description: "Foo."
    input_schema: "builtins.int"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not resolve to a pydantic"):
        load_tool_overrides(p)


def test_dotted_schema_path_resolves_basemodel(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(
        f"""
tools:
  foo:
    description: "Foo."
    input_schema: "{_EchoInput.__module__}._EchoInput"
""",
        encoding="utf-8",
    )
    loaded = load_tool_overrides(p)
    assert loaded["foo"].input_schema is _EchoInput


def test_malformed_top_level_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_tool_overrides(p)


# ----- _resolve_dotted_model: edge cases -----------------------------------


def test_resolve_dotted_with_empty_module_part() -> None:
    with pytest.raises(ValueError, match=r"must be 'package\.module\.Class'"):
        _resolve_dotted_model(Path("dummy.yaml"), "tools.x", "NoModulePrefix")


# ----- apply_tool_overrides: happy path ------------------------------------


def test_apply_replaces_description_preserves_everything_else() -> None:
    spec = _make_spec("get_weather", description="Original")
    overrides = {"get_weather": ToolOverride(description="Better description.")}

    out = apply_tool_overrides([spec], overrides, source="x.yaml")
    assert out[0].description == "Better description."
    assert out[0].name == "get_weather"
    assert out[0].input_schema is _EchoInput
    assert out[0].output_schema is _EchoOutput
    assert out[0].fn is spec.fn


def test_apply_swaps_input_schema() -> None:
    spec = _make_spec("foo")

    class NewSchema(BaseModel):
        url: str

    overrides = {"foo": ToolOverride(input_schema=NewSchema)}
    out = apply_tool_overrides([spec], overrides, source="x.yaml")
    assert out[0].input_schema is NewSchema


def test_apply_preserves_unchanged_tools() -> None:
    a = _make_spec("a", description="A")
    b = _make_spec("b", description="B")
    overrides = {"a": ToolOverride(description="A better.")}
    out = apply_tool_overrides([a, b], overrides, source="x.yaml")
    assert out[0].description == "A better."
    assert out[1] is b


def test_apply_no_overrides_returns_equivalent_specs() -> None:
    a = _make_spec("a")
    overrides: dict[str, ToolOverride] = {}
    out = apply_tool_overrides([a], overrides, source="x.yaml")
    # When no override applies, the original spec passes through unchanged.
    # Field equality is the contract; identity is intentionally preserved.
    assert out[0] is a
    assert out[0].description == a.description


# ----- apply_tool_overrides: warn-on-load ----------------------------------


def test_apply_warns_when_override_tool_missing(caplog) -> None:
    specs = [_make_spec("get_weather")]
    overrides = {"get_wether": ToolOverride(description="typo")}  # typo

    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        out = apply_tool_overrides(specs, overrides, source="x.yaml")

    assert out[0] is specs[0]  # unchanged
    assert any("get_wether" in rec.message for rec in caplog.records)
    assert any("x.yaml" in rec.message for rec in caplog.records)
    assert any("get_weather" in rec.message for rec in caplog.records)
    assert all(rec.levelno == logging.WARNING for rec in caplog.records)


def test_apply_warns_with_correct_logger_name(caplog) -> None:
    specs = [_make_spec("a")]
    overrides = {"missing": ToolOverride(description="x")}

    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        apply_tool_overrides(specs, overrides, source="x.yaml")

    assert any(rec.name == "herding_cats.mcp" for rec in caplog.records)


def test_apply_empty_yaml_produces_no_warnings(caplog) -> None:
    specs = [_make_spec("a")]
    overrides: dict[str, ToolOverride] = {}

    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        apply_tool_overrides(specs, overrides, source="x.yaml")

    assert not any(rec.name == "herding_cats.mcp" for rec in caplog.records)


def test_apply_multi_server_yaml_warns_per_missing(caplog) -> None:
    """One YAML with entries for tools on multiple servers; only some
    servers are connected. We warn per missing entry."""
    specs = [_make_spec("get_weather")]  # only one server connected
    overrides = {
        "get_weather": ToolOverride(description="Fine."),
        "get_forecast": ToolOverride(description="Other server."),
        "raw_query": ToolOverride(description="SQLite."),
    }

    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        out = apply_tool_overrides(specs, overrides, source="multi.yaml")

    assert out[0].description == "Fine."  # the one that matched
    missing_records = [
        r for r in caplog.records
        if "multi.yaml" in r.message and "not found" in r.message
    ]
    assert len(missing_records) == 2  # get_forecast + raw_query


def test_apply_matches_namespaced_name_via_suffix() -> None:
    """YAML uses the un-prefixed name; spec is namespaced (`fetch_foo`)."""
    spec = _make_spec("fetch_foo", description="orig")
    overrides = {"foo": ToolOverride(description="better")}
    out = apply_tool_overrides([spec], overrides, source="x.yaml")
    assert out[0].description == "better"


def test_apply_matches_unnamespaced_name_directly() -> None:
    spec = _make_spec("foo", description="orig")
    overrides = {"foo": ToolOverride(description="better")}
    out = apply_tool_overrides([spec], overrides, source="x.yaml")
    assert out[0].description == "better"


def test_apply_no_warnings_when_override_matches(caplog) -> None:
    spec = _make_spec("foo")
    overrides = {"foo": ToolOverride(description="better")}
    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        apply_tool_overrides([spec], overrides, source="x.yaml")
    assert not any(rec.name == "herding_cats.mcp" for rec in caplog.records)


def test_apply_warns_on_oversized_description(caplog) -> None:
    spec = _make_spec("foo")
    long = "x" * (DESCRIPTION_MAX_CHARS + 1)
    overrides = {"foo": ToolOverride(description=long)}
    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        apply_tool_overrides([spec], overrides, source="x.yaml")
    assert any(
        "description for tool 'foo'" in r.message and str(DESCRIPTION_MAX_CHARS) in r.message
        for r in caplog.records
    )


def test_apply_does_not_warn_on_exact_max_length_description(caplog) -> None:
    spec = _make_spec("foo")
    exact = "x" * DESCRIPTION_MAX_CHARS
    overrides = {"foo": ToolOverride(description=exact)}
    with caplog.at_level(logging.WARNING, logger="herding_cats.mcp"):
        apply_tool_overrides([spec], overrides, source="x.yaml")
    assert not any(
        "recommended" in r.message for r in caplog.records
    )


# ----- apply_tool_overrides: ToolOverride fields not on ToolSpec -----------


def test_apply_ignores_category_and_triggers_silently() -> None:
    """`category` and `triggers` are not ToolSpec fields today; the
    override applier should silently ignore them (they're forward-
    looking metadata for a future structured-description layer)."""
    spec = _make_spec("foo")
    overrides = {
        "foo": ToolOverride(
            description="d",
            category="weather",
            triggers=["forecast"],
        )
    }
    out = apply_tool_overrides([spec], overrides, source="x.yaml")
    assert out[0].description == "d"
