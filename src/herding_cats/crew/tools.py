"""Tool registry + dataclasses for the fetcher's plan.

A tool is a plain callable registered by name. The fetcher emits a
`ToolCall`; the executor turns it into a function call.

Tools may optionally declare pydantic input/output schemas. When they
do, the registry renders a JSON-schema block into the fetcher's prompt
automatically, so the model sees the exact shape it must produce. The
executor validates inputs against the schema before invoking the
callable, and validates outputs before recording them on the state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# A tool is a sync or async function from `dict` to a JSON-serializable value.
ToolFn = Callable[[dict[str, Any]], Any | Awaitable[Any]]


def _json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema for a pydantic model, simplified for prompts.

    We strip `$defs`, `$ref`, and other deep-nesting artifacts that
    confuse local models. The output is a flat dict like:

        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "..."},
                "unit": {"type": "string", "enum": ["c", "f"]}
            },
            "required": ["city"]
        }
    """
    raw = model.model_json_schema()
    # Flatten refs — local models don't reuse definitions much, and
    # local LLMs mishandle $ref. Walk properties and inline any $ref.
    defs = raw.pop("$defs", {})
    properties = raw.get("properties", {})
    flat_props: dict[str, Any] = {}
    for name, prop in properties.items():
        if "$ref" in prop:
            ref_name = prop["$ref"].rsplit("/", 1)[-1]
            if ref_name in defs:
                # Inline the referenced schema's properties.
                ref_schema = defs[ref_name]
                flat_props[name] = {
                    k: v for k, v in ref_schema.items() if k != "title"
                }
                continue
        flat_props[name] = {k: v for k, v in prop.items() if k != "title"}
    raw["properties"] = flat_props
    return raw


@dataclass
class ToolSpec:
    """The fetcher-visible description of one tool.

    `name` must be unique within a registry.
    `description` is shown to the model; keep it short.
    `input_schema` and `output_schema` are optional pydantic models.
    When provided, they drive both the prompt rendering and the executor's
    validation. When omitted, the legacy `input_hint` string is used.
    """

    name: str
    description: str
    fn: ToolFn
    # Pydantic schemas (preferred over `input_hint` when both are set).
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    # Legacy free-form hint — only used when no input_schema is set.
    input_hint: str = ""


@dataclass
class ToolRegistry:
    """Name → callable mapping with descriptions."""

    specs: list[ToolSpec] = field(default_factory=list)

    def register(
        self,
        name: str,
        fn: ToolFn,
        *,
        description: str = "",
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        input_hint: str = "",
    ) -> ToolSpec:
        """Register a tool. Returns the spec for chaining / inspection."""
        spec = ToolSpec(
            name=name,
            description=description,
            fn=fn,
            input_schema=input_schema,
            output_schema=output_schema,
            input_hint=input_hint,
        )
        self.specs.append(spec)
        return spec

    def by_name(self, name: str) -> ToolSpec | None:
        for s in self.specs:
            if s.name == name:
                return s
        return None

    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def describe(self) -> list[dict[str, Any]]:
        """Return a list of `{name, description, input, output}` for the prompt.

        Each tool's `input` and `output` are either a JSON-schema dict
        (when schemas are provided) or a free-form hint string. This is
        the structure the fetcher's prompt template iterates over.
        """
        out: list[dict[str, Any]] = []
        for s in self.specs:
            entry: dict[str, Any] = {
                "name": s.name,
                "description": s.description,
            }
            if s.input_schema is not None:
                entry["input"] = _json_schema_for(s.input_schema)
            else:
                entry["input"] = s.input_hint
            if s.output_schema is not None:
                entry["output"] = _json_schema_for(s.output_schema)
            else:
                entry["output"] = "(any JSON-serializable value)"
            out.append(entry)
        return out


@dataclass
class ToolCall:
    """One tool call emitted by the fetcher."""

    tool: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The executor's response for one ToolCall."""

    tool: str
    input: dict[str, Any]
    output: Any
    error: str | None = None
    #: Set when the input failed schema validation; the fetcher can retry
    #: with corrected input on the next pass.
    validation_error: str | None = None
