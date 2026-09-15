"""Consume MCP (Model Context Protocol) servers as a tool source.

`herding_cats` already has a tight contract for tools: `ToolSpec` +
`Executor.arun()`. MCP servers expose tools over JSON-RPC. This module
is the adapter that bridges the two.

Three pieces:

1. `McpServerSpec` + `connect_mcp_servers(...)` — describe one or more
   MCP servers (stdio subprocess or HTTP+SSE) and open them as
   `ClientSession`s inside an async context manager.
2. `discover_tools(session, ...)` — call `session.list_tools()` and
   build `ToolSpec` objects whose async `fn` proxies back through
   `session.call_tool(name, input)`. Schemas are converted from
   JSON Schema to pydantic `BaseModel` in loose mode.
3. `load_tool_overrides(path)` + `apply_tool_overrides(specs, overrides)`
   — let users replace the descriptions (and optionally schemas) that
   MCP servers advertise via a YAML file. Unknown overrides emit a
   WARNING log and are silently skipped (design decision #4).

Usage:

    import asyncio
    from pathlib import Path

    from herding_cats import Crew, CrewRunner, OllamaClient
    from herding_cats.mcp import (
        McpServerSpec,
        apply_tool_overrides,
        connect_mcp_servers,
        crew_with_mcp,
        discover_tools,
        load_tool_overrides,
    )


    async def main():
        crew = Crew(ollama=OllamaClient())
        async with crew_with_mcp(
            crew,
            servers=[McpServerSpec(
                transport="stdio",
                command="uvx",
                args=["mcp-server-fetch"],
                namespace="fetch",
                overrides_path=Path("./mcp_tool_overrides.yaml"),
            )],
        ) as ctx:
            answer = await CrewRunner(ctx.crew).arun(
                "Fetch the latest llama.cpp release notes"
            )
            print(answer)


    if __name__ == "__main__":
        asyncio.run(main())

Install with the `mcp` extra:

    pip install herding-cats[mcp]
"""

from __future__ import annotations

import importlib
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, create_model

from herding_cats.crew.runner import Crew
from herding_cats.crew.tools import ToolSpec

logger = logging.getLogger("herding_cats.mcp")

# Default cap on description length in the YAML file. The fetcher prompt
# still renders longer descriptions, but the loader warns at this size so
# the user gets feedback before the runtime eats the tokens.
DESCRIPTION_MAX_CHARS = 800


# ---------------------------------------------------------------------------
# Optional mcp SDK imports
# ---------------------------------------------------------------------------
#
# The official `mcp` Python SDK is an optional dependency — the same
# pattern used for the FastAPI server extra. We import at module load
# and expose the symbols; if the user didn't install the `[mcp]` extra
# we leave the names unbound and raise a clear ImportError on first use.
try:
    from mcp import ClientSession as _ClientSession
    from mcp.client.sse import sse_client as _sse_client
    from mcp.client.stdio import StdioServerParameters as _StdioServerParameters
    from mcp.client.stdio import stdio_client as _stdio_client

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _ClientSession = None  # type: ignore[assignment]
    _sse_client = None  # type: ignore[assignment]
    _StdioServerParameters = None  # type: ignore[assignment]
    _stdio_client = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


def _require_mcp() -> None:
    if not _MCP_AVAILABLE:
        raise ImportError(
            "herding_cats.mcp requires the 'mcp' extra. "
            "Install with: pip install herding-cats[mcp]"
        )


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


Transport = Literal["stdio", "sse"]


@dataclass
class McpServerSpec:
    """Description of one MCP server to connect to.

    Attributes:
        transport: `"stdio"` spawns a subprocess; `"sse"` opens HTTP+SSE.
        command: stdio only. The executable to spawn.
        args: stdio only. CLI arguments.
        env: stdio only. Extra env vars (merged with `os.environ`).
        url: SSE only. The HTTP+SSE endpoint URL.
        headers: SSE only. Optional HTTP headers (e.g. for auth).
        namespace: Optional prefix added to every discovered tool name to
            avoid collisions when connecting to multiple servers (e.g.
            `fetch` becomes `fetch_fetch` with `namespace="fetch"`).
        overrides_path: Optional path to a YAML file with description /
            schema overrides. See `load_tool_overrides`.
    """

    transport: Transport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    namespace: str | None = None
    overrides_path: Path | None = None

    def __post_init__(self) -> None:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("McpServerSpec(transport='stdio') requires `command`")
        elif self.transport == "sse":
            if not self.url:
                raise ValueError("McpServerSpec(transport='sse') requires `url`")
        else:  # pragma: no cover - Literal guards this
            raise ValueError(f"unknown transport: {self.transport!r}")


@dataclass
class ToolOverride:
    """A single per-tool override loaded from the YAML file.

    All fields are optional; supplying only some is fine. Unknown fields
    in the YAML raise — see `load_tool_overrides`.
    """

    description: str | None = None
    input_schema: type[BaseModel] | None = None
    category: str | None = None
    triggers: list[str] = field(default_factory=list)


@dataclass
class _McpSessionHandle:
    """One open MCP client session plus the context managers that own it."""

    spec: McpServerSpec
    session: Any  # mcp.ClientSession
    # The stack keeps the underlying transport + session cmgrs alive.
    _stack: AsyncExitStack


@dataclass
class _McpBoundCrew:
    """Result of entering `crew_with_mcp(...)`.

    `crew` is the original `Crew` with its `tool_specs` extended by all
    discovered + overridden MCP tools. `sessions` is the list of open
    `ClientSession`s, in the same order as the input `McpServerSpec`s.
    """

    crew: Crew
    sessions: list[Any]  # list[mcp.ClientSession]


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


async def _open_one(spec: McpServerSpec, stack: AsyncExitStack) -> Any:
    """Open one MCP client session and push all its cmgrs onto `stack`.

    Returns an initialized `ClientSession`. Callers are responsible for
    closing the stack when done.
    """
    _require_mcp()
    if spec.transport == "stdio":
        import os

        params = _StdioServerParameters(
            command=spec.command or "",
            args=list(spec.args),
            env={**os.environ, **(spec.env or {})},
        )
        read, write = await stack.enter_async_context(_stdio_client(params))
    elif spec.transport == "sse":
        read, write = await stack.enter_async_context(
            _sse_client(spec.url or "", headers=spec.headers)
        )
    else:  # pragma: no cover - guarded by McpServerSpec.__post_init__
        raise ValueError(f"unknown transport: {spec.transport!r}")

    session = await stack.enter_async_context(_ClientSession(read, write))
    await session.initialize()
    return session


class connect_mcp_servers:
    """Async context manager: open N MCP servers, yield their sessions.

    Usage::

        async with connect_mcp_servers([McpServerSpec(...), ...]) as sessions:
            for session in sessions:
                tools = await discover_tools(session)

    One `ClientSession` per spec; concurrent calls to MCP servers run
    in parallel naturally because each session has its own transport.
    """

    def __init__(self, specs: list[McpServerSpec]) -> None:
        self._specs = list(specs)
        self._stack: AsyncExitStack | None = None
        self._handles: list[_McpSessionHandle] = []

    async def __aenter__(self) -> list[Any]:
        self._stack = AsyncExitStack()
        try:
            for spec in self._specs:
                session = await _open_one(spec, self._stack)
                self._handles.append(
                    _McpSessionHandle(spec=spec, session=session, _stack=self._stack)
                )
        except BaseException:
            await self._stack.__aexit__(type(self._stack), None, None)
            raise
        return [h.session for h in self._handles]

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
            self._stack = None
            self._handles = []


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def discover_tools(
    session: Any, *, namespace: str | None = None
) -> list[ToolSpec]:
    """List the tools an MCP server advertises and wrap each as a `ToolSpec`.

    The returned specs have an async `fn` that proxies through
    `session.call_tool(name, input)`. The input schema is generated from
    the MCP tool's `inputSchema` (loose mode); if the server also ships an
    `outputSchema`, the output schema gets the same treatment.
    """
    _require_mcp()
    result = await session.list_tools()
    raw_tools = getattr(result, "tools", []) or []
    specs: list[ToolSpec] = []
    for raw in raw_tools:
        specs.append(_build_tool_spec(session, raw, namespace=namespace))
    return specs


def _build_tool_spec(
    session: Any, raw: Any, *, namespace: str | None
) -> ToolSpec:
    """Build one `ToolSpec` from one MCP `Tool` object."""
    name: str = raw.name
    description: str = getattr(raw, "description", "") or ""
    if namespace:
        name = f"{namespace}_{name}"
    input_schema = _json_schema_to_model(f"{name}_input", raw.inputSchema or {})
    output_schema = None
    out_schema_raw = getattr(raw, "outputSchema", None)
    if out_schema_raw:
        output_schema = _json_schema_to_model(f"{name}_output", out_schema_raw)
    fn = _make_call_closure(session, raw.name)
    return ToolSpec(
        name=name,
        description=description,
        fn=fn,
        input_schema=input_schema,
        output_schema=output_schema,
    )


def _make_call_closure(session: Any, original_name: str):
    """Async closure: forwards `ToolCall`s to `session.call_tool`.

    Renamed because the spec's `name` may have a namespace prefix; we
    always call the MCP server with the un-prefixed original name.
    """
    async def _call(input_dict: dict[str, Any]) -> dict[str, Any]:
        # `input_dict` may be a pydantic model instance if the executor
        # validated the input; flatten it.
        if isinstance(input_dict, BaseModel):
            input_dict = input_dict.model_dump()
        result = await session.call_tool(original_name, input_dict)
        return _normalize_call_result(result)

    return _call


def _normalize_call_result(result: Any) -> dict[str, Any]:
    """Turn an MCP `CallToolResult` into a JSON-serializable dict.

    - A single text content block whose text parses as JSON → the parsed object.
    - Multiple text content blocks → first parseable JSON, else `{"text": [...]}`.
    - `isError=True` → `{"error": <joined text>}` so `ToolResult.error` paths fire.
    - Mixed content types → `{"text": [...], "image": [...], ...}`.
    """
    is_error = bool(getattr(result, "isError", False))
    content = getattr(result, "content", None) or []

    if not content:
        return {"error": "empty result"} if is_error else {}

    text_blocks: list[str] = []
    other: dict[str, list[Any]] = {}
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_blocks.append(getattr(block, "text", "") or "")
        else:
            other.setdefault(str(btype), []).append(
                # .data is base64; we keep it as-is and let the caller decide.
                getattr(block, "data", None)
                or getattr(block, "text", None)
                or str(block)
            )

    if is_error:
        joined = "\n".join(text_blocks) or json.dumps(other)
        return {"error": joined}

    # If there are non-text blocks alongside text blocks, we must keep both.
    # Single text + others → try to JSON-parse the text and merge with `other`.
    if len(text_blocks) == 1:
        try:
            parsed = json.loads(text_blocks[0])
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and isinstance(parsed, dict):
            return {**parsed, **other}
        base: dict[str, Any] = {"text": text_blocks[0]} if text_blocks[0] else {}
        base.update(other)
        return base

    # Multiple text blocks → try each in order, fall back to a list.
    if len(text_blocks) > 1:
        for block in text_blocks:
            try:
                parsed = json.loads(block)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return {**parsed, **other}
        payload: dict[str, Any] = {"text": text_blocks}
        payload.update(other)
        return payload

    # No text blocks, only other types.
    return other if other else {}


# ---------------------------------------------------------------------------
# JSON Schema → pydantic (loose mode)
# ---------------------------------------------------------------------------


_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_type_for(prop_schema: dict[str, Any]) -> Any:
    """Resolve a JSON Schema property to a Python type for pydantic."""
    if prop_schema.get("enum"):
        # Keep as the declared type, or default to str.
        declared = prop_schema.get("type")
        if isinstance(declared, str) and declared in _TYPE_MAP:
            return _TYPE_MAP[declared]
        return str
    if "anyOf" in prop_schema or "oneOf" in prop_schema:
        # Loose: accept the union as Any. The fetcher prompt is the
        # source of truth on shape.
        return Any
    if "$ref" in prop_schema:
        return Any
    t = prop_schema.get("type")
    if isinstance(t, list):
        # ["string", "null"] — treat as Optional[str] (handled by caller).
        non_null = [x for x in t if x != "null"]
        return _TYPE_MAP.get(non_null[0], Any) if non_null else Any
    if isinstance(t, str):
        return _TYPE_MAP.get(t, Any)
    return Any


def _json_schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a permissive pydantic model from a JSON Schema object."""
    properties: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_type_for(prop_schema)
        if prop_name in required and not _is_nullable(prop_schema):
            fields[prop_name] = (py_type, ...)
        else:
            fields[prop_name] = (py_type | None, None)

    if not fields:
        # MCP servers may ship `{"type": "object"}` with no properties;
        # create an empty model so the executor has something to call.
        return create_model(name)  # type: ignore[call-overload]

    return create_model(name, **fields)  # type: ignore[call-overload]


def _is_nullable(prop_schema: dict[str, Any]) -> bool:
    """Return True if the schema explicitly allows null in addition to a type."""
    t = prop_schema.get("type")
    return bool(isinstance(t, list) and "null" in t)


def build_input_schema_from_json_schema(
    name: str, schema: dict[str, Any]
) -> type[BaseModel]:
    """Public entrypoint: build an input schema from a JSON Schema dict.

    Same loose-mode behavior as `_json_schema_to_model`. Kept as a
    public function so users can prototype schemas independently of an
    MCP session.
    """
    return _json_schema_to_model(name, schema)


# ---------------------------------------------------------------------------
# YAML overrides
# ---------------------------------------------------------------------------


def load_tool_overrides(path: Path) -> dict[str, ToolOverride]:
    """Read a YAML file of description / schema overrides.

    File format::

        defaults:
          category: mcp
          triggers: []

        tools:
          get_weather:
            description: |
              Use when the user asks about current weather...
            triggers: ["current weather", "temperature in"]
            category: weather
          raw_query:
            description: "Run a raw SQL query."
            input_schema: myapp.schemas.SqlQueryInput

    Returns a dict keyed by MCP tool name. `defaults` are merged into
    each entry but do not appear as keys themselves. `input_schema`,
    when given as a string, is resolved as a dotted Python import path
    to a `BaseModel` subclass; when given as a class, used directly.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "load_tool_overrides() requires PyYAML. Install with: "
            "pip install herding-cats[dev]  (PyYAML is in the dev extra)"
        ) from e

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    defaults_raw = raw.get("defaults") or {}
    tools_raw = raw.get("tools") or {}
    if not isinstance(defaults_raw, dict):
        raise ValueError(f"{path}: 'defaults' must be a mapping")
    if not isinstance(tools_raw, dict):
        raise ValueError(f"{path}: 'tools' must be a mapping")

    defaults = _parse_override_entry(path, "<defaults>", defaults_raw)
    allowed_keys = {"description", "input_schema", "category", "triggers"}
    out: dict[str, ToolOverride] = {}
    for tool_name, entry in tools_raw.items():
        if not isinstance(tool_name, str):
            raise ValueError(f"{path}: tool keys must be strings, got {tool_name!r}")
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: entry for tool {tool_name!r} must be a mapping"
            )
        unknown = set(entry) - allowed_keys
        if unknown:
            raise ValueError(
                f"{path}: tool {tool_name!r} has unknown field(s): "
                f"{sorted(unknown)}; allowed: {sorted(allowed_keys)}"
            )
        parsed = _parse_override_entry(path, f"tools.{tool_name}", entry)
        out[tool_name] = _merge_defaults(parsed, defaults)
    return out


def _parse_override_entry(
    path: Path, where: str, entry: dict[str, Any]
) -> ToolOverride:
    description = entry.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{path}: {where}.description must be a string")
    if isinstance(description, str) and not description.strip():
        raise ValueError(f"{path}: {where}.description must be non-empty")

    triggers = entry.get("triggers") or []
    if not isinstance(triggers, list) or not all(
        isinstance(t, str) for t in triggers
    ):
        raise ValueError(f"{path}: {where}.triggers must be a list of strings")

    category = entry.get("category")
    if category is not None and not isinstance(category, str):
        raise ValueError(f"{path}: {where}.category must be a string")

    raw_schema = entry.get("input_schema")
    input_schema: type[BaseModel] | None = None
    if raw_schema is not None:
        if isinstance(raw_schema, str):
            input_schema = _resolve_dotted_model(path, where, raw_schema)
        elif isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel):
            input_schema = raw_schema
        else:
            raise ValueError(
                f"{path}: {where}.input_schema must be a dotted import path "
                f"string or a pydantic BaseModel subclass"
            )

    return ToolOverride(
        description=description,
        input_schema=input_schema,
        category=category,
        triggers=list(triggers),
    )


def _merge_defaults(
    entry: ToolOverride, defaults: ToolOverride
) -> ToolOverride:
    """Apply `defaults` for any field `entry` did not set."""
    return ToolOverride(
        description=entry.description if entry.description is not None else defaults.description,
        input_schema=(
            entry.input_schema if entry.input_schema is not None else defaults.input_schema
        ),
        category=entry.category if entry.category is not None else defaults.category,
        triggers=list(entry.triggers) if entry.triggers else list(defaults.triggers),
    )


def _resolve_dotted_model(
    path: Path, where: str, dotted: str
) -> type[BaseModel]:
    try:
        module_name, _, attr = dotted.rpartition(".")
    except Exception as e:
        raise ValueError(
            f"{path}: {where}.input_schema {dotted!r} is not a valid dotted path"
        ) from e
    if not module_name or not attr:
        raise ValueError(
            f"{path}: {where}.input_schema {dotted!r} must be 'package.module.Class'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(
            f"{path}: {where}.input_schema: could not import {module_name!r}: {e}"
        ) from e
    obj = getattr(module, attr, None)
    if obj is None or not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ValueError(
            f"{path}: {where}.input_schema: {dotted!r} did not resolve to a "
            f"pydantic BaseModel subclass"
        )
    return obj


def apply_tool_overrides(
    specs: list[ToolSpec],
    overrides: dict[str, ToolOverride],
    *,
    source: Path | str | None = None,
) -> list[ToolSpec]:
    """Return a new `list[ToolSpec]` with override fields replaced.

    Iterates `overrides` (not `specs`) so a missing tool produces a
    WARNING via `herding_cats.mcp` and execution continues. See design
    decision #4 in the MCP adapter plan.

    Args:
        specs: Discovered `ToolSpec`s from `discover_tools()`.
        overrides: Mapping of MCP tool name → override entry.
        source: Optional source label for the warning message (typically
            the YAML file path). Defaults to "<overrides>".

    The override is applied per non-None field on the `ToolOverride`.
    Fields the override does not mention are preserved.
    """
    source_label = str(source) if source is not None else "<overrides>"
    known = [s.name for s in specs]

    out: list[ToolSpec] = []
    for spec in specs:
        ov = overrides.get(spec.name)
        if ov is None:
            out.append(spec)
            continue
        updates: dict[str, Any] = {}
        if ov.description is not None:
            updates["description"] = _checked_description(spec.name, ov.description, source_label)
        if ov.input_schema is not None:
            updates["input_schema"] = ov.input_schema
        # `category` and `triggers` are not fields on ToolSpec today;
        # they live in the YAML for users who later build a structured
        # description layer on top. Silently ignored here for back-compat.
        out.append(replace(spec, **updates) if updates else spec)

    # Warn for every override key that matched no spec directly.
    # Also try the namespaced form (e.g. YAML key `get_weather` against
    # spec `fetch_get_weather`) before giving up.
    spec_names = {s.name for s in specs}
    for overridden_name in overrides:
        if overridden_name in spec_names:
            continue
        candidate = None
        for s in specs:
            if s.name.endswith(f"_{overridden_name}"):
                candidate = s.name
                break
        if candidate is not None:
            ov = overrides[overridden_name]
            idx = next(i for i, s in enumerate(out) if s.name == candidate)
            updates = {}
            if ov.description is not None:
                updates["description"] = _checked_description(
                    candidate, ov.description, source_label
                )
            if ov.input_schema is not None:
                updates["input_schema"] = ov.input_schema
            out[idx] = replace(out[idx], **updates) if updates else out[idx]
            continue
        logger.warning(
            "mcp override: tool %r from %s was not found on any connected "
            "MCP server (known: %s)",
            overridden_name,
            source_label,
            known,
        )
    return out


def _checked_description(
    tool_name: str, description: str, source_label: str
) -> str:
    if len(description) > DESCRIPTION_MAX_CHARS:
        logger.warning(
            "mcp override: description for tool %r is %d chars; recommended <= %d",
            tool_name,
            len(description),
            DESCRIPTION_MAX_CHARS,
        )
    return description


# ---------------------------------------------------------------------------
# Crew integration
# ---------------------------------------------------------------------------


class crew_with_mcp:
    """Async context manager: open MCP servers, return an augmented Crew.

    Usage::

        async with crew_with_mcp(crew, servers=[...]) as ctx:
            runner = CrewRunner(ctx.crew)
            answer = await runner.arun("...")

    On `__aenter__`:

    1. Open all servers via `connect_mcp_servers`.
    2. Discover tools on each session (optionally with a namespace).
    3. Load overrides from each `McpServerSpec.overrides_path` and apply.
    4. Return a `_McpBoundCrew` with `crew.tool_specs` extended by the
       discovered + overridden specs.

    On `__aexit__`: closes all transports (sessions, then stdio pipes /
    SSE streams) regardless of whether the body raised.

    If `servers` is empty, the context manager is a no-op: returns
    the original `Crew` and an empty session list.
    """

    def __init__(
        self,
        crew: Crew,
        *,
        servers: list[McpServerSpec] | None = None,
    ) -> None:
        self._crew = crew
        self._servers = list(servers or [])
        self._conn: connect_mcp_servers | None = None

    async def __aenter__(self) -> _McpBoundCrew:
        if not self._servers:
            return _McpBoundCrew(crew=self._crew, sessions=[])

        self._conn = connect_mcp_servers(self._servers)
        sessions = await self._conn.__aenter__()
        try:
            extra_specs: list[ToolSpec] = []
            for spec, session in zip(self._servers, sessions, strict=False):
                discovered = await discover_tools(session, namespace=spec.namespace)
                if spec.overrides_path is not None:
                    overrides = load_tool_overrides(spec.overrides_path)
                else:
                    overrides = {}
                if overrides:
                    discovered = apply_tool_overrides(
                        discovered,
                        overrides,
                        source=spec.overrides_path or spec.namespace or spec.command or spec.url,
                    )
                extra_specs.extend(discovered)

            # Cross-server collision check (when neither is namespaced).
            names = [s.name for s in extra_specs]
            dups = {n for n in names if names.count(n) > 1}
            if dups:
                raise ValueError(
                    "Duplicate tool names across MCP servers without namespace "
                    f"disambiguation: {sorted(dups)}. Set a `namespace=` on "
                    "the conflicting McpServerSpec(s)."
                )

            new_crew = replace(
                self._crew,
                tool_specs=list(self._crew.tool_specs) + extra_specs,
            )
            return _McpBoundCrew(crew=new_crew, sessions=list(sessions))
        except BaseException:
            await self._conn.__aexit__(type(self._conn), None, None)
            self._conn = None
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            await self._conn.__aexit__(exc_type, exc, tb)
            self._conn = None


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "DESCRIPTION_MAX_CHARS",
    "McpServerSpec",
    "ToolOverride",
    "apply_tool_overrides",
    "build_input_schema_from_json_schema",
    "connect_mcp_servers",
    "crew_with_mcp",
    "discover_tools",
    "load_tool_overrides",
]
