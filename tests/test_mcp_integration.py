"""Slow end-to-end integration tests for the MCP adapter.

These spawn a real in-process MCP server (built with `mcp.server.fastmcp.FastMCP`)
and connect to it over stdio. No Docker, no external software required
beyond the `mcp` Python SDK itself.

Marked `@pytest.mark.slow`. Skipped if `mcp` is not installed.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

try:
    from exceptiongroup import BaseExceptionGroup as _BaseExceptionGroup
except ImportError:
    _BaseExceptionGroup = getattr(builtins, "BaseExceptionGroup", RuntimeError)


pytest.importorskip("mcp")
pytestmark = pytest.mark.slow

from herding_cats.crew.runner import Crew  # noqa: E402
from herding_cats.mcp import (  # noqa: E402
    McpServerSpec,
    apply_tool_overrides,
    connect_mcp_servers,
    crew_with_mcp,
    discover_tools,
    load_tool_overrides,
)
from herding_cats.ollama import OllamaClient  # noqa: E402

# ----- in-process MCP server definition -----------------------------------
#
# Written as a separate string so we can spawn it as a fresh subprocess
# via the stdio transport without dragging the FastMCP object across a
# process boundary.

_ECHO_SERVER_SOURCE = """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool()
def echo(text: str) -> str:
    \"\"\"Echo back the input text.\"\"\"
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b


if __name__ == "__main__":
    mcp.run()
"""


@pytest.fixture(scope="module")
def echo_server_script(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("mcp") / "echo_server.py"
    p.write_text(_ECHO_SERVER_SOURCE)
    return p


# ----- connect + discover via real stdio transport ------------------------


@pytest.mark.asyncio
async def test_connect_discover_echo_server(echo_server_script: Path) -> None:
    spec = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
    )
    async with connect_mcp_servers([spec]) as sessions:
        assert len(sessions) == 1
        specs = await discover_tools(sessions[0])
        names = {s.name for s in specs}
        assert "echo" in names
        assert "add" in names


@pytest.mark.asyncio
async def test_tool_call_round_trip(echo_server_script: Path) -> None:
    spec = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
    )
    async with connect_mcp_servers([spec]) as sessions:
        specs = await discover_tools(sessions[0])
        echo_spec = next(s for s in specs if s.name == "echo")
        result = await echo_spec.fn({"text": "hello"})
        assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_session_exception_inside_body_does_not_hang(
    echo_server_script: Path,
) -> None:
    """If the body of `async with connect_mcp_servers(...)` raises, the
    underlying stdio transport must still be cleaned up. We assert this
    by causing an exception inside the block — the cleanup must
    complete (the context manager exits) and the original error should
    propagate (possibly wrapped by anyio's task-group machinery; both
    shapes are acceptable cleanup evidence)."""
    spec = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
    )
    with pytest.raises((RuntimeError, _BaseExceptionGroup)):
        async with connect_mcp_servers([spec]) as _sessions:
            raise RuntimeError("boom")
    # If we got here, the contexts were cleaned up cleanly enough for
    # Python to surface the original error (even if wrapped).


# ----- override applied end-to-end -----------------------------------------


@pytest.mark.asyncio
async def test_override_replaces_description_in_spec(
    echo_server_script: Path, tmp_path: Path
) -> None:
    spec = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
        overrides_path=tmp_path / "overrides.yaml",
    )
    overrides_yaml = tmp_path / "overrides.yaml"
    overrides_yaml.write_text(
        """
tools:
  echo:
    description: "Use this when the user asks for text echo."
""",
        encoding="utf-8",
    )

    async with connect_mcp_servers([spec]) as sessions:
        discovered = await discover_tools(sessions[0])
        # Sanity: the original MCP description is something else.
        echo_before = next(s for s in discovered if s.name == "echo")
        assert "echo" in echo_before.description.lower()

        overrides = load_tool_overrides(overrides_yaml)
        out = apply_tool_overrides(discovered, overrides, source=overrides_yaml)
        echo_after = next(s for s in out if s.name == "echo")
        assert echo_after.description == "Use this when the user asks for text echo."


# ----- crew_with_mcp end-to-end ------------------------------------------


@pytest.mark.asyncio
async def test_crew_with_mcp_extends_tool_specs(
    echo_server_script: Path, tmp_path: Path
) -> None:
    overrides_path = tmp_path / "overrides.yaml"
    overrides_path.write_text(
        """
tools:
  echo:
    description: "Better echo description."
""",
        encoding="utf-8",
    )
    spec = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
        namespace="echo",
        overrides_path=overrides_path,
    )
    crew = Crew(ollama=OllamaClient())
    async with crew_with_mcp(crew, servers=[spec]) as ctx:
        assert ctx.crew is not crew  # new crew instance
        assert len(ctx.sessions) == 1
        # The augmented crew's tool_specs must contain the namespaced echo.
        names = {s.name for s in ctx.crew.tool_specs}
        assert "echo_echo" in names
        # With the override applied.
        namespaced_echo = next(
            s for s in ctx.crew.tool_specs if s.name == "echo_echo"
        )
        assert namespaced_echo.description == "Better echo description."


@pytest.mark.asyncio
async def test_crew_with_mcp_collides_when_unnamespaced(
    echo_server_script: Path,
) -> None:
    """Two servers with the same tool, neither namespaced → raises."""
    spec_a = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
    )
    spec_b = McpServerSpec(
        transport="stdio",
        command=sys.executable,
        args=[str(echo_server_script)],
    )
    crew = Crew(ollama=OllamaClient())
    with pytest.raises(ValueError, match="Duplicate tool names"):
        async with crew_with_mcp(crew, servers=[spec_a, spec_b]) as _ctx:
            pass


@pytest.mark.asyncio
async def test_crew_with_mcp_no_servers(echo_server_script: Path) -> None:
    """Empty servers list → no-op, returns the original crew."""
    crew = Crew(ollama=OllamaClient())
    async with crew_with_mcp(crew, servers=[]) as ctx:
        assert ctx.crew is crew
        assert ctx.sessions == []
