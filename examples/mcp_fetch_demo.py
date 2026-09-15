"""End-to-end demo: herding-cats driving the `mcp-server-fetch` MCP server.

Run locally:

    uvx mcp-server-fetch &
    PYTHONPATH=src python examples/mcp_fetch_demo.py

Or via Docker Compose (after the `mcp` profile is added):

    docker compose --profile mcp up -d
    docker compose run --rm herding-cats python examples/mcp_fetch_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.mcp import McpServerSpec, crew_with_mcp


async def main() -> int:
    overrides = Path(__file__).with_name("mcp_tool_overrides.yaml")
    if not overrides.exists():
        print(f"missing: {overrides}", file=sys.stderr)
        return 1

    servers = [
        McpServerSpec(
            transport="stdio",
            # `uvx` is the standard launcher for the official MCP servers.
            # Override with `command=` to point at a different binary.
            command="uvx",
            args=["mcp-server-fetch"],
            namespace="fetch",
            overrides_path=overrides,
        ),
    ]

    crew = Crew(ollama=OllamaClient())
    async with crew_with_mcp(crew, servers=servers) as ctx:
        runner = CrewRunner(ctx.crew)
        answer = await runner.arun(
            "Fetch the README of the herding-cats repo on GitHub "
            "(https://raw.githubusercontent.com/hoffmannandre-git/"
            "herding-cats/main/README.md) and summarize the architecture."
        )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
