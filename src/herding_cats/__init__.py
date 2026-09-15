"""herding-cats — multi-agent LLM crew runtime backed by a local Ollama instance.

(Like herding cats: a small cast of opinionated, mostly-quirky agents
that you point at a problem and hope ends up pointing back at an
answer. The "cats" here are the orchestrator, thinker, fetcher, and
finalist; the "herding" is the budget-aware loop that keeps them
out of each other's food bowls.)

The public surface is intentionally small: an OllamaClient, a Crew
configuration object, a CrewRunner that drives one user query
through the orchestrator → thinker → fetcher → finalist loop, and
serializable state types so the loop can be steered from outside.

Example:
    >>> from herding_cats import Crew, CrewRunner, OllamaClient
    >>> crew = Crew(ollama=OllamaClient(), tools={"echo": lambda i: i})
    >>> answer = CrewRunner(crew).run("hello")
"""

from herding_cats.crew.executor import Executor
from herding_cats.crew.orchestrator import (
    OrchestratorBudget,
    OrchestratorDecision,
    compute_budget,
    decide_rules,
    looks_like_smalltalk,
    pipeline_status,
)
from herding_cats.crew.runner import Crew, CrewRunner, RunnerEvent, StepResult
from herding_cats.crew.state import CrewState, NextAgent, TurnMemory
from herding_cats.crew.tools import ToolCall, ToolRegistry, ToolResult, ToolSpec

# Optional MCP integration. Importing succeeds even without the `mcp`
# extra installed; the symbols are bound and raise on first use.
from herding_cats.mcp import (
    McpServerSpec,
    ToolOverride,
    apply_tool_overrides,
    build_input_schema_from_json_schema,
    connect_mcp_servers,
    crew_with_mcp,
    discover_tools,
    load_tool_overrides,
)
from herding_cats.ollama import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    OllamaClient,
    OllamaError,
)
from herding_cats.session import (
    CrewSession,
    InMemorySession,
    JsonFileSession,
    SessionStore,
    TurnRecord,
)
from herding_cats.tools_filesystem import default_data_dir, filesystem_tools

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Crew",
    "CrewRunner",
    "CrewSession",
    "CrewState",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "Executor",
    "InMemorySession",
    "JsonFileSession",
    "McpServerSpec",
    "NextAgent",
    "OllamaClient",
    "OllamaError",
    "OrchestratorBudget",
    "OrchestratorDecision",
    "RunnerEvent",
    "SessionStore",
    "StepResult",
    "ToolCall",
    "ToolOverride",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TurnMemory",
    "TurnRecord",
    "apply_tool_overrides",
    "build_input_schema_from_json_schema",
    "compute_budget",
    "connect_mcp_servers",
    "crew_with_mcp",
    "decide_rules",
    "default_data_dir",
    "discover_tools",
    "filesystem_tools",
    "load_tool_overrides",
    "looks_like_smalltalk",
    "pipeline_status",
]

__version__ = "0.3.0"
