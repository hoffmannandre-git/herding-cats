"""State types shared across crew roles.

All public state types are pydantic `BaseModel`s so a `CrewState` can be
serialized to JSON, sent across an HTTP boundary, mutated by a UI, and
handed back to `CrewRunner.step(state)` to resume the loop. That
round-trip is the basis of the per-step steering API.

A `CrewState` is everything a single user query needs to be answered.
A `TurnMemory` is what gets handed to the next role inside one crew turn.

These are plain dataclasses — the runner wires them up and the roles
read from / append to them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

NextAgent = Literal["think", "fetch", "final", "stop"]


class TurnMemory(BaseModel):
    """Short-lived state that travels across roles in one crew turn."""

    notes: str = ""
    search_terms: list[str] = Field(default_factory=list)
    search_term_index: int = 0
    hits: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: Filled in by the runner when the crew produces a final answer;
    #: lets sync `arun` return the answer without an extra event channel.
    answer: str = ""


class CrewState(BaseModel):
    """Top-level state for a single user query.

    Serializable: every field is a pydantic type, so the state can be
    JSON-encoded and round-tripped through `CrewRunner.step(state)` to
    resume a loop from outside the runtime (HTTP, websocket, CLI).

    Round-trip cost: every mutation creates a new model instance. The
    runner does not depend on object identity, only on field values.
    """

    # Inputs (set by the caller, never mutated by the runner).
    question: str
    prior_turns: list[TurnMemory] = Field(default_factory=list)
    prior_context: str | None = None
    seen_chunk_ids: list[str] = Field(default_factory=list)
    # Free-form tool descriptors from `ToolRegistry.describe()`. May
    # contain nested JSON-schema objects (input/output), so we don't
    # constrain this to `dict[str, str]`.
    tools: list[dict[str, Any]] = Field(default_factory=list)

    # Per-turn working memory.
    current: TurnMemory = Field(default_factory=TurnMemory)

    # Round bookkeeping (filled in by the runner, not by the role code).
    round_number: int = 0
    max_rounds: int = 12
    fetch_used: int = 0
    fetch_max: int = 3
    think_used: int = 0
    think_max: int = 4
    searches_per_fetch: int = 2

    # Which role ran last in the current turn — used by the rules-mode
    # orchestrator to decide what comes next. `None` at turn start.
    last_role: str | None = None

    # Reflection control: when set, the runner forces a second think pass
    # after a fetch returned hits, and clears the flag. Prevents infinite
    # think → fetch → think ping-pong.
    reflection_think_done: bool = False

    # Dedup: queries already executed by the fetcher this turn, used to
    # silently retry with the next search term if the model loops.
    queries_executed: list[str] = Field(default_factory=list)

    # Step log: human-readable trace of what the runner did. Append-only
    # within a turn. Useful for the UI and for debugging stuck loops.
    trace: list[str] = Field(default_factory=list)

    def append_trace(self, line: str) -> None:
        self.trace.append(line)
