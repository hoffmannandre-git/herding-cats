"""CrewRunner: drives one user question through the full crew loop.

Public surface:

* `Crew(...)` — configuration: ollama, tools, prompts, budgets.
* `CrewRunner(crew).run(question)` — single-turn sync-ish entrypoint.
* `CrewRunner(crew).run_stream(question)` — yields `RunnerEvent` for UIs.
* `CrewRunner(crew).step(state)` — single orchestrator decision + one
  downstream agent; returns `(decision, new_state, event)`. This is the
  hook for per-step steering (UI buttons, HTTP step endpoints, etc).

`run` is synchronous because every HTTP call inside the loop is quick
(we're talking to a local daemon). If you want async-native, call
`await runner.arun(question)` from an event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from herding_cats.crew.executor import Executor
from herding_cats.crew.fetcher import run_fetcher
from herding_cats.crew.finalist import run_direct, run_finalist
from herding_cats.crew.orchestrator import (
    MAX_FETCH_TURNS,
    MAX_THINK_TURNS,
    ORCHESTRATOR_MODE,
    SEARCHES_PER_FETCH,
    OrchestratorDecision,
    compute_budget,
    decide_llm,
    decide_rules,
    looks_like_smalltalk,
    pipeline_status,
)
from herding_cats.crew.state import CrewState
from herding_cats.crew.thinker import run_thinker
from herding_cats.crew.tools import ToolRegistry, ToolSpec
from herding_cats.helpers.env import get_env
from herding_cats.helpers.prompts import load_prompt, render
from herding_cats.ollama import OllamaClient

logger = logging.getLogger("herding_cats.crew.runner")

DEFAULT_MODEL = get_env("MODEL") or "llama3.1:8b"
DEFAULT_PROMPTS_DIR: Path | None = None  # resolved lazily


def _prompts_dir() -> Path:
    global DEFAULT_PROMPTS_DIR
    if DEFAULT_PROMPTS_DIR is None:
        from herding_cats.helpers.prompts import default_prompts_dir

        DEFAULT_PROMPTS_DIR = default_prompts_dir()
    return DEFAULT_PROMPTS_DIR


@dataclass
class Crew:
    """Configuration for a crew. Reusable across questions.

    Attributes:
        ollama: The Ollama client. Required.
        tools: A mapping of tool name → callable. Optional.
        prompts: Per-role prompt overrides (raw strings).
        prompts_dir: Override the prompts directory.
        model: Model name for all roles.
        max_loop, max_fetch, max_think, searches_per_fetch: Budget overrides.
        orchestrator_mode: "rules" (default) or "llm".
        short_circuit_smalltalk: If True (default), smalltalk goes straight
            to `final` via the direct prompt, skipping the crew.
    """

    ollama: OllamaClient
    tools: dict[str, Any] = field(default_factory=dict)
    # Pre-typed tool specs (with pydantic schemas). When set, these override
    # any plain-function entry in `tools` with the same name.
    tool_specs: list[Any] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)
    prompts_dir: Path | None = None
    model: str = DEFAULT_MODEL
    max_loop: int = int(get_env("MAX_LOOP") or "12")
    max_fetch: int = MAX_FETCH_TURNS
    max_think: int = MAX_THINK_TURNS
    searches_per_fetch: int = SEARCHES_PER_FETCH
    orchestrator_mode: Literal["rules", "llm"] = ORCHESTRATOR_MODE  # type: ignore[assignment]
    short_circuit_smalltalk: bool = True
    shortcuts: dict[str, Any] = field(default_factory=dict)

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        # Schema-bearing specs first (highest fidelity prompt rendering).
        for spec in self.tool_specs:
            if isinstance(spec, ToolSpec):
                reg.specs.append(spec)
            else:
                # Treat unknown types as a plain tool for safety.
                reg.register(name=getattr(spec, "name", "?"), fn=spec)
        # Plain function fallbacks; skip names already present.
        existing = {s.name for s in reg.specs}
        for name, fn in self.tools.items():
            if name in existing:
                continue
            reg.register(name, fn)
        return reg


@dataclass
class RunnerEvent:
    kind: Literal[
        "orchestrator_decision",
        "think",
        "fetch_plan",
        "tool_call",
        "tool_result",
        "final",
        "warning",
        "trace",
        "stopped",
    ]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of one `CrewRunner.step()` call.

    `state` is the new (possibly mutated) state — pass it back to the next
    `step()` call to continue the loop. `done` is True when the runner
    will return `final` or `stop` on the next call; the caller can stop
    then.
    """

    decision: OrchestratorDecision
    state: CrewState
    event: RunnerEvent | None = None
    done: bool = False


class CrewRunner:
    """Drives one user question through the orchestrator → thinker → fetcher → finalist loop."""

    def __init__(self, crew: Crew) -> None:
        self.crew = crew
        self._registry = crew.registry()
        self._executor = Executor(self._registry)

    # ----- public entrypoints ----------------------------------------------

    def run(self, question: str) -> str:
        return asyncio.run(self.arun(question))

    async def arun(self, question: str) -> str:
        state = self._initial_state(question)
        async for _event in self.arun_stream(question, state=state):
            pass
        return state.current.answer

    def run_stream(
        self, question: str, *, state: CrewState | None = None
    ) -> Iterator[RunnerEvent]:
        async def _drain():
            return [ev async for ev in self.arun_stream(question, state=state)]

        events = asyncio.run(_drain())
        yield from events

    async def arun_stream(
        self, question: str, *, state: CrewState | None = None
    ) -> AsyncIterator[RunnerEvent]:
        state = state or self._initial_state(question)

        # Shortcut path (Gap 5): registered patterns bypass the crew.
        shortcut_hit = await self._try_shortcut(question, state)
        if shortcut_hit is not None:
            yield shortcut_hit
            yield RunnerEvent("final", {"answer": state.current.answer})
            return

        # Hard cap on the loop, enforced here.
        while state.round_number < self.crew.max_loop:
            result = await self.astep(state)
            if result.event is not None:
                yield result.event
            state = result.state
            if result.done:
                return

        # Hard cap reached.
        state.append_trace(f"runner: max_loop={self.crew.max_loop} reached; forcing final")
        state.current.warnings.append("runner: max_loop reached without final; forcing answer")
        answer = await self._do_final(state, self._prompts())
        state.current.answer = answer
        yield RunnerEvent("warning", {"message": "max_loop reached; forcing final"})
        yield RunnerEvent("trace", {"line": state.trace[-1]})
        yield RunnerEvent("final", {"answer": answer})

    # ----- per-step steering (Gap 1) --------------------------------------

    def step(self, state: CrewState) -> StepResult:
        """One orchestrator decision + one downstream agent.

        Returns a `StepResult` carrying the decision, the new `state`
        (mutated), the corresponding event (so callers that don't want
        the streaming API can still inspect what happened), and a `done`
        flag indicating the crew is finished after this step.
        """
        return asyncio.run(self.astep(state))

    async def astep(self, state: CrewState) -> StepResult:
        return await self._astep(state)

    async def _astep(self, state: CrewState) -> StepResult:
        if state.round_number >= self.crew.max_loop:
            state.append_trace("astep: max_loop reached before orchestrator ran")
            state.current.warnings.append("runner: max_loop already reached")
            answer = await self._do_final(state, self._prompts())
            state.current.answer = answer
            return StepResult(
                decision=OrchestratorDecision(next="stop", reason="max_loop", mode="rules"),
                state=state,
                event=RunnerEvent("warning", {"message": "max_loop reached"}),
                done=True,
            )

        prompts = self._prompts()
        decision = self._orchestrate(state, prompts["orchestrator"])
        state.append_trace(
            f"orchestrator → {decision.next} ({decision.reason})"
        )

        if decision.next == "stop":
            return StepResult(
                decision=decision,
                state=state,
                event=RunnerEvent("stopped", {"reason": decision.reason}),
                done=True,
            )

        if decision.next == "final":
            answer = await self._do_final(state, prompts)
            state.current.answer = answer
            return StepResult(
                decision=decision,
                state=state,
                event=RunnerEvent("final", {"answer": answer}),
                done=True,
            )

        if decision.next == "think":
            result = await self._do_think(state, prompts)
            state.last_role = "think"
            # If the orchestrator asked for reflection, mark it done so
            # the next fetch → think cycle goes straight to final.
            if (
                state.fetch_used > 0
                and state.current.hits
                and not state.reflection_think_done
            ):
                state.reflection_think_done = True
                state.append_trace("think: reflection pass complete")
            event = RunnerEvent(
                "think",
                {"notes": result.notes, "search_terms": result.search_terms},
            )
            return StepResult(decision=decision, state=state, event=event, done=False)

        if decision.next == "fetch":
            event = await self._do_fetch(state, prompts)
            state.last_role = "fetch"
            return StepResult(decision=decision, state=state, event=event, done=False)

        # Unknown decision — treat as stop.
        return StepResult(
            decision=decision,
            state=state,
            event=RunnerEvent("warning", {"message": f"unknown decision {decision.next!r}"}),
            done=True,
        )

    # ----- internal step implementations -----------------------------------

    def step_sync(self, state: CrewState) -> StepResult:
        """Synchronous wrapper for callers that don't want an event loop."""
        return asyncio.run(self.astep(state))

    async def _do_think(self, state: CrewState, prompts: dict[str, str]) -> Any:
        # Pop the next unconsumed search term off the queue, if any.
        suggested_term = _pop_search_term(state)
        # Summarize prior turns (last 5) so the model can resolve
        # follow-up questions without us dumping the full transcript.
        prior_summary = [
            {
                "question": t.notes[:120] if t.notes else "",
                "answer": t.answer[:200] if t.answer else "",
            }
            for t in state.prior_turns[-5:]
        ]
        prompt = render(
            prompts["thinker"],
            question=state.question,
            prior_turns=prior_summary,
            prior_context=state.prior_context or "",
            notes=state.current.notes,
            search_terms=state.current.search_terms,
            hits=state.current.hits,
            seen_chunk_ids=state.seen_chunk_ids,
            suggested_term=suggested_term or "",
        )
        result = await run_thinker(
            ollama=self.crew.ollama, state=state, prompt=prompt, model=self.crew.model
        )
        state.current.notes = result.notes
        # Extend, don't replace — the fetcher will pop from the front.
        state.current.search_terms.extend(result.search_terms)
        state.think_used += 1
        state.round_number += 1
        state.append_trace(
            f"think #{state.think_used}: notes={len(result.notes)}c "
            f"new_terms={len(result.search_terms)}"
        )
        return result

    async def _do_fetch(
        self, state: CrewState, prompts: dict[str, str]
    ) -> RunnerEvent:
        suggested_term = _pop_search_term(state)
        prompt = render(
            prompts["fetcher"],
            question=state.question,
            prior_turns=[
                {"answer": t.answer[:200]} for t in state.prior_turns[-3:]
            ],
            notes=state.current.notes,
            search_terms=state.current.search_terms,
            hits=state.current.hits,
            seen_chunk_ids=state.seen_chunk_ids,
            tools=self._registry.describe(),
            suggested_term=suggested_term or "",
        )
        result = await run_fetcher(
            ollama=self.crew.ollama, state=state, prompt=prompt, model=self.crew.model
        )

        # Dedup retry (Gap 4): if the fetcher emitted a query that's already
        # been executed this turn, silently retry with the next queued term.
        retry_attempts = 0
        while (
            result.call is not None
            and _extract_query(result.call) in _executed_queries(state)
            and suggested_term is None  # we already popped; need next one
        ):
            suggested_term = _pop_search_term(state)
            if not suggested_term:
                break
            state.append_trace(
                f"fetcher: dup query {_extract_query(result.call)!r}; "
                f"retrying with {suggested_term!r}"
            )
            retry_prompt = render(
                prompts["fetcher"],
                question=state.question,
                prior_turns=[
                    {"answer": t.answer[:200]} for t in state.prior_turns[-3:]
                ],
                notes=state.current.notes,
                search_terms=state.current.search_terms,
                hits=state.current.hits,
                seen_chunk_ids=state.seen_chunk_ids,
                tools=self._registry.describe(),
                suggested_term=suggested_term,
            )
            result = await run_fetcher(
                ollama=self.crew.ollama,
                state=state,
                prompt=retry_prompt,
                model=self.crew.model,
            )
            retry_attempts += 1
            if retry_attempts >= 3:
                break

        state.round_number += 1
        state.fetch_used += 1

        if result.call is None:
            state.current.warnings.append(f"fetcher skipped: {result.reason}")
            state.append_trace(f"fetch #{state.fetch_used}: skipped ({result.reason})")
            return RunnerEvent(
                "warning", {"message": f"fetcher skipped: {result.reason}"}
            )

        # Record the executed query for future dedup.
        q = _extract_query(result.call)
        if q and q not in _executed_queries(state):
            state.queries_executed.append(q)

        tool_result = await self._executor.arun(result.call)
        # Cross-turn memory: collect chunk IDs returned by the tool.
        _record_chunk_ids(state, tool_result.output)
        state.current.hits.append(
            {
                "tool": tool_result.tool,
                "input": tool_result.input,
                "output": tool_result.output,
                "error": tool_result.error,
            }
        )
        state.append_trace(
            f"fetch #{state.fetch_used}: {tool_result.tool}({tool_result.input})"
            + (f" error={tool_result.error}" if tool_result.error else "")
        )
        # Yield both events: the plan and the result. Stream consumers get both.
        return RunnerEvent(
            "tool_result",
            {
                "tool": tool_result.tool,
                "input": tool_result.input,
                "output": tool_result.output,
                "error": tool_result.error,
                "plan": {"tool": result.call.tool, "input": result.call.input},
            },
        )

    async def _do_final(self, state: CrewState, prompts: dict[str, str]) -> str:
        # Public smalltalk hook (Gap 9): skip the crew when configured + matched.
        if self.crew.short_circuit_smalltalk and looks_like_smalltalk(state.question):
            state.append_trace("final: smalltalk short-circuit")
            prompt = render(prompts["direct"], question=state.question)
            return await run_direct(
                ollama=self.crew.ollama,
                question=state.question,
                prompt=prompt,
                model=self.crew.model,
            )

        prompt = render(
            prompts["finalist"],
            question=state.question,
            prior_turns=[
                {"answer": t.answer[:200]} for t in state.prior_turns[-5:]
            ],
            notes=state.current.notes,
            hits=state.current.hits,
        )
        return await run_finalist(
            ollama=self.crew.ollama, state=state, prompt=prompt, model=self.crew.model
        )

    async def _try_shortcut(
        self, question: str, state: CrewState
    ) -> RunnerEvent | None:
        """Apply registered shortcuts before the crew runs.

        A shortcut is `{pattern: handler}` where `pattern` is a regex string
        and `handler(input: dict) -> str` is a function that produces the
        final answer without any LLM call. Use this to fast-path known
        query shapes ("list all contacts", "what's my status", etc.).

        Returns the `RunnerEvent` for the matched shortcut, or `None` when
        no pattern matches. The caller is expected to consume the event
        and end the loop.
        """
        if not self.crew.shortcuts:
            return None
        import re

        for pattern, handler in self.crew.shortcuts.items():
            if re.search(pattern, question, flags=re.IGNORECASE):
                state.append_trace(
                    f"shortcut: matched pattern {pattern!r}"
                )
                try:
                    answer = handler({"question": question})
                    state.current.answer = str(answer)
                except Exception as e:
                    state.current.warnings.append(f"shortcut {pattern!r} failed: {e}")
                    state.append_trace(
                        f"shortcut: handler raised {e!r}; falling through"
                    )
                    return None
                return RunnerEvent(
                    "tool_result",
                    {
                        "tool": f"shortcut:{pattern}",
                        "input": {"question": question},
                        "output": state.current.answer,
                    },
                )
        return None

    # ----- internals --------------------------------------------------------

    def _initial_state(self, question: str) -> CrewState:
        state = CrewState(
            question=question,
            tools=self._registry.describe(),
            max_rounds=self.crew.max_loop,
            fetch_max=self.crew.max_fetch,
            think_max=self.crew.max_think,
            searches_per_fetch=self.crew.searches_per_fetch,
        )
        state.append_trace(f"crew started for question={question!r}")
        return state

    def _prompts(self) -> dict[str, str]:
        base = self.crew.prompts_dir or _prompts_dir()
        roles = ("orchestrator", "thinker", "fetcher", "finalist", "direct")
        out: dict[str, str] = {}
        for role in roles:
            if role in self.crew.prompts:
                out[role] = self.crew.prompts[role]
            else:
                out[role] = load_prompt(role, base_dir=base)
        return out

    def _orchestrate(self, state: CrewState, prompt_tpl: str) -> OrchestratorDecision:
        # Build a state-summary block the LLM-mode orchestrator can read.
        budget = compute_budget(state)
        user_prompt = render(
            prompt_tpl,
            question=state.question,
            round=state.round_number,
            think_used=state.think_used,
            fetch_used=state.fetch_used,
            hits_count=len(state.current.hits),
            reflection_done=state.reflection_think_done,
            has_hits=bool(state.current.hits),
            budget=budget.summary_line(),
            should_finish=budget.should_finish,
            finish_hint=budget.finish_hint,
        )

        if self.crew.orchestrator_mode == "rules":
            return decide_rules(state)

        # LLM mode: split into system (role + rules) and user (state summary).
        # The system prompt is the original prompt_tpl unchanged;
        # the user prompt gets the rendered state summary.
        return asyncio.run(
            _call_llm_decide_sync(
                system_prompt=prompt_tpl,
                user_prompt=user_prompt,
                ollama=self.crew.ollama,
                state=state,
            )
        )


# ----- helpers ---------------------------------------------------------------


def _pop_search_term(state: CrewState) -> str | None:
    """Return the next unconsumed search term, advancing the index."""
    idx = state.current.search_term_index
    terms = state.current.search_terms
    while idx < len(terms):
        term = terms[idx].strip()
        idx += 1
        state.current.search_term_index = idx
        if term:
            return term
    return None


def _executed_queries(state: CrewState) -> set[str]:
    return {q.lower() for q in state.queries_executed}


def _extract_query(call) -> str | None:
    """Pull the 'query' string out of a fetcher tool call, if present."""
    if call is None:
        return None
    q = call.input.get("query")
    return str(q).lower() if q else None


def _record_chunk_ids(state: CrewState, output: object) -> None:
    """Walk a tool's output and pull out any `id` / `chunk_id` fields.

    Adds them to `state.seen_chunk_ids` so the fetcher can avoid asking
    for the same chunk again in a later turn. Works on dicts, lists of
    dicts, and nested structures (one level deep — most retrieval
    tools return flat lists).
    """
    if output is None:
        return
    seen = state.seen_chunk_ids
    if isinstance(output, dict):
        for k in ("id", "chunk_id", "doc_id"):
            v = output.get(k)
            if isinstance(v, (str, int)) and str(v) not in seen:
                seen.append(str(v))
    elif isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                for k in ("id", "chunk_id", "doc_id"):
                    v = item.get(k)
                    if isinstance(v, (str, int)) and str(v) not in seen:
                        seen.append(str(v))


async def _call_llm_decide_sync(
    *, system_prompt: str, user_prompt: str, ollama: OllamaClient, state: CrewState
) -> OrchestratorDecision:
    """Async LLM-mode orchestrator decide — kept as a free function so the
    runner doesn't have to manage an event loop inside `_orchestrate`."""
    return decide_llm(
        state, system_prompt=system_prompt, user_prompt=user_prompt, llm_caller=ollama.achat
    )


# Public re-exports for the API surface.
__all__ = [
    "Crew",
    "CrewRunner",
    "RunnerEvent",
    "StepResult",
    "pipeline_status",
]
