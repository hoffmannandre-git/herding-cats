"""Deterministic tool executor.

The executor is the *only* crew component that does not call the LLM.
It receives a `ToolCall` from the fetcher, looks up the callable, runs it
(sync or async), and packages the result.

Why deterministic? Because every other failure mode in the crew is a
language-model failure mode. The executor is the one place we can be
unambiguous: the tool either ran or it didn't, the result is what it is.

Schema validation: when a tool has an `input_schema` (pydantic model),
the executor validates `call.input` before invoking the callable. When
it has an `output_schema`, the executor validates the result before
returning. Failures become `validation_error` on the `ToolResult` so
the fetcher can retry with corrected input on the next round.

Output normalization: if a tool returns a pydantic `BaseModel`, the
executor calls `.model_dump()` before storing it on `ToolResult.output`
so consumers always see a dict (or list, or scalar), never a model
object. This makes downstream processing (JSON serialization, prompt
rendering, state.record_chunk_ids) work uniformly.
"""

from __future__ import annotations

import asyncio
import inspect

from pydantic import BaseModel, ValidationError

from herding_cats.crew.tools import ToolCall, ToolRegistry, ToolResult


def _maybe_dump(value: object) -> object:
    """Convert a pydantic BaseModel to a dict; pass everything else through."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


class Executor:
    """Runs tool calls against a `ToolRegistry`.

    `safe` mode wraps user exceptions into a `ToolResult.error` string so
    a broken tool never crashes the crew loop. Set `safe=False` in tests
    when you want exceptions to propagate.
    """

    def __init__(self, registry: ToolRegistry, *, safe: bool = True) -> None:
        self.registry = registry
        self.safe = safe

    def _maybe_build_input(self, spec, call: ToolCall) -> object:
        """If the tool has an `input_schema`, validate the dict into a
        model instance. Otherwise return the dict unchanged."""
        if spec.input_schema is None:
            return call.input
        try:
            return spec.input_schema.model_validate(call.input)
        except ValidationError as e:
            raise ValueError(f"input_schema: {e}") from e

    def _validate_input(self, spec, call: ToolCall) -> str | None:
        if spec.input_schema is None:
            return None
        try:
            spec.input_schema.model_validate(call.input)
            return None
        except ValidationError as e:
            return f"input_schema: {e}"

    def _validate_output(self, spec, value: object) -> str | None:
        if spec.output_schema is None:
            return None
        try:
            spec.output_schema.model_validate(value)
            return None
        except ValidationError as e:
            return f"output_schema: {e}"

    def run(self, call: ToolCall) -> ToolResult:
        spec = self.registry.by_name(call.tool)
        if spec is None:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                error=f"unknown tool: {call.tool}",
            )
        verr = self._validate_input(spec, call)
        if verr is not None:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                validation_error=verr,
            )
        try:
            call_input = self._maybe_build_input(spec, call)
        except ValueError as exc:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                validation_error=str(exc),
            )
        try:
            value = spec.fn(call_input)
            if inspect.isawaitable(value):
                # The sync entrypoint can't await — caller should use arun.
                raise RuntimeError(
                    f"Tool {call.tool!r} is async; use Executor.arun() instead."
                )
            oerr = self._validate_output(spec, value)
            if oerr is not None:
                return ToolResult(
                    tool=call.tool,
                    input=call.input,
                    output=value,
                    validation_error=oerr,
                )
            return ToolResult(
                tool=call.tool, input=call.input, output=_maybe_dump(value)
            )
        except Exception as exc:
            if not self.safe:
                raise
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def arun(self, call: ToolCall) -> ToolResult:
        spec = self.registry.by_name(call.tool)
        if spec is None:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                error=f"unknown tool: {call.tool}",
            )
        verr = self._validate_input(spec, call)
        if verr is not None:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                validation_error=verr,
            )
        try:
            call_input = self._maybe_build_input(spec, call)
        except ValueError as exc:
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                validation_error=str(exc),
            )
        try:
            value = spec.fn(call_input)
            if inspect.isawaitable(value):
                value = await value
            oerr = self._validate_output(spec, value)
            if oerr is not None:
                return ToolResult(
                    tool=call.tool,
                    input=call.input,
                    output=value,
                    validation_error=oerr,
                )
            return ToolResult(
                tool=call.tool, input=call.input, output=_maybe_dump(value)
            )
        except Exception as exc:
            if not self.safe:
                raise
            return ToolResult(
                tool=call.tool,
                input=call.input,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def arun_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Run a batch of tool calls concurrently."""
        return await asyncio.gather(*(self.arun(c) for c in calls))
