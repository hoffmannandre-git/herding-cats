"""Built-in tools shipped with `herding_cats`.

These are the boring, useful tools every assistant ends up wanting:
a clock, a calculator, an echo for sanity-checking, and a stub web
search so the fetcher has something to call when no real search is
registered.

Each tool declares a pydantic input/output schema so the fetcher's
prompt is auto-generated with the right shape. The registry's schema
flattener (`ToolRegistry.describe()`) reads the schemas and emits a
JSON-schema block.

The web search is a *stub*: it returns an empty list and a clear note.
Wire your own (DuckDuckGo, SearXNG, Tavily) by passing a real function
to `Crew(tools={"web_search": your_fn})` — see `examples/web_search_real.py`.
"""

from __future__ import annotations

import datetime as _dt
import math

from pydantic import BaseModel, Field

from herding_cats.crew.tools import ToolSpec

# ----- get_current_time ------------------------------------------------------


class TimeInput(BaseModel):
    timezone: str = Field(
        default="UTC",
        description="IANA timezone name (e.g. 'UTC', 'Europe/Berlin').",
    )


class TimeOutput(BaseModel):
    iso: str
    timezone: str
    epoch_s: int


def get_current_time(input: TimeInput) -> TimeOutput:
    """Return the current wall-clock time in the requested timezone."""
    tz = _dt.datetime.now(_dt.timezone.utc).astimezone(
        _dt.datetime.now().astimezone().tzinfo  # local tz as fallback
    )
    if input.timezone and input.timezone != "UTC":
        try:
            from zoneinfo import ZoneInfo

            tz = _dt.datetime.now(ZoneInfo(input.timezone))
        except Exception:
            tz = _dt.datetime.now(_dt.timezone.utc)
    return TimeOutput(
        iso=tz.isoformat(),
        timezone=str(tz.tzinfo),
        epoch_s=int(tz.timestamp()),
    )


# ----- date_now (just the date, simpler shape) -------------------------------


class DateInput(BaseModel):
    timezone: str = Field(default="UTC", description="IANA timezone name.")


class DateOutput(BaseModel):
    date: str = Field(description="ISO 8601 date, e.g. '2026-09-09'.")
    weekday: str


def date_now(input: DateInput) -> DateOutput:
    """Return today's date in the requested timezone."""
    try:
        from zoneinfo import ZoneInfo

        tz = _dt.datetime.now(ZoneInfo(input.timezone or "UTC"))
    except Exception:
        tz = _dt.datetime.now(_dt.timezone.utc)
    return DateOutput(date=tz.date().isoformat(), weekday=tz.strftime("%A"))


# ----- calculator ------------------------------------------------------------


class CalcInput(BaseModel):
    expression: str = Field(
        description="A Python-arithmetic expression, e.g. '2 + 2' or 'math.sqrt(16)'.",
    )


class CalcOutput(BaseModel):
    expression: str
    value: float | int | str
    error: str | None = None


_SAFE_GLOBALS: dict[str, object] = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "math": math,
    "pi": math.pi,
    "e": math.e,
}


def calculator(input: CalcInput) -> CalcOutput:
    """Evaluate a safe arithmetic expression.

    Allowed: numbers, arithmetic operators, math.* functions, abs/round/min/max.
    Not allowed: arbitrary function calls, attribute access, imports.
    """
    expr = input.expression.strip()
    if not expr:
        return CalcOutput(expression=expr, value="", error="empty expression")
    try:
        # Reject any top-level function call other than the allowlist.
        # A "top-level" call is one not preceded by a `.` (i.e. not
        # `math.sqrt` — that one is allowed).
        import re as _re

        for match in _re.finditer(r"(?<![\w.])([a-zA-Z_]\w*)\s*\(", expr):
            tok = match.group(1)
            if tok not in {"math", "abs", "round", "min", "max", "sum"}:
                return CalcOutput(
                    expression=expr,
                    value="",
                    error=f"function {tok!r} not allowed",
                )
        value: float | int = eval(expr, _SAFE_GLOBALS, {})
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return CalcOutput(expression=expr, value=value)
    except Exception as exc:
        return CalcOutput(expression=expr, value="", error=f"{type(exc).__name__}: {exc}")


# ----- echo (sanity check) ---------------------------------------------------


class EchoInput(BaseModel):
    text: str
    repeat: int = Field(default=1, ge=1, le=10)


class EchoOutput(BaseModel):
    text: str
    repeats: int


def echo(input: EchoInput) -> EchoOutput:
    """Repeat `text` `repeat` times. Useful as a smoke test."""
    return EchoOutput(text=input.text * input.repeat, repeats=input.repeat)


# ----- web_search_stub -------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query string.")
    top_k: int = Field(default=5, ge=1, le=20)


class WebSearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    query: str
    hits: list[WebSearchHit]
    note: str


def web_search_stub(input: WebSearchInput) -> WebSearchOutput:
    """A placeholder web search. Returns an empty result set.

    Replace by passing your own `web_search` function to `Crew(tools=...)`.
    """
    return WebSearchOutput(
        query=input.query,
        hits=[],
        note="web_search_stub returns no results; provide a real implementation.",
    )


# ----- registry --------------------------------------------------------------


def builtin_tools() -> list[ToolSpec]:
    """Return a list of all built-in `ToolSpec` instances.

    Use directly:

        from herding_cats import Crew
        from herding_cats.tools_builtin import builtin_tools

        registry = ToolRegistry()
        for spec in builtin_tools():
            registry.specs.append(spec)
        crew = Crew(ollama=ollama, tools={s.name: s for s in builtin_tools()})

    Or simply:

        from herding_cats.tools_builtin import BUILTIN_TOOL_FUNCS
        crew = Crew(ollama=ollama, tools=BUILTIN_TOOL_FUNCS)
    """
    return [
        ToolSpec(
            name="get_current_time",
            description="Return the current wall-clock time in a timezone.",
            fn=get_current_time,
            input_schema=TimeInput,
            output_schema=TimeOutput,
        ),
        ToolSpec(
            name="date_now",
            description="Return today's date (ISO 8601) and weekday.",
            fn=date_now,
            input_schema=DateInput,
            output_schema=DateOutput,
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate a sandboxed arithmetic expression.",
            fn=calculator,
            input_schema=CalcInput,
            output_schema=CalcOutput,
        ),
        ToolSpec(
            name="echo",
            description="Repeat text back N times. Useful for sanity checks.",
            fn=echo,
            input_schema=EchoInput,
            output_schema=EchoOutput,
        ),
        ToolSpec(
            name="web_search",
            description="Search the public web. Returns up to top_k hits.",
            fn=web_search_stub,
            input_schema=WebSearchInput,
            output_schema=WebSearchOutput,
        ),
    ]


# Convenience: {name: fn} for passing straight into `Crew(tools=...)`.
# Note: when you use this, schemas are NOT auto-discovered (Crew only
# sees the callable). Register the full ToolSpec list if you want
# schema-driven prompts.
BUILTIN_TOOL_FUNCS: dict[str, object] = {
    "get_current_time": get_current_time,
    "date_now": date_now,
    "calculator": calculator,
    "echo": echo,
    "web_search": web_search_stub,
}
