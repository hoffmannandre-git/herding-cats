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

import ast
import datetime as _dt
import math
import operator
from typing import Any

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


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Plain names that resolve to callables (never via arbitrary attributes).
_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}

# ``math.<name>`` only — Attribute base must be the bare Name ``math``.
_SAFE_MATH: dict[str, Any] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "log",
        "log10",
        "log2",
        "exp",
        "floor",
        "ceil",
        "fabs",
        "pow",
        "degrees",
        "radians",
        "hypot",
        "isfinite",
        "isinf",
        "isnan",
    )
}

_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
}


class _UnsafeExpression(ValueError):
    """Raised when the expression AST contains a disallowed node shape."""


def _eval_ast(node: ast.AST) -> Any:
    """Recursively evaluate a whitelisted expression AST.

    No ``eval``/``exec``: we are the interpreter, and we only know arithmetic.
    Attribute access is allowed only as ``math.<allowlisted_fn>`` in a Call.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            # Reject bool (subclass of int), strings, None, bytes, etc.
            raise _UnsafeExpression(f"constant {node.value!r} not allowed")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise _UnsafeExpression(f"unary operator {op_type.__name__} not allowed")
        return _UNARY_OPS[op_type](_eval_ast(node.operand))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise _UnsafeExpression(f"binary operator {op_type.__name__} not allowed")
        return _BIN_OPS[op_type](_eval_ast(node.left), _eval_ast(node.right))

    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise _UnsafeExpression(f"name {node.id!r} not allowed")
        return _SAFE_NAMES[node.id]

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast(elt) for elt in node.elts]

    if isinstance(node, ast.Call):
        func = node.func
        if node.keywords:
            raise _UnsafeExpression("keyword arguments not allowed")

        if isinstance(func, ast.Name):
            if func.id not in _SAFE_FUNCS:
                raise _UnsafeExpression(f"function {func.id!r} not allowed")
            fn = _SAFE_FUNCS[func.id]
            args = [_eval_ast(a) for a in node.args]
            return fn(*args)

        if isinstance(func, ast.Attribute):
            # Only ``math.<fn>`` — never attribute chains on literals/expressions.
            if not isinstance(func.value, ast.Name) or func.value.id != "math":
                raise _UnsafeExpression("attribute access not allowed")
            if func.attr not in _SAFE_MATH:
                raise _UnsafeExpression(f"math.{func.attr} not allowed")
            fn = _SAFE_MATH[func.attr]
            args = [_eval_ast(a) for a in node.args]
            return fn(*args)

        raise _UnsafeExpression("call target not allowed")

    # Explicitly refuse Attribute outside of the Call rule above, plus
    # subscripts / comprehensions / lambdas / awaits / etc.
    raise _UnsafeExpression(f"{type(node).__name__} not allowed")


def _safe_eval_expression(expr: str) -> float | int:
    tree = ast.parse(expr, mode="eval")
    value = _eval_ast(tree)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _UnsafeExpression(f"result type {type(value).__name__} not allowed")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def calculator(input: CalcInput) -> CalcOutput:
    """Evaluate a safe arithmetic expression via AST whitelist (no ``eval``).

    Allowed: numbers, ``+ - * / // % **``, unary ``+/-``, ``pi``/``e``,
    ``abs``/``round``/``min``/``max``/``sum``, and ``math.<fn>`` for a fixed
    math-function allowlist. Attribute access on any other base is rejected.
    """
    expr = input.expression.strip()
    if not expr:
        return CalcOutput(expression=expr, value="", error="empty expression")
    try:
        value = _safe_eval_expression(expr)
        return CalcOutput(expression=expr, value=value)
    except _UnsafeExpression as exc:
        return CalcOutput(expression=expr, value="", error=f"not allowed: {exc}")
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
            description="Evaluate a safe arithmetic expression (AST whitelist, no eval).",
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
