"""Fetcher role: emit exactly one tool call as JSON.

The fetcher does not run tools. It picks one tool from the registry
description and emits a JSON plan the executor can run. The plan is:

    {"tool": "<name>", "input": {...}}

Or, when the fetcher decides no tool fits:

    {"tool": null, "reason": "..."}

The runner enforces the plan structure. The prompt is the source of
truth for *which* tool to pick and what shape the input should have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from herding_cats.crew.state import CrewState
from herding_cats.crew.tools import ToolCall
from herding_cats.helpers.env import get_env
from herding_cats.helpers.json_util import extract_json_object
from herding_cats.ollama import ChatMessage, ChatRequest, OllamaClient

logger = logging.getLogger("herding_cats.crew.fetcher")

MAX_TOKENS = int(get_env("FETCHER_MAX_TOKENS") or "1024")


@dataclass(frozen=True)
class FetcherResult:
    call: ToolCall | None
    reason: str = ""


async def run_fetcher(
    *,
    ollama: OllamaClient,
    state: CrewState,
    prompt: str,
    model: str,
) -> FetcherResult:
    req = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        format="json",
        options={"temperature": 0.0, "num_predict": MAX_TOKENS},
    )
    resp = await ollama.achat(req)
    obj = extract_json_object(resp.message.content)
    if obj is None:
        logger.warning("fetcher returned unparseable output: %r", resp.message.content[:200])
        state.current.warnings.append("fetcher: unparseable JSON output")
        return FetcherResult(call=None, reason="unparseable output")

    tool_name = obj.get("tool")
    if tool_name in (None, "", "null"):
        return FetcherResult(
            call=None, reason=str(obj.get("reason", "no tool needed"))
        )

    if not isinstance(tool_name, str):
        state.current.warnings.append(f"fetcher: invalid tool name {tool_name!r}")
        return FetcherResult(call=None, reason="invalid tool name")

    inp = obj.get("input") or {}
    if not isinstance(inp, dict):
        state.current.warnings.append(f"fetcher: input is not a dict: {inp!r}")
        inp = {}

    return FetcherResult(call=ToolCall(tool=tool_name, input=inp))
