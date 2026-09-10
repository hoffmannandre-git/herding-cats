"""Finalist role: write the user-facing answer.

The finalist is the *only* role the user sees. It receives the question,
the thinker's notes, and the tool results, and produces the answer.

Default behavior: plain prose. No JSON. No `FINAL:` markers. The
runner prints `resp.message.content` verbatim.

If you want a structured final answer (JSON, Markdown with explicit
sections, citations), write a custom prompt and pass it via
`Crew(prompts={"finalist": "..."})`.
"""

from __future__ import annotations

import logging

from herding_cats.crew.state import CrewState
from herding_cats.helpers.env import get_env
from herding_cats.ollama import ChatMessage, ChatRequest, OllamaClient

logger = logging.getLogger("herding_cats.crew.finalist")

MAX_TOKENS = int(get_env("FINALIST_MAX_TOKENS") or "2048")


async def run_finalist(
    *,
    ollama: OllamaClient,
    state: CrewState,
    prompt: str,
    model: str,
) -> str:
    req = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        options={"temperature": 0.0, "num_predict": MAX_TOKENS},
    )
    resp = await ollama.achat(req)
    answer = resp.message.content.strip()
    if not answer:
        logger.warning("finalist returned empty answer")
        state.current.warnings.append("finalist: empty answer")
        answer = "(the crew ran out of room — try a larger model or tighter prompt)"
    return answer


async def run_direct(
    *,
    ollama: OllamaClient,
    question: str,
    prompt: str,
    model: str,
) -> str:
    """Short-circuit path for smalltalk / greetings — no crew loop.

    Returns the model's reply as-is. Used by the runner when the
    orchestrator's rules-mode (or LLM-mode) decides this turn does not
    need the full crew.
    """
    req = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        options={"temperature": 0.0, "num_predict": 256},
    )
    resp = await ollama.achat(req)
    return resp.message.content.strip() or "(no reply)"
