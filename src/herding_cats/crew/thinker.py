"""Thinker role: produce a planning note and concrete search terms.

The thinker does not search. It does not answer the user. It does two
things:

1. Reads the question (and, on later rounds, the tool results so far)
2. Emits a `notes` paragraph and a `search_terms` list

The notes become context for the finalist; the search terms become a
queue the fetcher pulls from. The planner does not run tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from herding_cats.crew.state import CrewState
from herding_cats.helpers.env import get_env
from herding_cats.helpers.json_util import extract_json_object
from herding_cats.ollama import ChatMessage, ChatRequest, OllamaClient

logger = logging.getLogger("herding_cats.crew.thinker")

MAX_TOKENS = int(get_env("THINKER_MAX_TOKENS") or "1024")


@dataclass(frozen=True)
class ThinkerResult:
    notes: str
    search_terms: list[str]


async def run_thinker(
    *,
    ollama: OllamaClient,
    state: CrewState,
    prompt: str,
    model: str,
) -> ThinkerResult:
    """Run the thinker and parse the JSON contract.

    Returns `(notes, search_terms)`. If the model returns unparseable
    output, falls back to empty fields with a warning so the orchestrator
    can still proceed.
    """
    req = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        format="json",
        options={"temperature": 0.0, "num_predict": MAX_TOKENS},
    )
    resp = await ollama.achat(req)
    obj = extract_json_object(resp.message.content)
    if obj is None:
        logger.warning("thinker returned unparseable output: %r", resp.message.content[:200])
        state.current.warnings.append("thinker: unparseable JSON output")
        return ThinkerResult(notes="", search_terms=[])

    notes = str(obj.get("notes", "")).strip()
    raw_terms = obj.get("search_terms") or []
    if not isinstance(raw_terms, list):
        raw_terms = [str(raw_terms)]
    search_terms = [str(t).strip() for t in raw_terms if str(t).strip()]
    return ThinkerResult(notes=notes, search_terms=search_terms)
