"""Orchestrator: the only role that decides what happens next.

The orchestrator's job is the *smallest* in the crew: pick exactly one of
{think, fetch, final, stop} and a one-line reason.

Two modes:

* `rules` — A pure-Python state machine. Fast, deterministic, testable.
  Used by default and by the test suite.
* `llm` — Ask the model itself to pick. Slower, less predictable, but
  can pick up nuances the rules miss. Used when you set
  `HERDING_CATS_ORCHESTRATOR_MODE=llm` or pass `orchestrator_mode="llm"`.

In both modes the runner enforces the *real* max-loop cap; the
orchestrator is told a budget one round lower so it plans to finish
in time (the n-1 trick).

Anti-ping-pong: the rules-mode orchestrator uses `state.reflection_think_done`
to force exactly one reflection-think pass after a successful fetch
and then go to `final`. Without the flag, the runner can spin
think → fetch → think forever on conversational queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from herding_cats.crew.state import CrewState, NextAgent
from herding_cats.helpers.env import get_env
from herding_cats.helpers.json_util import extract_json_object

logger = logging.getLogger("herding_cats.crew.orchestrator")

ORCHESTRATOR_MODE: Literal["rules", "llm"] = (
    (get_env("ORCHESTRATOR_MODE") or "rules").strip().lower()  # type: ignore[assignment]
)

MAX_ORCHESTRATOR_ROUNDS = int(get_env("MAX_LOOP") or "12")
# Tell the orchestrator one round less than the real cap so it finishes early.
BUDGET_ROUND_PAD = int(get_env("BUDGET_PAD") or "1")
MAX_FETCH_TURNS = int(get_env("MAX_FETCH") or "3")
MAX_THINK_TURNS = int(get_env("MAX_THINK") or "4")
SEARCHES_PER_FETCH = int(get_env("SEARCHES_PER_FETCH") or "2")


@dataclass(frozen=True)
class OrchestratorBudget:
    """What the orchestrator sees — not the runner's real loop cap."""

    max_rounds: int
    round_number: int
    rounds_after_this: int
    fetch_used: int
    fetch_remaining: int
    think_used: int
    think_remaining: int
    should_finish: bool
    finish_hint: str

    def summary_line(self) -> str:
        tail = (
            f"fetch {self.fetch_used}/{MAX_FETCH_TURNS} "
            f"(remaining {self.fetch_remaining}), "
            f"think {self.think_used}/{MAX_THINK_TURNS} "
            f"(remaining {self.think_remaining})"
        )
        if self.rounds_after_this <= 0:
            return f"budget: planning limit {self.max_rounds} reached — wrap up. {tail}"
        return (
            f"budget: round {self.round_number}/{self.max_rounds}, "
            f"after this step {self.rounds_after_this} rounds left. {tail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "round_number": self.round_number,
            "rounds_after_this": self.rounds_after_this,
            "fetch_used": self.fetch_used,
            "fetch_remaining": self.fetch_remaining,
            "think_used": self.think_used,
            "think_remaining": self.think_remaining,
            "should_finish": self.should_finish,
            "finish_hint": self.finish_hint,
        }


@dataclass(frozen=True)
class OrchestratorDecision:
    next: NextAgent
    reason: str
    mode: Literal["rules", "llm"]
    budget: OrchestratorBudget | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "next": self.next,
            "reason": self.reason,
            "mode": self.mode,
        }
        if self.budget is not None:
            out["budget"] = self.budget.to_dict()
        return out


def compute_budget(state: CrewState) -> OrchestratorBudget:
    """Compute the budget the orchestrator is told about.

    This is `max_rounds - 1` (n-1 trick) so the orchestrator finishes
    one round before the hard cap.
    """
    rounds_after = max(0, MAX_ORCHESTRATOR_ROUNDS - BUDGET_ROUND_PAD - state.round_number)
    fetch_remaining = max(0, MAX_FETCH_TURNS - state.fetch_used)
    think_remaining = max(0, MAX_THINK_TURNS - state.think_used)

    # Finish conditions: budget exhausted, all counters zero, smalltalk,
    # or hit-fetch with no reflection room (so don't start another cycle).
    fetch_exhausted_no_hits = (
        state.fetch_used >= MAX_FETCH_TURNS and not state.current.hits
    )
    should_finish = (
        rounds_after <= 0
        or (fetch_remaining == 0 and think_remaining == 0)
        or _looks_like_smalltalk(state.question)
        or fetch_exhausted_no_hits
    )
    if fetch_exhausted_no_hits:
        hint = "all fetches returned nothing — finish, don't keep searching"
    elif rounds_after <= 0:
        hint = "no rounds left after this — go to final"
    elif fetch_remaining == 0 and think_remaining == 0:
        hint = "no fetch and no think left — go to final"
    else:
        hint = "planning budget available"

    return OrchestratorBudget(
        max_rounds=MAX_ORCHESTRATOR_ROUNDS - BUDGET_ROUND_PAD,
        round_number=state.round_number + 1,
        rounds_after_this=rounds_after - 1 if rounds_after > 0 else 0,
        fetch_used=state.fetch_used,
        fetch_remaining=fetch_remaining,
        think_used=state.think_used,
        think_remaining=think_remaining,
        should_finish=should_finish,
        finish_hint=hint,
    )


_GREETINGS = frozenset(
    {
        "hi", "hello", "hey", "hi.", "hello.", "hey.",
        "hallo", "moin", "servus", "guten morgen", "guten tag",
        "thanks", "thank you", "danke", "thanks!", "thank you!",
        "bye", "goodbye", "tschüss", "ciao",
    }
)


def looks_like_smalltalk(question: str) -> bool:
    """Public smalltalk heuristic — also exposed as `Crew.short_circuit_smalltalk`."""
    q = question.strip().lower().rstrip(".!?")
    return bool(q in _GREETINGS or len(q) <= 4)


# Backwards-compat alias used by older imports.
_looks_like_smalltalk = looks_like_smalltalk


def decide_rules(state: CrewState) -> OrchestratorDecision:
    """The rules-mode state machine.

    Logic, roughly:
    * If smalltalk → final.
    * If first round → think.
    * After think → fetch (until fetch budget exhausted).
    * After fetch:
      - no hits → final (don't loop on nothing)
      - hits + reflection not done + budget → think (reflect once)
      - hits + reflection done → final
    * Always respect `should_finish`.
    """
    budget = compute_budget(state)
    if budget.should_finish:
        # The user asked something but we never fetched — let one think pass try.
        if (
            state.fetch_used == 0
            and not looks_like_smalltalk(state.question)
            and state.think_used < state.think_max
        ):
            return OrchestratorDecision(
                next="think",
                reason="no rounds left; giving thinker one chance to plan before final",
                mode="rules",
                budget=budget,
            )
        return OrchestratorDecision(
            next="final",
            reason=budget.finish_hint,
            mode="rules",
            budget=budget,
        )

    last_role = state.last_role
    # First round.
    if last_role is None:
        if looks_like_smalltalk(state.question):
            return OrchestratorDecision(
                next="final", reason="smalltalk", mode="rules", budget=budget
            )
        return OrchestratorDecision(
            next="think",
            reason="first round — plan before fetching",
            mode="rules",
            budget=budget,
        )

    if last_role == "think":
        if state.fetch_used < state.fetch_max:
            return OrchestratorDecision(
                next="fetch",
                reason="thinker planned; try fetching",
                mode="rules",
                budget=budget,
            )
        return OrchestratorDecision(
            next="final",
            reason="thinker planned; fetch budget exhausted → answer",
            mode="rules",
            budget=budget,
        )

    if last_role == "fetch":
        if not state.current.hits:
            return OrchestratorDecision(
                next="final",
                reason="fetch returned nothing — answer with that",
                mode="rules",
                budget=budget,
            )
        # Hits present — reflect once, then answer. The runner clears
        # `reflection_think_done` when the reflection actually runs.
        if not state.reflection_think_done and state.think_used < state.think_max:
            return OrchestratorDecision(
                next="think",
                reason="fetch returned hits — thinker evaluates before next fetch",
                mode="rules",
                budget=budget,
            )
        return OrchestratorDecision(
            next="final",
            reason="hits evaluated — answer",
            mode="rules",
            budget=budget,
        )

    # After `final` or anything unexpected: stop.
    return OrchestratorDecision(
        next="stop", reason="nothing left to do", mode="rules", budget=budget
    )


def decide_llm(
    state: CrewState,
    *,
    system_prompt: str,
    user_prompt: str,
    llm_caller,
) -> OrchestratorDecision:
    """Ask the model itself to decide, given the rendered prompt.

    `llm_caller` is `OllamaClient.achat` (or a stub) — a coroutine that
    takes a `ChatRequest` and returns a `ChatResponse`. `system_prompt`
    is the static orchestrator prompt (role + rules); `user_prompt` is
    the per-step state summary + budget block.
    """
    from herding_cats.ollama import ChatMessage, ChatRequest

    req = ChatRequest(
        model=get_env("MODEL") or "llama3.1:8b",
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        format="json",
        options={"temperature": 0.0, "num_predict": 256},
    )
    response = llm_caller(req)  # may be sync or awaitable
    text = response.message.content
    obj = extract_json_object(text)
    if obj is None:
        return OrchestratorDecision(
            next="final", reason="orchestrator returned unparseable output", mode="llm"
        )
    nxt = obj.get("next")
    reason = str(obj.get("reason", ""))
    if nxt not in {"think", "fetch", "final", "stop"}:
        return OrchestratorDecision(
            next="final", reason=f"orchestrator picked invalid {nxt!r}", mode="llm"
        )
    return OrchestratorDecision(next=nxt, reason=reason, mode="llm")  # type: ignore[arg-type]


def pipeline_status(state: CrewState, decision: OrchestratorDecision | None) -> dict[str, str]:
    """Return a `{role: status}` dict for the UI.

    Statuses:
    * `pending` — role hasn't run this turn
    * `running` — runner is about to call / is calling this role
    * `complete` — role produced output this turn
    * `skipped` — role wasn't needed this turn
    * `error` — role failed and the runner recorded a warning
    """
    statuses: dict[str, str] = {
        "orchestrator": "complete",
        "thinker": "pending",
        "fetcher": "pending",
        "finalist": "pending",
        "executor": "pending",
    }
    if state.think_used > 0:
        statuses["thinker"] = "complete"
    if state.fetch_used > 0:
        statuses["fetcher"] = "complete"
    if state.current.answer:
        statuses["finalist"] = "complete"
    if decision is not None:
        # Mark the next-decided role as running.
        if decision.next == "think":
            statuses["thinker"] = "running"
        elif decision.next == "fetch":
            statuses["fetcher"] = "running"
        elif decision.next == "final":
            statuses["finalist"] = "running"
    # Executor runs alongside fetcher.
    if statuses["fetcher"] in {"complete", "running"}:
        statuses["executor"] = statuses["fetcher"]
    # Skipped: smalltalk went straight to finalist, never fetched.
    if (
        state.fetch_used == 0
        and state.think_used == 0
        and state.current.answer
    ):
        statuses["thinker"] = "skipped"
        statuses["fetcher"] = "skipped"
        statuses["executor"] = "skipped"
    return statuses
