"""CrewSession: a multi-turn conversation with cross-turn memory.

A `CrewSession` is a thin layer on top of `CrewRunner` that remembers
prior turns. Each new `ask()` builds a `CrewState` with `prior_turns`,
`prior_context`, and `seen_chunk_ids` populated from the session's
history, so the thinker can resolve follow-up questions ("what about
tomorrow?") and the fetcher can avoid re-retrieving the same chunks.

Two storage backends:
* `InMemorySession` — the default, lives in the process.
* `JsonFileSession` — persists to a JSON file. Each session_id maps to
  one file under `base_dir`.

Use either directly or via the FastAPI endpoint in
`herding_cats.server` (when installed with the `server` extra).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from herding_cats.crew.runner import CrewRunner
from herding_cats.crew.state import CrewState, TurnMemory

logger = logging.getLogger("herding_cats.session")


@dataclass
class TurnRecord:
    """A completed turn, ready to be stored on the session."""

    question: str
    answer: str
    notes: str = ""
    seen_chunk_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_memory(self) -> TurnMemory:
        return TurnMemory(
            notes=self.notes,
            answer=self.answer,
        )


@runtime_checkable
class SessionStore(Protocol):
    """A backing store for conversation history."""

    def load(self, session_id: str) -> list[TurnRecord] | None: ...
    def save(self, session_id: str, turns: list[TurnRecord]) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def list_sessions(self) -> list[str]: ...


class InMemorySession:
    """Process-local session store. Lost on restart."""

    def __init__(self) -> None:
        self._turns: dict[str, list[TurnRecord]] = {}

    def load(self, session_id: str) -> list[TurnRecord] | None:
        return self._turns.get(session_id)

    def save(self, session_id: str, turns: list[TurnRecord]) -> None:
        self._turns[session_id] = list(turns)

    def delete(self, session_id: str) -> None:
        self._turns.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._turns.keys())


class JsonFileSession:
    """Persistent session store, one JSON file per session."""

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        # Defensive: avoid path traversal in the session id.
        if "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.base_dir / f"{session_id}.json"

    def load(self, session_id: str) -> list[TurnRecord] | None:
        p = self._path(session_id)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [TurnRecord(**r) for r in raw]

    def save(self, session_id: str, turns: list[TurnRecord]) -> None:
        p = self._path(session_id)
        payload = [
            {
                "question": t.question,
                "answer": t.answer,
                "notes": t.notes,
                "seen_chunk_ids": t.seen_chunk_ids,
                "timestamp": t.timestamp,
            }
            for t in turns
        ]
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.json")]


@dataclass
class CrewSession:
    """A multi-turn conversation. Composes a `CrewRunner` and a `SessionStore`.

    Usage:

        runner = CrewRunner(crew)
        store = InMemorySession()
        sess = CrewSession(runner=runner, store=store)

        sess.ask("What's the weather in Berlin?")
        sess.ask("And tomorrow?")   # <-- prior turn is in $prior_turns
    """

    runner: CrewRunner
    store: SessionStore
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prior_context: str = ""
    # Cap on prior turns we hand to the model.
    max_prior_turns: int = 5

    @classmethod
    def with_in_memory(
        cls, runner: CrewRunner, *, prior_context: str = ""
    ) -> CrewSession:
        return cls(
            runner=runner,
            store=InMemorySession(),
            prior_context=prior_context,
        )

    @classmethod
    def with_json_file(
        cls,
        runner: CrewRunner,
        base_dir: str | os.PathLike[str],
        *,
        session_id: str | None = None,
        prior_context: str = "",
    ) -> CrewSession:
        return cls(
            runner=runner,
            store=JsonFileSession(base_dir),
            session_id=session_id or str(uuid.uuid4()),
            prior_context=prior_context,
        )

    def _load(self) -> list[TurnRecord]:
        loaded = self.store.load(self.session_id)
        return list(loaded) if loaded else []

    def _persist(self, turns: list[TurnRecord]) -> None:
        self.store.save(self.session_id, turns)

    def _build_state(self, question: str, prior: list[TurnRecord]) -> CrewState:
        # Cap prior turns to keep prompts bounded.
        recent = prior[-self.max_prior_turns :]
        return CrewState(
            question=question,
            prior_turns=[t.to_memory() for t in recent],
            prior_context=self.prior_context or None,
            seen_chunk_ids=list({cid for t in prior for cid in t.seen_chunk_ids}),
        )

    def ask(self, question: str) -> str:
        """Run one crew turn, remembering everything the crew saw."""
        prior = self._load()
        state = self._build_state(question, prior)
        # Drive the crew via the streaming API so we capture everything.
        final_answer = ""
        events = list(self.runner.run_stream(question, state=state))
        for ev in events:
            if ev.kind == "final":
                final_answer = ev.payload.get("answer", "")
                break
        # Update history: append the new turn.
        turn = TurnRecord(
            question=question,
            answer=state.current.answer or final_answer,
            notes=state.current.notes,
            seen_chunk_ids=list(state.seen_chunk_ids),
        )
        prior.append(turn)
        self._persist(prior)
        return turn.answer

    def history(self) -> list[TurnRecord]:
        return self._load()

    def clear(self) -> None:
        self.store.delete(self.session_id)


__all__ = [
    "CrewSession",
    "InMemorySession",
    "JsonFileSession",
    "SessionStore",
    "TurnRecord",
]
