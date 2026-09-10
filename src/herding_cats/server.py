"""FastAPI HTTP server for `herding_cats`.

Endpoints:

* `POST /crew/run`      — one-shot: question in, answer + state out.
* `POST /crew/step`     — per-step steering: optional state in, decision + new state out.
* `POST /crew/stream`   — Server-Sent Events stream of `RunnerEvent`s.
* `GET  /sessions`      — list session ids.
* `GET  /sessions/{id}` — get a session's history.
* `POST /sessions/{id}/ask` — ask one question, persist turn.
* `DELETE /sessions/{id}`   — drop a session.
* `GET  /health`        — Ollama liveness check (`/healthz` alias).

Install with the `server` extra:

    pip install herding-cats[server]

Run:

    uvicorn herding_cats.server:app --host 127.0.0.1 --port 8765

Or programmatically:

    from herding_cats.server import create_app
    app = create_app(crew=my_crew)
    uvicorn.run(app, host="127.0.0.1", port=8765)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from herding_cats.crew.runner import Crew, CrewRunner, StepResult
from herding_cats.crew.state import CrewState
from herding_cats.session import (
    CrewSession,
    InMemorySession,
    JsonFileSession,
    TurnRecord,
)

logger = logging.getLogger("herding_cats.server")


# ----- optional imports (server is an extra) --------------------------------
# FastAPI / Pydantic are only required for `herding_cats[server]`. Import
# them at module level (not lazily inside `_require_fastapi`) so the
# type names — especially `StreamingResponse` — end up in the module's
# `__globals__`. Without that, the
# `from __future__ import annotations` plus Pydantic 2.13's
# `TypeAdapter` / FastAPI 0.110's openapi generator will throw
# ``PydanticUserError: TypeAdapter[ForwardRef('StreamingResponse')] is
# not fully defined`` when somebody hits ``/openapi.json``.
try:
    from fastapi import Body, FastAPI, HTTPException  # noqa: F401
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel  # noqa: F401
except ImportError:  # server extra not installed
    pass


def _require_fastapi():  # type: ignore[no-untyped-def]
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(
            "herding_cats.server requires the 'server' extra. "
            "Install with: pip install herding-cats[server]"
        ) from e
    return FastAPI, HTTPException, StreamingResponse, BaseModel, Body


# ----- request / response models (module-level so FastAPI introspects them)


# Pydantic models are defined at module level so FastAPI's introspection
# can resolve their `__module__` and forward refs correctly. Defining
# them inside a function makes FastAPI treat them as forward refs,
# which then fail to resolve at request time.
try:
    from pydantic import BaseModel as _PMBaseModel

    class AskIn(_PMBaseModel):
        question: str

    class RunOut(_PMBaseModel):
        answer: str
        state: dict[str, Any]
        warnings: list[str]

    class StepIn(_PMBaseModel):
        state: dict[str, Any] | None = None
        question: str | None = None

    class StepOut(_PMBaseModel):
        decision: dict[str, Any]
        new_state: dict[str, Any]
        done: bool
        event: dict[str, Any] | None = None

    class StreamIn(_PMBaseModel):
        question: str
        session_id: str | None = None

    class SessionAskIn(_PMBaseModel):
        question: str

    class SessionAskOut(_PMBaseModel):
        answer: str
        turn_number: int

    _MODELS = (
        AskIn,
        RunOut,
        StepIn,
        StepOut,
        StreamIn,
        SessionAskIn,
        SessionAskOut,
    )
except ImportError:  # pydantic not installed
    _MODELS = None  # type: ignore[assignment]


# ----- factory -------------------------------------------------------------


def create_app(*, crew: Crew, storage_dir: str | None = None):  # type: ignore[no-untyped-def]
    """Build a FastAPI app bound to a specific `Crew` instance.

    `storage_dir`, if given, persists sessions to disk under that path.
    Otherwise sessions live in memory.
    """
    if _MODELS is None:
        raise ImportError(
            "herding_cats.server requires pydantic. "
            "Install with: pip install herding-cats[server]"
        )
    FastAPI, HTTPException, StreamingResponse, _BaseModel, _Body = _require_fastapi()
    (AskIn, RunOut, StepIn, StepOut, StreamIn, _SessionAskIn, SessionAskOut) = _MODELS

    app = FastAPI(
        title="herding-cats",
        description="Multi-agent LLM crew runtime (HTTP).",
        version="0.3.0",
    )
    runner = CrewRunner(crew)
    if storage_dir is not None:
        store: Any = JsonFileSession(storage_dir)
    else:
        store = InMemorySession()
    sessions: dict[str, CrewSession] = {}

    def get_session(session_id: str) -> CrewSession:
        if session_id not in sessions:
            sessions[session_id] = CrewSession(
                runner=runner,
                store=store,
                session_id=session_id,
            )
        return sessions[session_id]

    # ----- routes -----

    @app.get("/health")
    @app.get("/healthz")  # alias — k8s-style name kept for older probes
    async def health() -> dict[str, Any]:
        ok = False
        try:
            ok = crew.ollama.health()
        except Exception as e:
            logger.warning("ollama health check failed: %r", e)
        return {"ok": ok, "model": crew.model}

    @app.post("/crew/run", response_model=RunOut)
    async def crew_run(
        payload: AskIn, session_id: str | None = None
    ) -> RunOut:
        if session_id is not None:
            return await _session_ask(payload, session_id)
        # Stateless: build a fresh state, run to completion.
        state = CrewState(question=payload.question)
        events = []
        async for ev in runner.arun_stream(payload.question, state=state):
            events.append(ev)
        answer = next(
            (ev.payload.get("answer", "") for ev in events if ev.kind == "final"),
            "",
        )
        return RunOut(
            answer=answer,
            state=state.model_dump(),
            warnings=list(state.current.warnings),
        )

    async def _session_ask(payload: AskIn, session_id: str) -> RunOut:
        sess = get_session(session_id)
        prior = sess.history()
        state = CrewState(
            question=payload.question,
            prior_turns=[t.to_memory() for t in prior[-sess.max_prior_turns :]],
            prior_context=sess.prior_context or None,
            seen_chunk_ids=list(
                {cid for t in prior for cid in t.seen_chunk_ids}
            ),
        )
        async for _ev in runner.arun_stream(payload.question, state=state):
            pass
        answer = state.current.answer
        turn = TurnRecord(
            question=payload.question,
            answer=answer,
            notes=state.current.notes,
            seen_chunk_ids=list(state.seen_chunk_ids),
        )
        prior.append(turn)
        sess.store.save(session_id, prior)
        return RunOut(
            answer=answer,
            state=state.model_dump(),
            warnings=list(state.current.warnings),
        )

    @app.post("/crew/step", response_model=StepOut)
    async def crew_step(payload: StepIn) -> StepOut:
        if payload.state is not None:
            state = CrewState.model_validate(payload.state)
        elif payload.question is not None:
            state = CrewState(question=payload.question)
        else:
            raise HTTPException(
                status_code=400, detail="state or question is required"
            )
        result: StepResult = await runner.astep(state)
        return StepOut(
            decision=result.decision.to_dict(),
            new_state=result.state.model_dump(),
            done=result.done,
            event=(
                {"kind": result.event.kind, "payload": result.event.payload}
                if result.event
                else None
            ),
        )

    @app.post("/crew/stream")
    async def crew_stream(payload: StreamIn) -> StreamingResponse:
        # SSE format: data: <json>\n\n
        async def event_source() -> AsyncIterator[bytes]:
            state: CrewState
            if payload.session_id is not None:
                sess = get_session(payload.session_id)
                prior = sess.history()
                state = CrewState(
                    question=payload.question,
                    prior_turns=[t.to_memory() for t in prior[-sess.max_prior_turns :]],
                    prior_context=sess.prior_context or None,
                    seen_chunk_ids=list(
                        {cid for t in prior for cid in t.seen_chunk_ids}
                    ),
                )
            else:
                state = CrewState(question=payload.question)

            async for ev in runner.arun_stream(payload.question, state=state):
                line = json.dumps(
                    {"kind": ev.kind, "payload": ev.payload}, ensure_ascii=False
                )
                yield f"data: {line}\n\n".encode()

            if payload.session_id is not None:
                sess = get_session(payload.session_id)
                prior = sess.history()
                turn = TurnRecord(
                    question=payload.question,
                    answer=state.current.answer,
                    notes=state.current.notes,
                    seen_chunk_ids=list(state.seen_chunk_ids),
                )
                prior.append(turn)
                sess.store.save(payload.session_id, prior)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/sessions")
    async def list_sessions() -> dict[str, Any]:
        ids = list(sessions.keys()) if isinstance(store, InMemorySession) else store.list_sessions()
        return {"sessions": ids}

    @app.get("/sessions/{session_id}")
    async def get_session_history(session_id: str) -> dict[str, Any]:
        turns = store.load(session_id) or []
        return {
            "session_id": session_id,
            "turns": [
                {
                    "question": t.question,
                    "answer": t.answer,
                    "timestamp": t.timestamp,
                    "notes": t.notes,
                    "seen_chunk_ids": t.seen_chunk_ids,
                }
                for t in turns
            ],
        }

    @app.post("/sessions/{session_id}/ask", response_model=SessionAskOut)
    async def session_ask(
        session_id: str, payload: SessionAskIn
    ) -> SessionAskOut:
        sess = get_session(session_id)
        prior = sess.history()
        state = CrewState(
            question=payload.question,
            prior_turns=[t.to_memory() for t in prior[-sess.max_prior_turns :]],
            prior_context=sess.prior_context or None,
            seen_chunk_ids=list(
                {cid for t in prior for cid in t.seen_chunk_ids}
            ),
        )
        async for _ev in runner.arun_stream(payload.question, state=state):
            pass
        turn = TurnRecord(
            question=payload.question,
            answer=state.current.answer,
            notes=state.current.notes,
            seen_chunk_ids=list(state.seen_chunk_ids),
        )
        prior.append(turn)
        sess.store.save(session_id, prior)
        return SessionAskOut(answer=turn.answer, turn_number=len(prior))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        store.delete(session_id)
        sessions.pop(session_id, None)
        return {"deleted": session_id}

    return app


__all__ = ["create_app"]
