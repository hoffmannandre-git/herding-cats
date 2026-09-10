"""Tests for the FastAPI HTTP server.

The server is optional (requires the `server` extra). We skip the
tests if FastAPI isn't installed; otherwise we drive the app via
`TestClient`.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from herding_cats import (  # noqa: E402
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Crew,
    OllamaClient,
)


class _StubOllama(OllamaClient):
    def __init__(self) -> None:  # type: ignore[no-untyped-def]
        self.base_url = "http://stub"
        self.timeout_s = 0.0
        self.calls: list = []
        self._responses: list[str] = ['{"notes":"","search_terms":[]}', "stub answer"]
        self._default = '{"notes":"","search_terms":[]}'

    async def achat(self, request: ChatRequest) -> ChatResponse:  # type: ignore[override]
        self.calls.append(request)
        text = self._responses.pop(0) if self._responses else self._default
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role="assistant", content=text),
            done=True,
        )

    def health(self) -> bool:  # type: ignore[override]
        return True


def _make_app(storage_dir=None):  # type: ignore[no-untyped-def]
    from herding_cats.server import create_app

    crew = Crew(ollama=_StubOllama(), tools={"echo": lambda i: i})  # type: ignore[arg-type]
    return create_app(crew=crew, storage_dir=storage_dir)


def test_health_ok() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "model" in body


def test_healthz_alias() -> None:
    """`/healthz` stays as a k8s-style alias for `/health`."""
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["ok"] is True


def test_crew_run_returns_answer_and_state() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.post("/crew/run", json={"question": "hello?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]
        assert "state" in body
        assert "warnings" in body


def test_crew_step_round_trips_state() -> None:
    app = _make_app()
    with TestClient(app) as client:
        # Start with just a question.
        r = client.post(
            "/crew/step",
            json={"question": "what is the weather in Berlin?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"]["next"] in {"think", "fetch", "final", "stop"}
        assert body["new_state"]["question"].startswith("what")
        # Send the new_state back in to advance.
        r2 = client.post("/crew/step", json={"state": body["new_state"]})
        assert r2.status_code == 200
        assert r2.json()["new_state"]["round_number"] >= 1


def test_crew_step_requires_state_or_question() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.post("/crew/step", json={})
        assert r.status_code == 400


def test_crew_stream_emits_sse() -> None:
    app = _make_app()
    with TestClient(app) as client, client.stream(
        "POST", "/crew/stream", json={"question": "hi"}
    ) as response:
        assert response.status_code == 200
        chunks = list(response.iter_lines())
        assert any("data:" in c for c in chunks)


def test_session_ask_creates_then_persists() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        app = _make_app(storage_dir=d)
        with TestClient(app) as client:
            r = client.post(
                "/sessions/sess1/ask", json={"question": "first"}
            )
            assert r.status_code == 200
            assert r.json()["turn_number"] == 1
            r2 = client.post(
                "/sessions/sess1/ask", json={"question": "second"}
            )
            assert r2.status_code == 200
            assert r2.json()["turn_number"] == 2
            h = client.get("/sessions/sess1")
            assert h.status_code == 200
            assert len(h.json()["turns"]) == 2
            d_res = client.delete("/sessions/sess1")
            assert d_res.status_code == 200


def test_list_sessions_empty() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/sessions")
        assert r.status_code == 200
        assert r.json()["sessions"] == []
