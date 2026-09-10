"""Tests for the OllamaClient.

Network-free tests use `respx` to mock `httpx`. Live tests are marked `slow`.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from herding_cats.ollama import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    OllamaClient,
)


def test_health_short_circuits_when_down() -> None:
    client = OllamaClient(base_url="http://localhost:1", timeout_s=0.5)
    assert client.health() is False


def test_chat_returns_typed_response() -> None:
    with respx.mock(base_url="http://mock-ollama") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "llama3.1:8b",
                    "message": {"role": "assistant", "content": "hello"},
                    "done": True,
                    "total_duration": 123456789,
                },
            )
        )
        client = OllamaClient(base_url="http://mock-ollama")
        resp = client.chat(
            ChatRequest(
                model="llama3.1:8b",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )
        assert resp.message.content == "hello"
        assert resp.total_duration_ns == 123456789


def test_chat_raises_on_http_error() -> None:
    with respx.mock(base_url="http://mock-ollama") as mock:
        mock.post("/api/chat").mock(return_value=httpx.Response(500, text="boom"))
        client = OllamaClient(base_url="http://mock-ollama")
        from herding_cats.ollama import OllamaError
        with pytest.raises(OllamaError):
            client.chat(
                ChatRequest(
                    model="llama3.1:8b",
                    messages=[ChatMessage(role="user", content="hi")],
                )
            )


def test_embed_returns_vectors() -> None:
    with respx.mock(base_url="http://mock-ollama") as mock:
        mock.post("/api/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"embedding": [[0.1, 0.2, 0.3]]},
            )
        )
        client = OllamaClient(base_url="http://mock-ollama")
        resp = client.embed(EmbeddingRequest(model="nomic-embed-text", prompt="hi"))
        assert resp.embeddings == [[0.1, 0.2, 0.3]]


def test_streaming_yields_deltas() -> None:
    import json

    chunks = [
        json.dumps({"message": {"role": "assistant", "content": "hel"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "lo"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    body = "\n".join(chunks).encode("utf-8")

    with respx.mock(base_url="http://mock-ollama") as mock:
        mock.post("/api/chat").mock(return_value=httpx.Response(200, content=body))
        client = OllamaClient(base_url="http://mock-ollama")
        req = ChatRequest(
            model="llama3.1:8b",
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
        )
        tokens = list(client.chat_stream(req))
        assert tokens == ["hel", "lo"]


@pytest.mark.slow
def test_list_models_live(skip_if_no_ollama) -> None:
    client = OllamaClient()
    models = client.list_models()
    assert isinstance(models, list)
