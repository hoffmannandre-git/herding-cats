"""Thin, typed wrapper around the Ollama HTTP API.

This is intentionally small. We cover the four endpoints the crew actually
uses (chat, generate is not used, embed, list models for health checks) and
we surface JSON-mode and streaming because both matter for the crew runtime.

If you need full coverage, use `ollama-python` instead. We don't aim to
replace it — we aim to be the smallest wrapper that lets the crew be tested.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


class OllamaError(RuntimeError):
    """Raised when an Ollama HTTP call fails or returns malformed JSON."""


class ChatMessage(BaseModel):
    """One message in a chat history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Optional: when role == "tool", the tool name and call id.
    tool_name: str | None = None


class ChatRequest(BaseModel):
    """A /chat/completions-equivalent request body.

    The fields below are the subset the crew uses; passthrough extras go
    through `options`.
    """

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    format: Literal["", "json"] = ""
    options: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """A non-streamed chat response.

    Timing fields follow Ollama's wire format (already in nanoseconds when
    suffixed `_duration`); token counts are integer counts.
    """

    model: str
    message: ChatMessage
    done: bool = True
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    @classmethod
    def from_ollama(cls, payload: dict[str, Any]) -> ChatResponse:
        """Build a ChatResponse from Ollama's wire dict, mapping fields."""
        # Ollama's wire names are e.g. `total_duration` (already ns), `eval_count`.
        # We expose them with `_ns` suffix where they are nanosecond durations
        # to make the unit explicit at the call site.
        rename = {
            "total_duration": "total_duration_ns",
            "load_duration": "load_duration_ns",
            "prompt_eval_duration": "prompt_eval_duration_ns",
            "eval_duration": "eval_duration_ns",
        }
        body = dict(payload)
        for old, new in rename.items():
            if old in body and new not in body:
                body[new] = body.pop(old)
        return cls.model_validate(body)


class EmbeddingRequest(BaseModel):
    model: str
    prompt: str | list[str]


class EmbeddingResponse(BaseModel):
    model: str = ""
    embeddings: list[list[float]] = Field(default_factory=list)

    @classmethod
    def from_ollama(cls, payload: dict[str, Any]) -> EmbeddingResponse:
        """Normalize Ollama's `embedding` (singular) / `embeddings` (plural) shapes.

        Ollama returns `{"embedding": [...]}` for a single prompt and
        `{"embeddings": [[...], ...]}` for a batch. We accept both and
        normalize to `embeddings`.
        """
        if "embeddings" in payload and isinstance(payload["embeddings"], list):
            vectors = payload["embeddings"]
        elif "embedding" in payload and isinstance(payload["embedding"], list):
            vec = payload["embedding"]
            vectors = [vec] if vec and isinstance(vec[0], (int, float)) else vec
        else:
            vectors = []
        return cls(model=str(payload.get("model", "")), embeddings=vectors)


class OllamaClient:
    """Synchronous + asynchronous client for the Ollama HTTP API.

    Example:
        >>> client = OllamaClient()
        >>> client.health()
        True
        >>> client.chat(ChatRequest(
        ...     model="llama3.1:8b",
        ...     messages=[ChatMessage(role="user", content="hi")],
        ... ))
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Env lookup order: HERDING_CATS_OLLAMA_URL -> LOCALCREW_OLLAMA_URL
        # (legacy) -> OLLAMA_BASE_URL (older still) -> default.
        from herding_cats.helpers.env import get_env

        self.base_url = (
            base_url
            or get_env("OLLAMA_URL")
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)
        self._async_client = async_client or httpx.AsyncClient(timeout=timeout_s)

    # ----- Health & discovery ------------------------------------------------

    def health(self) -> bool:
        """Return True if /api/tags responds (cheap, no model load)."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def ahealth(self) -> bool:
        try:
            r = await self._async_client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        r = self._client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]

    # ----- Chat (sync) -------------------------------------------------------

    def chat(self, req: ChatRequest) -> ChatResponse:
        if req.stream:
            raise OllamaError("Use chat_stream() for streaming requests.")
        body = req.model_dump(exclude_none=True)
        r = self._client.post(f"{self.base_url}/api/chat", json=body)
        if r.status_code != 200:
            raise OllamaError(f"Ollama /api/chat {r.status_code}: {r.text[:500]}")
        return ChatResponse.from_ollama(r.json())

    def chat_stream(self, req: ChatRequest) -> Iterator[str]:
        """Yield token deltas from a streamed /api/chat response."""
        if not req.stream:
            req = req.model_copy(update={"stream": True})
        body = req.model_dump(exclude_none=True)
        with self._client.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
            if r.status_code != 200:
                raise OllamaError(f"Ollama /api/chat {r.status_code}: {r.text[:500]}")
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    import json as _json

                    chunk = _json.loads(line)
                except Exception as exc:  # pragma: no cover
                    raise OllamaError(f"Bad stream chunk: {line!r}") from exc
                msg = chunk.get("message") or {}
                delta = msg.get("content") or ""
                if delta:
                    yield delta
                if chunk.get("done"):
                    return

    # ----- Chat (async) ------------------------------------------------------

    async def achat(self, req: ChatRequest) -> ChatResponse:
        if req.stream:
            raise OllamaError("Use achat_stream() for streaming requests.")
        body = req.model_dump(exclude_none=True)
        # Fresh client per call so we bind to the *current* event loop.
        # Reusing ``_async_client`` across ``asyncio.run()`` (or across a
        # closed Proactor loop on Windows) raises
        # ``RuntimeError: Event loop is closed`` during response cleanup.
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=body)
        if r.status_code != 200:
            raise OllamaError(f"Ollama /api/chat {r.status_code}: {r.text[:500]}")
        return ChatResponse.from_ollama(r.json())

    async def achat_stream(self, req: ChatRequest) -> AsyncIterator[str]:
        if not req.stream:
            req = req.model_copy(update={"stream": True})
        body = req.model_dump(exclude_none=True)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
                if r.status_code != 200:
                    raise OllamaError(f"Ollama /api/chat {r.status_code}: {r.text[:500]}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json as _json

                        chunk = _json.loads(line)
                    except Exception as exc:  # pragma: no cover
                        raise OllamaError(f"Bad stream chunk: {line!r}") from exc
                    msg = chunk.get("message") or {}
                    delta = msg.get("content") or ""
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        return

    # ----- Embeddings --------------------------------------------------------

    def embed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        r = self._client.post(f"{self.base_url}/api/embeddings", json=req.model_dump())
        r.raise_for_status()
        return EmbeddingResponse.from_ollama(r.json())

    async def aembed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                f"{self.base_url}/api/embeddings", json=req.model_dump()
            )
        r.raise_for_status()
        return EmbeddingResponse.from_ollama(r.json())

    # ----- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._async_client.aclose()
