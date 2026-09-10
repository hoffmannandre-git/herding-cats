"""Facts-first answer judge for the QA catalogue, with optional Ollama fallback.

Pass if enough ``expected_facts`` appear as substrings in the answer
(after light normalization). If that misses and ``use_llm_fallback`` is
True, ask a separate Ollama model for a JSON ``{"answered", "reason"}``
verdict. This is an **eval aid**, not a claim of hallucination tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from herding_cats.helpers.env import get_env
from herding_cats.helpers.json_util import extract_json_object


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    reason: str
    facts_hit: tuple[str, ...]
    mode: str = "facts-first"  # "facts-first" | "llm-judge" | "facts-first-fail"


class ChatCapable(Protocol):
    """Minimal Ollama surface the LLM judge needs."""

    def health(self) -> bool: ...

    def chat(self, req: Any) -> Any: ...


def _normalize(text: str) -> str:
    return text.lower().replace(",", ".")


def count_facts_in_answer(answer: str, expected_facts: list[str] | tuple[str, ...]) -> list[str]:
    """Return the expected facts that appear as substrings in ``answer``."""
    hay = _normalize(answer or "")
    hits: list[str] = []
    for fact in expected_facts:
        needle = _normalize(fact)
        if needle and needle in hay:
            hits.append(fact)
    return hits


def judge_model_name() -> str:
    """Model used for the optional LLM judge (not necessarily the crew model)."""
    return get_env("JUDGE_MODEL") or get_env("MODEL") or "llama3.1:8b"


def _coerce_answered(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return None


def llm_judge_answer(
    *,
    question: str,
    answer: str,
    expected_facts: list[str] | tuple[str, ...],
    min_fact_hits: int,
    ollama: ChatCapable,
    model: str | None = None,
) -> JudgeResult:
    """Ask Ollama whether the answer adequately covers the expected facts.

    Expects JSON: ``{"answered": bool, "reason": str}``.
    """
    from herding_cats.ollama import ChatMessage, ChatRequest

    if not ollama.health():
        hits = count_facts_in_answer(answer, expected_facts)
        return JudgeResult(
            passed=False,
            reason="llm-judge: Ollama unreachable",
            facts_hit=tuple(hits),
            mode="llm-judge",
        )

    facts_block = "\n".join(f"- {fact}" for fact in expected_facts)
    system = (
        "You are a strict test grader. "
        "Reply with a single JSON object only — no markdown, no prose."
    )
    user = f"""QUESTION:
{question.strip()}

ANSWER:
{(answer or "").strip()}

EXPECTED FACTS (tolerate spelling / hyphenation / plural variants):
{facts_block}

Set "answered" to true only if the ANSWER concretely covers at least
{min_fact_hits} of the EXPECTED FACTS (paraphrase OK if the same fact
is clearly present) AND the ANSWER is a relevant reply to the QUESTION.

JSON shape:
{{"answered": true or false, "reason": "one short sentence"}}"""

    req = ChatRequest(
        model=model or judge_model_name(),
        messages=[
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ],
        stream=False,
        format="json",
        options={"temperature": 0},
    )
    try:
        resp = ollama.chat(req)
    except Exception as exc:
        hits = count_facts_in_answer(answer, expected_facts)
        return JudgeResult(
            passed=False,
            reason=f"llm-judge: call failed ({type(exc).__name__}: {exc})",
            facts_hit=tuple(hits),
            mode="llm-judge",
        )

    content = (resp.message.content if resp and resp.message else "") or ""
    hits = count_facts_in_answer(answer, expected_facts)
    if not content.strip():
        return JudgeResult(
            passed=False,
            reason="llm-judge: empty model content",
            facts_hit=tuple(hits),
            mode="llm-judge",
        )

    parsed = extract_json_object(content)
    if not parsed:
        return JudgeResult(
            passed=False,
            reason=f"llm-judge: JSON parse failed: {content[:200]!r}",
            facts_hit=tuple(hits),
            mode="llm-judge",
        )

    answered = _coerce_answered(parsed.get("answered", parsed.get("passes")))
    reason = str(parsed.get("reason", "")).strip() or "no reason"
    if answered is None:
        return JudgeResult(
            passed=False,
            reason=f"llm-judge: missing answered field ({reason})",
            facts_hit=tuple(hits),
            mode="llm-judge",
        )
    return JudgeResult(
        passed=bool(answered),
        reason=f"llm-judge: {reason}",
        facts_hit=tuple(hits),
        mode="llm-judge",
    )


def judge_answer(
    *,
    question: str,
    answer: str,
    expected_facts: list[str] | tuple[str, ...],
    min_fact_hits: int = 2,
    use_llm_fallback: bool = False,
    ollama: ChatCapable | None = None,
    judge_model: str | None = None,
) -> JudgeResult:
    """Grade one catalogue row: facts-first, then optional LLM fallback."""
    hits = count_facts_in_answer(answer, expected_facts)
    if len(hits) >= min_fact_hits:
        return JudgeResult(
            passed=True,
            reason=f"facts-first: {len(hits)}/{min_fact_hits} hits ({', '.join(hits)})",
            facts_hit=tuple(hits),
            mode="facts-first",
        )

    if use_llm_fallback:
        client = ollama
        if client is None:
            from herding_cats import OllamaClient

            client = OllamaClient()
        return llm_judge_answer(
            question=question,
            answer=answer,
            expected_facts=expected_facts,
            min_fact_hits=min_fact_hits,
            ollama=client,
            model=judge_model,
        )

    missing = [f for f in expected_facts if f not in hits]
    return JudgeResult(
        passed=False,
        reason=(
            f"facts-first: {len(hits)}/{min_fact_hits} hits; "
            f"missing={missing!r}"
        ),
        facts_hit=tuple(hits),
        mode="facts-first-fail",
    )
