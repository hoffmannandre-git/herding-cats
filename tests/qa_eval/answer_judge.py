"""Facts-first answer judge for the QA catalogue.

Pass if enough ``expected_facts`` appear as substrings in the answer
(after light normalization). An optional Ollama LLM fallback is stubbed
for a later slice — leave ``use_llm_fallback=False`` until then.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    reason: str
    facts_hit: tuple[str, ...]


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


def judge_answer(
    *,
    question: str,
    answer: str,
    expected_facts: list[str] | tuple[str, ...],
    min_fact_hits: int = 2,
    use_llm_fallback: bool = False,
) -> JudgeResult:
    """Grade one catalogue row.

    Parameters
    ----------
    question:
        Unused for facts-first grading; reserved for the LLM judge.
    answer:
        Model / crew output to grade.
    expected_facts:
        Substrings that should appear in a grounded answer.
    min_fact_hits:
        Minimum number of distinct fact hits required to pass.
    use_llm_fallback:
        If True and facts miss the bar, call the optional Ollama judge
        (not implemented in slice 1 — raises if enabled).
    """
    _ = question  # reserved for LLM judge prompt
    hits = count_facts_in_answer(answer, expected_facts)
    if len(hits) >= min_fact_hits:
        return JudgeResult(
            passed=True,
            reason=f"facts-first: {len(hits)}/{min_fact_hits} hits ({', '.join(hits)})",
            facts_hit=tuple(hits),
        )

    if use_llm_fallback:
        raise NotImplementedError(
            "LLM judge fallback is not implemented yet; "
            "run with use_llm_fallback=False (facts-first only)."
        )

    missing = [f for f in expected_facts if f not in hits]
    return JudgeResult(
        passed=False,
        reason=(
            f"facts-first: {len(hits)}/{min_fact_hits} hits; "
            f"missing={missing!r}"
        ),
        facts_hit=tuple(hits),
    )
