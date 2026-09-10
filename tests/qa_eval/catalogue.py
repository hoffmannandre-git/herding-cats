"""Load and run the YAML question catalogue against an in-process crew."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qa_eval.answer_judge import JudgeResult, judge_answer

DEFAULT_CATALOGUE = Path(__file__).resolve().parent.parent / "question_catalogue.yaml"


@dataclass(frozen=True)
class CatalogueQuestion:
    id: str
    retrieval: str
    question: str
    expected_facts: tuple[str, ...]
    min_fact_hits: int = 2
    sources: tuple[str, ...] = ()


@dataclass
class CatalogueRowResult:
    question: CatalogueQuestion
    answer: str
    elapsed_s: float
    judge: JudgeResult


@dataclass
class CatalogueRunResult:
    rows: list[CatalogueRowResult] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        return any(not r.judge.passed for r in self.rows)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.rows if r.judge.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.rows if not r.judge.passed)


def load_catalogue(path: Path | None = None) -> list[CatalogueQuestion]:
    """Parse ``question_catalogue.yaml`` into typed rows."""
    path = path or DEFAULT_CATALOGUE
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows: list[CatalogueQuestion] = []
    for item in raw.get("questions") or []:
        facts = item.get("expected_facts") or []
        sources = item.get("sources") or []
        rows.append(
            CatalogueQuestion(
                id=str(item["id"]),
                retrieval=str(item.get("retrieval", "direct")),
                question=str(item["question"]),
                expected_facts=tuple(str(f) for f in facts),
                min_fact_hits=int(item.get("min_fact_hits", 2)),
                sources=tuple(str(s) for s in sources),
            )
        )
    return rows


def run_catalogue(
    *,
    ask: Callable[[str], str],
    questions: list[CatalogueQuestion] | None = None,
    catalogue_path: Path | None = None,
    use_llm_fallback: bool = False,
    retrieval_filter: set[str] | None = None,
    ollama: Any | None = None,
    judge_model: str | None = None,
) -> CatalogueRunResult:
    """Ask each catalogue question and facts-first judge the answers.

    ``ask`` is typically ``CrewRunner(crew).run``. HTTP adapters can
    wrap ``/crew/run`` the same way later.

    When ``use_llm_fallback`` is True, non-trivial rows that miss the
    fact bar are re-graded by an Ollama JSON judge (``ollama`` client
    optional; a default ``OllamaClient`` is created if omitted).
    """
    qs = questions if questions is not None else load_catalogue(catalogue_path)
    if retrieval_filter is not None:
        qs = [q for q in qs if q.retrieval in retrieval_filter]

    out = CatalogueRunResult()
    for q in qs:
        t0 = time.perf_counter()
        answer = ask(q.question)
        elapsed = time.perf_counter() - t0
        # Trivial rows stay facts-only even when LLM fallback is enabled.
        llm = use_llm_fallback and q.retrieval != "trivial"
        judge = judge_answer(
            question=q.question,
            answer=answer or "",
            expected_facts=q.expected_facts,
            min_fact_hits=q.min_fact_hits,
            use_llm_fallback=llm,
            ollama=ollama,
            judge_model=judge_model,
        )
        out.rows.append(
            CatalogueRowResult(question=q, answer=answer or "", elapsed_s=elapsed, judge=judge)
        )
    return out
