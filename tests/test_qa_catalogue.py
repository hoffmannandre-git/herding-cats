"""QA catalogue tests: facts-first judge (fast) + full crew report (slow).

The catalogue is an **eval/report** harness. It does not replace
``test_grounding.py`` (forbidden / anti-hallucination claims stay there).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qa_eval.answer_judge import count_facts_in_answer, judge_answer
from qa_eval.catalogue import load_catalogue, run_catalogue
from qa_eval.report import write_report

CATALOGUE = Path(__file__).resolve().parent / "question_catalogue.yaml"


# ----- fast: judge + loader ----------------------------------------------


def test_load_catalogue_has_rows() -> None:
    rows = load_catalogue(CATALOGUE)
    assert len(rows) >= 5
    assert {r.retrieval for r in rows} >= {"trivial", "direct", "implied"}
    assert all(r.expected_facts for r in rows)


def test_facts_first_pass() -> None:
    result = judge_answer(
        question="q",
        answer="Cobble is a 12-key macropad with e-ink for $89.",
        expected_facts=["macropad", "12-key", "$89", "Rust"],
        min_fact_hits=2,
    )
    assert result.passed
    assert "macropad" in result.facts_hit


def test_facts_first_fail() -> None:
    result = judge_answer(
        question="q",
        answer="It is a nice gadget.",
        expected_facts=["macropad", "12-key"],
        min_fact_hits=2,
    )
    assert not result.passed
    assert result.facts_hit == ()


def test_count_facts_normalizes_comma_decimal() -> None:
    hits = count_facts_in_answer("costs $6,99 monthly", ["$6.99"])
    assert hits == ["$6.99"]


def test_llm_fallback_pass_with_stub() -> None:
    class _Stub:
        def health(self) -> bool:
            return True

        def chat(self, req):
            from herding_cats.ollama import ChatMessage, ChatResponse

            return ChatResponse(
                model=req.model,
                message=ChatMessage(
                    role="assistant",
                    content='{"answered": true, "reason": "covers macropad clearly"}',
                ),
                done=True,
            )

    result = judge_answer(
        question="What is Cobble?",
        answer="It is a compact keyboard accessory.",
        expected_facts=["macropad", "12-key"],
        min_fact_hits=2,
        use_llm_fallback=True,
        ollama=_Stub(),
    )
    assert result.passed
    assert result.mode == "llm-judge"
    assert "covers macropad" in result.reason


def test_llm_fallback_fail_with_stub() -> None:
    class _Stub:
        def health(self) -> bool:
            return True

        def chat(self, req):
            from herding_cats.ollama import ChatMessage, ChatResponse

            return ChatResponse(
                model=req.model,
                message=ChatMessage(
                    role="assistant",
                    content='{"answered": false, "reason": "no concrete facts"}',
                ),
                done=True,
            )

    result = judge_answer(
        question="What is Cobble?",
        answer="Something vague.",
        expected_facts=["macropad"],
        min_fact_hits=1,
        use_llm_fallback=True,
        ollama=_Stub(),
    )
    assert not result.passed
    assert result.mode == "llm-judge"


def test_llm_fallback_unreachable() -> None:
    class _Down:
        def health(self) -> bool:
            return False

        def chat(self, req):
            raise AssertionError("chat should not be called")

    result = judge_answer(
        question="q",
        answer="no facts",
        expected_facts=["macropad"],
        min_fact_hits=1,
        use_llm_fallback=True,
        ollama=_Down(),
    )
    assert not result.passed
    assert "unreachable" in result.reason


def test_run_catalogue_with_stub_ask(tmp_path: Path) -> None:
    rows = [r for r in load_catalogue(CATALOGUE) if r.id == "smoke_cobble_name"]
    assert len(rows) == 1

    def ask(_q: str) -> str:
        return "Cobble is a macropad."

    result = run_catalogue(ask=ask, questions=rows, use_llm_fallback=False)
    assert not result.any_failed
    md, js = write_report(result, report_dir=tmp_path)
    assert md.exists() and js.exists()
    assert "PASS" in md.read_text(encoding="utf-8")


# ----- slow: live crew ---------------------------------------------------


def _real_ollama_or_skip():
    from herding_cats import OllamaClient

    url = os.environ.get("HERDING_CATS_OLLAMA_URL") or os.environ.get(
        "LOCALCREW_OLLAMA_URL", "http://localhost:11434"
    )
    client = OllamaClient(base_url=url)
    if not client.health():
        pytest.skip(f"Ollama not reachable at {url}")
    return client


@pytest.mark.slow
def test_qa_catalogue_smoke_trivial(tmp_path: Path) -> None:
    """One trivial row through the live crew — facts-first only."""
    from herding_cats import Crew, CrewRunner
    from herding_cats.tools_filesystem import filesystem_tools

    data = Path(__file__).resolve().parent.parent / "data"
    if not data.exists():
        pytest.skip(f"data directory not found: {data}")

    ollama = _real_ollama_or_skip()
    crew = Crew(ollama=ollama, tool_specs=filesystem_tools(root=data))
    runner = CrewRunner(crew)
    rows = [r for r in load_catalogue(CATALOGUE) if r.retrieval == "trivial"]
    result = run_catalogue(ask=runner.run, questions=rows, use_llm_fallback=False)
    write_report(result, report_dir=tmp_path)
    assert not result.any_failed, f"trivial smoke failed: {[r.judge.reason for r in result.rows]}"


@pytest.mark.slow
def test_qa_catalogue_full_report(tmp_path: Path) -> None:
    """Full catalogue against live Ollama; LLM judge on fact misses."""
    from herding_cats import Crew, CrewRunner
    from herding_cats.tools_filesystem import filesystem_tools

    data = Path(__file__).resolve().parent.parent / "data"
    if not data.exists():
        pytest.skip(f"data directory not found: {data}")

    ollama = _real_ollama_or_skip()
    crew = Crew(ollama=ollama, tool_specs=filesystem_tools(root=data))
    runner = CrewRunner(crew)
    use_llm = os.environ.get("HERDING_CATS_QA_LLM_JUDGE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    result = run_catalogue(
        ask=runner.run,
        catalogue_path=CATALOGUE,
        use_llm_fallback=use_llm,
        ollama=ollama,
    )
    md, _js = write_report(result, report_dir=tmp_path)
    assert md.exists()
    assert not result.any_failed, (
        f"{result.failed_count} catalogue failures — see {md}\n"
        + "\n".join(f"{r.question.id}: {r.judge.reason}" for r in result.rows if not r.judge.passed)
    )
