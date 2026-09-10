"""Clean-room QA catalogue eval helpers (facts-first judge + report).

Inspired by the *pattern* of an office Alfred QA catalogue — not a
copy of that code. Keep separate from ``test_grounding.py`` (which
owns forbidden / anti-hallucination claims).
"""

from __future__ import annotations

from qa_eval.answer_judge import JudgeResult, count_facts_in_answer, judge_answer
from qa_eval.catalogue import CatalogueQuestion, load_catalogue, run_catalogue
from qa_eval.report import write_report

__all__ = [
    "CatalogueQuestion",
    "JudgeResult",
    "count_facts_in_answer",
    "judge_answer",
    "load_catalogue",
    "run_catalogue",
    "write_report",
]
