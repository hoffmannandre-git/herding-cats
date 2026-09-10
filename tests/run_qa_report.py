"""CLI: run the QA catalogue against a live Ollama-backed crew.

Usage (from repo root, with editable/dev install + Ollama up)::

    py tests/run_qa_report.py
    py tests/run_qa_report.py --llm-judge

Exit codes: 0 = all passed, 1 = some failed, 2 = setup error.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = Path(__file__).resolve().parent
for p in (SRC, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from herding_cats import Crew, CrewRunner, OllamaClient  # noqa: E402
from herding_cats.helpers.env import get_env  # noqa: E402
from herding_cats.tools_filesystem import filesystem_tools  # noqa: E402
from qa_eval.catalogue import load_catalogue, run_catalogue  # noqa: E402
from qa_eval.report import write_report  # noqa: E402


def _data_root() -> Path:
    env = os.environ.get("HERDING_CATS_DATA_DIR") or os.environ.get("LOCALCREW_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "data").resolve()


def _llm_judge_enabled(cli_flag: bool) -> bool:
    if cli_flag:
        return True
    raw = (get_env("QA_LLM_JUDGE") or os.environ.get("HERDING_CATS_QA_LLM_JUDGE") or "").strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run herding-cats QA catalogue report")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="When facts-first misses, ask Ollama for a JSON answered/reason verdict "
        "(non-trivial rows only). Also set HERDING_CATS_QA_LLM_JUDGE=1.",
    )
    args = parser.parse_args(argv)

    data = _data_root()
    if not data.exists():
        print(f"data directory missing: {data}", file=sys.stderr)
        return 2

    client = OllamaClient(timeout_s=float(get_env("OLLAMA_TIMEOUT") or "600"))
    if not client.health():
        print(f"Ollama not reachable at {client.base_url}", file=sys.stderr)
        return 2

    tools = filesystem_tools(root=data)
    crew = Crew(ollama=client, tool_specs=tools)
    runner = CrewRunner(crew)

    use_llm = _llm_judge_enabled(args.llm_judge)
    questions = load_catalogue()
    print(f"Running {len(questions)} catalogue questions (llm_judge={use_llm})…")
    result = run_catalogue(
        ask=runner.run,
        questions=questions,
        use_llm_fallback=use_llm,
        ollama=client,
    )
    md_path, json_path = write_report(result)
    print(f"Report: {md_path}")
    print(f"Timing: {json_path}")
    print(f"Passed {result.passed_count}/{len(result.rows)}")
    return 1 if result.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
