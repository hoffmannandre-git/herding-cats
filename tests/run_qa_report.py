"""CLI: run the QA catalogue against a live Ollama-backed crew.

Usage (from repo root, with editable/dev install + Ollama up)::

    py tests/run_qa_report.py

Exit codes: 0 = all passed, 1 = some failed, 2 = setup error.
"""

from __future__ import annotations

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
from herding_cats.tools_filesystem import filesystem_tools  # noqa: E402
from qa_eval.catalogue import load_catalogue, run_catalogue  # noqa: E402
from qa_eval.report import write_report  # noqa: E402


def _data_root() -> Path:
    env = os.environ.get("HERDING_CATS_DATA_DIR") or os.environ.get("LOCALCREW_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "data").resolve()


def main() -> int:
    data = _data_root()
    if not data.exists():
        print(f"data directory missing: {data}", file=sys.stderr)
        return 2

    client = OllamaClient()
    if not client.health():
        print(f"Ollama not reachable at {client.base_url}", file=sys.stderr)
        return 2

    tools = filesystem_tools(root=data)
    crew = Crew(ollama=client, tool_specs=tools)
    runner = CrewRunner(crew)

    questions = load_catalogue()
    print(f"Running {len(questions)} catalogue questions…")
    result = run_catalogue(ask=runner.run, questions=questions, use_llm_fallback=False)
    md_path, json_path = write_report(result)
    print(f"Report: {md_path}")
    print(f"Timing: {json_path}")
    print(f"Passed {result.passed_count}/{len(result.rows)}")
    return 1 if result.any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
