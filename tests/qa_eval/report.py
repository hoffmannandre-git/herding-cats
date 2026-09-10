"""Write Markdown (+ timing JSON) reports for a catalogue run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from qa_eval.catalogue import CatalogueRunResult

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def write_report(
    result: CatalogueRunResult,
    *,
    report_dir: Path | None = None,
    title: str = "herding-cats QA catalogue",
) -> tuple[Path, Path]:
    """Write ``qa-report-<ts>.md`` + ``.json`` and refresh ``qa-report-latest.*``.

    Returns ``(md_path, json_path)`` for the timestamped files.
    """
    report_dir = report_dir or DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = report_dir / f"qa-report-{ts}.md"
    json_path = report_dir / f"qa-report-{ts}.json"

    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated (UTC): `{ts}`",
        "",
        f"Passed: **{result.passed_count}** / {len(result.rows)}  "
        f"(failed: {result.failed_count})",
        "",
        "| id | retrieval | ok | hits | time_s | reason |",
        "|---|---|---|---|---|---|",
    ]
    timing: list[dict] = []
    for row in result.rows:
        q = row.question
        j = row.judge
        ok = "PASS" if j.passed else "FAIL"
        hits = f"{len(j.facts_hit)}/{q.min_fact_hits}"
        reason = j.reason.replace("|", "\\|")
        lines.append(
            f"| `{q.id}` | {q.retrieval} | {ok} | {hits} | "
            f"{row.elapsed_s:.2f} | {reason} |"
        )
        timing.append(
            {
                "id": q.id,
                "retrieval": q.retrieval,
                "passed": j.passed,
                "facts_hit": list(j.facts_hit),
                "min_fact_hits": q.min_fact_hits,
                "elapsed_s": round(row.elapsed_s, 3),
                "reason": j.reason,
                "answer": row.answer,
                "question": q.question,
            }
        )
        lines.extend(
            [
                "",
                f"## `{q.id}` — {ok}",
                "",
                f"**Q:** {q.question}",
                "",
                f"**Judge:** {j.reason}",
                "",
                "<details><summary>Answer</summary>",
                "",
                "```",
                row.answer.strip() or "(empty)",
                "```",
                "",
                "</details>",
            ]
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "title": title,
        "generated_utc": ts,
        "passed": result.passed_count,
        "failed": result.failed_count,
        "rows": timing,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    (report_dir / "qa-report-latest.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (report_dir / "qa-report-latest.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return md_path, json_path
