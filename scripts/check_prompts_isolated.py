"""Verify prompts are isolated as .md files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    prompts_dir = ROOT / "src" / "herding_cats" / "prompts"
    if not prompts_dir.exists():
        print(f"FAIL: {prompts_dir} does not exist")
        return 1

    print("Prompts present:")
    for f in sorted(prompts_dir.glob("*.md")):
        print(f"  {f.name:20} {f.stat().st_size:5} bytes")

    # Scan .py files for multi-line role-prompt strings.
    pattern = re.compile(
        r'"""[^"]*You are the (Orchestrator|Thinker|Fetcher|Finalist)[^"]*"""',
        re.DOTALL,
    )
    leaks = []
    for py in (ROOT / "src").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            leaks.append((str(py.relative_to(ROOT)), m.group()[:80]))
    if leaks:
        print("\nLEAKS DETECTED:")
        for f, snippet in leaks:
            print(f"  {f}: {snippet!r}")
        return 1
    print("\nOK — no role-prompt strings in any .py file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
