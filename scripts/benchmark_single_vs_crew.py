#!/usr/bin/env python3
"""Benchmark: single Ollama chat vs full herding-cats crew.

Honest worksheet — not a leaderboard. Re-run on your machine; times
vary by model, CPU/GPU, and cold vs warm Ollama.

Usage (Ollama must be reachable):

    docker compose --profile api up -d
    python scripts/benchmark_single_vs_crew.py

Env:

    HERDING_CATS_OLLAMA_URL   default http://localhost:11434
    HERDING_CATS_MODEL        default llama3.1:8b
    HERDING_CATS_DATA_DIR     default ./data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herding_cats import (  # noqa: E402
    ChatMessage,
    ChatRequest,
    Crew,
    CrewRunner,
    OllamaClient,
)
from herding_cats.helpers.env import get_env  # noqa: E402
from herding_cats.tools_filesystem import filesystem_tools  # noqa: E402

# Cobble claims from tests/test_grounding.py (keep in sync by hand).
QUESTION = "What is Cobble and how is it different from a Stream Deck?"
GROUNDED = ("macropad", "12-key", "e-ink", "$89", "Rust")
FORBIDDEN = ("Stream Deck Plus", "LCD keys", "icons", "Elgato", "plugins")


def _present(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower().replace(",", ".")


def _score(answer: str) -> dict[str, float | list[str]]:
    g_hits = [c for c in GROUNDED if _present(answer, c)]
    f_hits = [c for c in FORBIDDEN if _present(answer, c)]
    return {
        "grounded_recall": len(g_hits) / len(GROUNDED),
        "grounded_hits": g_hits,
        "forbidden_hits": f_hits,
        "forbidden_hit_rate": 1.0 - (len(f_hits) / len(FORBIDDEN)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default=QUESTION,
        help="question to ask both modes",
    )
    args = parser.parse_args()

    client = OllamaClient()
    if not client.health():
        print("Ollama not reachable. Start the stack first:", file=sys.stderr)
        print("  docker compose --profile api up -d", file=sys.stderr)
        return 2

    model = get_env("MODEL") or "llama3.1:8b"
    data = Path(get_env("DATA_DIR") or (ROOT / "data")).resolve()
    cobble_path = data / "cobble.md"
    cobble = cobble_path.read_text(encoding="utf-8") if cobble_path.exists() else ""

    print(f"model={model}  ollama={client.base_url}  data={data}")
    print(f"question={args.question!r}\n")

    # --- single-agent: one chat, file dumped into the prompt -----------------
    single_prompt = (
        "Answer the user using ONLY the brief below. "
        "If something is not in the brief, say you do not know.\n\n"
        f"BRIEF:\n{cobble}\n\nQUESTION:\n{args.question}"
    )
    t0 = time.perf_counter()
    single = client.chat(
        ChatRequest(
            model=model,
            messages=[ChatMessage(role="user", content=single_prompt)],
            stream=False,
        )
    )
    single_s = time.perf_counter() - t0
    single_ans = single.message.content if single.message else ""
    single_score = _score(single_ans)

    # --- crew: filesystem tools + full loop ---------------------------------
    crew = Crew(
        ollama=client,
        tool_specs=filesystem_tools(root=data),
        model=model,
    )
    t0 = time.perf_counter()
    crew_ans = CrewRunner(crew).run(args.question)
    crew_s = time.perf_counter() - t0
    crew_score = _score(crew_ans)

    def _row(name: str, seconds: float, answer: str, score: dict) -> None:
        print(f"=== {name} ===")
        print(f"wall_sec={seconds:.2f}")
        print(
            f"grounded_recall={score['grounded_recall']:.0%} "
            f"hits={score['grounded_hits']}"
        )
        print(
            f"forbidden_clean={score['forbidden_hit_rate']:.0%} "
            f"(forbidden_hits={score['forbidden_hits']})"
        )
        preview = answer[:240].replace("\n", " ")
        print(f"answer_preview={preview!r}...\n")

    _row(
        "single-agent (1x chat + brief in prompt)",
        single_s,
        single_ans,
        single_score,
    )
    _row(
        "crew (orchestrator -> final + ls/cat)",
        crew_s,
        crew_ans,
        crew_score,
    )

    print("--- summary ---")
    ratio = crew_s / max(single_s, 1e-6)
    print(f"single_sec={single_s:.2f}  crew_sec={crew_s:.2f}  ratio={ratio:.2f}x")
    print(
        "Note: crew usually costs more wall time (multiple LLM calls) "
        "and should ground via tools rather than a pasted brief."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
