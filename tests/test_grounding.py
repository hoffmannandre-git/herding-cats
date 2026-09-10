"""Grounding / hallucination tests against a folder of project briefs.

The crew is pointed at ``data/`` (the same mount the docker-compose
uses) and asked questions about five fictional / parody software
products. The test verifies that an answer derived from those files:

* **Recalls** facts that ARE in the files (grounding).
* **Does NOT** hallucinate facts that are explicitly forbidden in the
  files (anti-hallucination).

We assert on individual *claims* rather than on the full text — this
gives the test fine-grained signal and a clear failure mode when the
answer drifts.

Two test layers:

* **Fast tests** (``pytest -m "not slow"``)
    Exercise the *plumbing* — filesystem tools + claim classifier —
    without an LLM. They verify that reading the files via ``ls`` /
    ``cat`` exposes the grounded facts, and that the forbidden phrases
    do not leak into file content (so a well-behaved answer can only
    mention them by mistake).
* **Slow tests** (``pytest -m slow``)
    Run the full crew against a real Ollama instance, with
    ``orchestrator_mode="rules"`` so the loop is deterministic.
    Require a healthy Ollama at ``HERDING_CATS_OLLAMA_URL``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from herding_cats import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Crew,
    CrewRunner,
    OllamaClient,
)
from herding_cats.tools_filesystem import CatInput, LsInput, filesystem_tools

# ----- claim model --------------------------------------------------------


@dataclass(frozen=True)
class GroundingCase:
    """A single question + the claims the answer must / must-not contain."""

    question: str
    file: str  # the file the answer should be drawn from
    grounded: tuple[str, ...]
    forbidden: tuple[str, ...]

    @property
    def id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.question.lower()).strip("_")
        return slug[:40]


# ----- the cases ----------------------------------------------------------


CASES: tuple[GroundingCase, ...] = (
    # Cobble — should NOT be confused with Elgato's Stream Deck.
    GroundingCase(
        question="What is Cobble and how is it different from a Stream Deck?",
        file="cobble.md",
        grounded=(
            "macropad",
            "12-key",
            "e-ink",
            "$89",
            "Rust",
        ),
        forbidden=(
            # Real Stream Deck features that must NOT bleed in.
            "Stream Deck Plus",
            "LCD keys",
            "icons",
            "Elgato",
            "plugins",
        ),
    ),
    # Porthound — NOT a Gitea fork, NOT hosted.
    GroundingCase(
        question="Is Porthound a hosted SaaS product, and how is it licensed?",
        file="porthound.md",
        grounded=(
            "BSD",
            "open-source",
            "Zig",
            "single binary",
        ),
        forbidden=(
            "gitea.io",
            "Gitea fork",
            "porthound.io",
            "Postgres",
        ),
    ),
    # Sleepwell — NOT Calm/Headspace. Pricing must be $6.99.
    GroundingCase(
        question="Who makes Sleepwell and how much does it cost?",
        file="sleepwell.md",
        grounded=(
            "$6.99",
            "Lisbon",
            "AI",
            "bedtime story",
        ),
        forbidden=(
            "celebrity",
            "Headspace",
            "Calm app",
            "free tier",
            "breathing",
        ),
    ),
    # Manuscriptr — NOT a text editor, NOT a Git GUI fork.
    GroundingCase(
        question="Does Manuscriptr replace my text editor?",
        file="manuscriptr.md",
        grounded=(
            "version-control",
            "read-only",
            "novelists",
            "$49",
        ),
        forbidden=(
            "Scrivener alternative",
            "fork",
            "AI",
            "autocomplete",
            "GitKraken",
        ),
    ),
    # Astrolabe — NOT a planetarium, NOT free, NOT a competitor to Stellarium.
    GroundingCase(
        question="Is Astrolabe a planetarium app and what does it cost?",
        file="astrolabe.md",
        grounded=(
            "$1.99",
            "Belgrade",
            "offline",
            "star map",
        ),
        forbidden=(
            "Stellarium competitor",
            "real-time tracking",
            "AR",
            "free",
            "web app",
        ),
    ),
)


# ----- fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def data_root() -> Path:
    """Resolve the data directory the tests should run against.

    Defaults to ``./data`` next to the repo root (the same default the
    docker-compose uses). Override via ``HERDING_CATS_DATA_DIR``.
    """
    env = os.environ.get("HERDING_CATS_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = (Path(__file__).resolve().parent.parent / "data").resolve()
    if not p.exists():
        pytest.skip(f"data directory not found: {p}")
    return p


@pytest.fixture(scope="session")
def tool_specs(data_root: Path):
    return filesystem_tools(root=data_root)


# ----- helpers ------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so claim matching is forgiving."""
    return re.sub(r"\s+", " ", text.lower())


# Regex used to decide whether a forbidden phrase is being asserted or
# negated. If the phrase appears within ~30 chars of a negation cue
# (`not`, `n't`, `no`, `never`, `without`), it's a *safe* occurrence
# — the crew is correctly denying it, not hallucinating it.
_NEGATION_CUES = re.compile(
    r"\b(not|no|never|without|none|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|hasn't|haven't|won't|wouldn't|can't|cannot)\b",
    re.IGNORECASE,
)


def _is_safe_occurrence(text: str, phrase: str) -> bool:
    """True if ``phrase`` appears in ``text`` only in a negated context.

    Uses word-boundary matching for phrases that start/end with word
    characters, plain substring matching otherwise (e.g. ``$89``).
    Tolerates hyphens/spaces between words and simple plurals.
    """
    norm = _normalize(text)
    pn = _normalize(phrase).strip()
    if not pn:
        return True
    starts_word = pn[0].isalnum() or pn[0] == "_"
    ends_word = pn[-1].isalnum() or pn[-1] == "_"
    if starts_word and ends_word:
        parts = re.split(r"[ \-‐‑‒–—―]+", pn)
        connector = r"[ \-‐‑‒–—―]*"
        pattern_body = connector.join(_pluralize(p) for p in parts if p)
        pattern = re.compile(rf"\b{pattern_body}\b")
    else:
        pattern = re.compile(re.escape(pn))
    if not pattern.search(norm):
        return True
    for match in pattern.finditer(norm):
        start, end = match.span()
        window = norm[max(0, start - 30) : min(len(norm), end + 30)]
        if not _NEGATION_CUES.search(window):
            return False
    return True


def _pluralize(token: str) -> str:
    """Return a regex alternation matching ``token`` or its simple plural.

    Handles the regular cases that bit the slow tests:

    * ``story`` → ``stories`` (consonant + ``y`` → ``ies``)
    * ``box`` → ``boxes`` (sibilant → ``es``)
    * ``cat`` → ``cats`` (default → ``s``)

    Not smart enough for every English irregular (``child`` /
    ``children``) — for the claim set we use, this is enough.
    """
    if not token or not token[-1].isalpha():
        return re.escape(token)
    base = re.escape(token)
    if len(token) >= 2 and token[-1] == "y" and token[-2] not in "aeiou":
        # Consonant + y: replace `y` with `ies`. The escaped token
        # ends in `\\y`; match the stem + `ies`.
        stem = re.escape(token[:-1])
        return f"(?:{base}|{stem}ies)"
    if token.endswith(("s", "x", "z", "ch", "sh")):
        return f"(?:{base}|{base}es)"
    return f"(?:{base}|{base}s)"


def _is_present(text: str, phrase: str) -> bool:
    """Substring check for grounded/forbidden presence.

    Tolerates:

    * **Hyphenation** — ``single binary`` matches ``single-binary``.
    * **Pluralization** — ``bedtime story`` matches ``bedtime stories``.

    For phrases that start/end with a non-word character (e.g. ``$89``
    starts with ``$``), word-boundary anchoring doesn't work and we
    fall back to plain substring matching.
    """
    norm = _normalize(text)
    pn = _normalize(phrase).strip()
    if not pn:
        return False

    starts_word = pn[0].isalnum() or pn[0] == "_"
    ends_word = pn[-1].isalnum() or pn[-1] == "_"
    if not (starts_word and ends_word):
        return pn in norm

    # Split the phrase into tokens, allow flexible connectors between
    # them, and allow each token to match its simple plural form.
    parts = re.split(r"[ \-‐‑‒–—―]+", pn)
    connector = r"[ \-‐‑‒–—―]*"
    pattern_body = connector.join(_pluralize(p) for p in parts if p)
    return re.search(rf"\b{pattern_body}\b", norm) is not None


def _classify(answer: str, case: GroundingCase) -> dict[str, object]:
    grounded_hits = [c for c in case.grounded if _is_present(answer, c)]
    grounded_misses = [c for c in case.grounded if c not in grounded_hits]
    # A forbidden claim "hits" only when it appears without a nearby
    # negation cue — i.e. the model asserted it as fact.
    forbidden_hits = [
        c for c in case.forbidden
        if _is_present(answer, c) and not _is_safe_occurrence(answer, c)
    ]
    forbidden_misses = [c for c in case.forbidden if c not in forbidden_hits]

    return {
        "grounded_hits": grounded_hits,
        "grounded_misses": grounded_misses,
        "forbidden_hits": forbidden_hits,
        "forbidden_misses": forbidden_misses,
        "grounded_recall": (
            len(grounded_hits) / max(len(case.grounded), 1)
        ),
        "forbidden_hit_rate": (
            len(forbidden_misses) / max(len(case.forbidden), 1)
        ),
    }


def _answer_via_tools(data_root: Path, tool_specs, file: str) -> str:
    """Build the answer *only* from the file content via the tools.

    This is what a well-grounded crew output looks like: it lists the
    data directory, picks the right file, and reads it. The classifier
    is applied to that read result.
    """
    specs = {s.name: s for s in tool_specs}
    ls = specs["ls"].fn
    cat = specs["cat"].fn

    listing = ls(LsInput(directory=".", pattern="*.md"))
    names = {e.name for e in listing.entries}
    assert file in names, f"{file} missing from data/"

    content = cat(CatInput(path=file)).content
    return f"Files read:\n--- {file} ---\n{content.strip()}\n"


# ----- fast tests: tool pipeline + classifier -----------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_file_contains_grounded_facts(
    case: GroundingCase, data_root: Path
) -> None:
    """Sanity: each brief actually contains the grounded claims.

    If this fails, the test fixture is broken — not the crew. It also
    confirms the negation-aware classifier treats the file itself as
    safe (since each brief lists what the project is *not*, those
    forbidden phrases appear there only under "Not …" context).
    """
    text = (data_root / case.file).read_text(encoding="utf-8")
    missing = [c for c in case.grounded if not _is_present(text, c)]
    assert not missing, (
        f"{case.file} missing grounded facts: {missing}"
    )

    # The file itself mentions forbidden phrases — but only in
    # "Not …" context. The classifier should treat every one as safe.
    leaks = [
        c for c in case.forbidden
        if _is_present(text, c) and not _is_safe_occurrence(text, c)
    ]
    assert not leaks, (
        f"{case.file} asserts a forbidden claim positively: {leaks}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_grounding_via_tools(
    case: GroundingCase, data_root: Path, tool_specs
) -> None:
    """Fast pipeline: read the right file via the tools, classify.

    A well-grounded crew answer should look like ``ls("*.md")`` →
    pick the right file → ``cat``. This test runs that exact path
    through the registered ``ToolSpec``s and checks the claim classifier
    against the joined output. Catches plumbing regressions without
    needing an LLM.
    """
    answer = _answer_via_tools(data_root, tool_specs, case.file)
    score = _classify(answer, case)
    assert score["grounded_recall"] == 1.0, (
        f"missing grounded claims: {score['grounded_misses']}\nanswer:\n{answer}"
    )
    assert score["forbidden_hit_rate"] == 1.0, (
        f"hallucinated forbidden claims: {score['forbidden_hits']}\nanswer:\n{answer}"
    )


# ----- slow tests: real crew against real Ollama --------------------------


def _real_ollama_or_skip() -> OllamaClient:
    client = OllamaClient()
    try:
        ok = client.health()
    except Exception:
        ok = False
    if not ok:
        pytest.skip("Ollama not reachable — slow grounding test skipped")
    return client


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_grounding_with_real_ollama(
    case: GroundingCase, data_root: Path, tool_specs
) -> None:
    """End-to-end grounding test against a real Ollama instance.

    * Requires a healthy Ollama at ``HERDING_CATS_OLLAMA_URL``.
    * Requires ``HERDING_CATS_MODEL`` (default ``llama3.1:8b``) to be pulled.
    * Asserts at least **80% grounded recall** and **at least 80%
      forbidden-hit-rate** per case. Tune ``GROUNDING_THRESHOLD`` via
      the env var of the same name.

    The crew runs in **rules mode** (default) so the orchestrator's
    decision tree is deterministic. The interesting variability is in
    the thinker's notes, the fetcher's tool calls, and the finalist's
    prose — exactly where hallucination happens.
    """
    threshold = float(os.environ.get("GROUNDING_THRESHOLD", "0.8"))
    ollama = _real_ollama_or_skip()
    crew = Crew(ollama=ollama, tool_specs=tool_specs)
    runner = CrewRunner(crew)
    answer = runner.run(case.question)
    score = _classify(answer, case)

    assert score["grounded_recall"] >= threshold, (
        f"grounding recall {score['grounded_recall']:.0%} < {threshold:.0%}\n"
        f"misses: {score['grounded_misses']}\nanswer:\n{answer}"
    )
    assert score["forbidden_hit_rate"] >= threshold, (
        f"forbidden hit rate {score['forbidden_hit_rate']:.0%} < {threshold:.0%}\n"
        f"hallucinated: {score['forbidden_hits']}\nanswer:\n{answer}"
    )


# ----- classifier sanity (no I/O) ----------------------------------------


def test_classifier_basic_match() -> None:
    case = GroundingCase(
        question="dummy",
        file="x.md",
        grounded=("apple", "Banana"),
        forbidden=("kiwi",),
    )
    score = _classify("Apple pie and banana split", case)
    assert score["grounded_recall"] == 1.0
    assert score["forbidden_hit_rate"] == 1.0


def test_classifier_flags_forbidden() -> None:
    case = GroundingCase(
        question="dummy",
        file="x.md",
        grounded=("apple",),
        forbidden=("kiwi",),
    )
    score = _classify("apple kiwi smoothie", case)
    assert score["forbidden_hits"] == ["kiwi"]
    assert score["forbidden_hit_rate"] == 0.0


def test_classifier_handles_empty_answer() -> None:
    case = GroundingCase(
        question="dummy",
        file="x.md",
        grounded=("apple",),
        forbidden=("kiwi",),
    )
    score = _classify("", case)
    assert score["grounded_recall"] == 0.0
    assert score["forbidden_hit_rate"] == 1.0


def test_matcher_tolerates_hyphenation() -> None:
    """Hyphen-vs-space mismatches should still count as a hit.

    The slow test caught this in the wild: the crew wrote
    ``single-binary`` while the fixture said ``single binary``. Without
    normalization the matcher reports a miss for the right answer.
    """
    text = "Porthound is a single-binary Git server written in Zig."
    assert _is_present(text, "single binary")
    assert _is_present(text, "single-binary")

    # The phrase is present, so ``_is_safe_occurrence`` returns False
    # unless it's negated. The crew's answer above is *positive* about
    # "single binary" being a fact, so it's correctly not-safe.
    assert not _is_safe_occurrence(text, "single binary")

    # Same phrase in a "Not …" context IS safe.
    assert _is_safe_occurrence(
        "Porthound is not a single binary SaaS.",
        "single binary",
    )


def test_matcher_tolerates_simple_plurals() -> None:
    """``bedtime story`` should match ``bedtime stories``.

    The slow test caught this in the wild: Sleepwell's answer used the
    plural ``bedtime stories`` while the claim was singular
    ``bedtime story``. Without plural tolerance, that's a 25% recall
    hit on a perfectly correct answer.
    """
    assert _is_present(
        "Sleepwell generates bedtime stories for adults.",
        "bedtime story",
    )
    assert _is_present(
        "Sleepwell generates a bedtime story for adults.",
        "bedtime story",
    )
    # Sibilant plural
    assert _is_present(
        "The box has watch boxes inside.",
        "watch box",
    )
    # Should NOT match unrelated words
    assert not _is_present(
        "Sleepwell is a meditation app.",
        "bedtime story",
    )


# ----- LLM stub kept for reference (unused by default) -------------------


class _FileQuotingOllama:
    """Deterministic LLM stub. Kept for future high-fidelity fast tests."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._calls = 0

    def health(self) -> bool:
        return True

    def _build(self, req: ChatRequest) -> ChatResponse:
        self._calls += 1
        last_user = next(
            (m for m in reversed(req.messages) if m.role == "user"), None
        )
        user_text = last_user.content if last_user else ""
        body = user_text  # echo the prompt verbatim — useful for debug
        return ChatResponse(
            model=req.model,
            message=ChatMessage(role="assistant", content=body),
            done=True,
            total_duration_ns=1,
        )

    def chat(self, req: ChatRequest) -> ChatResponse:
        return self._build(req)

    async def achat(self, req: ChatRequest) -> ChatResponse:
        return self._build(req)

    async def embed(self, req):  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
