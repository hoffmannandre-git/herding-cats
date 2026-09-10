# herding-cats

A small, opinionated Python runtime for **multi-agent LLM crews backed by a local [Ollama](https://ollama.com)** instance.

> *Not a single agent in a loop. A real crew: orchestrator, thinker, fetcher, finalist — each with a budget, each with a JSON contract.*

`herding-cats` is the runtime half of the pattern. You bring the model, the prompts, and the tools; the runtime handles the orchestration loop, the budget enforcement, and the JSON contracts so the model has nowhere to hide sloppy output.

> **Why "herding-cats"?** A crew of opinionated, mostly-quirky agents is the
> model, and the orchestrator is the bored rancher trying to keep them all
> pointed in the same direction. The Python package uses the underscore form
> (`herding_cats`); the project, Docker image, and PyPI name use the hyphen
> (`herding-cats`).

**Docker Compose is the supported way to run this project.** Contributor
`pip install -e` lives in [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture

```
                       ┌─────────────┐
                       │  user query │
                       └──────┬──────┘
                              ▼
           ┌─────────────────────────────────────┐
           │            ORCHESTRATOR             │  ← routes: think | fetch | final | stop
           └──────┬───────────┬───────────┬──────┘
                  │           │           │
                  ▼           ▼           ▼
              ┌───────┐   ┌───────┐   ┌───────┐
              │ THINK │   │ FETCH │   │ FINAL │  ← one prompt, one JSON contract
              └───┬───┘   └───┬───┘   └───┬───┘
                  │           │           │
                  │           ▼           │
                  │       ┌───────┐       │
                  │       │  EXEC │       │  ← deterministic tool runner, no LLM
                  │       └───┬───┘       │
                  │           │           │
                  └───────────┴───────────┘
                              ▼
                       ┌─────────────┐
                       │   answer    │
                       └─────────────┘
```

The orchestrator is told a budget one round lower than the real cap (the
**n-1 trick**). The runner enforces the real cap either way.

## Quick start (Docker)

```bash
git clone https://github.com/hoffmannandre-git/herding-cats.git
cd herding-cats
cp .env.example .env          # optional: model name, pipeline mode

docker compose --profile api up -d --build
curl -fsS http://localhost:18000/health
# → {"ok":true,"model":"llama3.1:8b"}

curl -fsS -X POST http://localhost:18000/crew/run \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is Cobble?\"}"
```

Profiles:

| Profile | Purpose | Port |
|---|---|---|
| `api` | FastAPI + Ollama + model pull | `18000` |
| `test` | `ruff` + `pytest -m "not slow"` in a one-shot container | — |

```bash
docker compose --profile test run --rm test
```

`./data` is mounted at `/data` for the `ls` / `cat` tools. Put files there
and ask the crew about them. Full env table is under [Configuration](#configuration).

## A concrete turn

Question: *What is Cobble and how is it different from a Stream Deck?*

Typical rules-mode loop (trace is illustrative, not a recorded run):

| Step | Who | What happens |
|---|---|---|
| 1 | **Orchestrator** | `next=think` — no hits yet, need a plan |
| 2 | **Thinker** | Notes + search terms, e.g. `["Cobble", "macropad", "e-ink"]` |
| 3 | **Orchestrator** | `next=fetch` |
| 4 | **Fetcher** | JSON tool plan: `{"tool":"ls","input":{"directory":".","pattern":"*.md"}}` |
| 5 | **Executor** | Runs `ls` in code (no LLM). Hits land in `CrewState.current.hits` |
| 6 | **Orchestrator** | `next=fetch` again (or think) until a file is read |
| 7 | **Fetcher** | `{"tool":"cat","input":{"path":"cobble.md"}}` |
| 8 | **Executor** | Returns file text into `hits` |
| 9 | **Orchestrator** | One reflection `think`, then `next=final` |
| 10 | **Finalist** | User-facing prose from `question` + `notes` + `hits` only |

Sample state snippets after a fetch:

```json
{
  "question": "What is Cobble and how is it different from a Stream Deck?",
  "think_used": 1,
  "fetch_used": 2,
  "current": {
    "notes": "Need Cobble brief; compare to Stream Deck only via negation.",
    "search_terms": ["Cobble macropad", "e-ink"],
    "hits": [
      {"tool": "cat", "input": {"path": "cobble.md"}, "output": {"content": "…"}}
    ]
  },
  "trace": [
    "think #1: notes=… new_terms=2",
    "fetch #1: ls(…)",
    "fetch #2: cat(cobble.md)",
    "think: reflection pass complete"
  ]
}
```

## Compared to CrewAI / LangChain / Pydantic AI

Use those if they fit. This project optimizes for a different cut of the problem:

| | herding-cats | CrewAI | LangChain | Pydantic AI |
|---|---|---|---|---|
| **Shape** | Fixed 4-role loop + budgets | Multi-agent crews, richer DSL | Chains / graphs / tools | Typed agents + tools |
| **Local Ollama** | First-class | Via adapters | Via adapters | Via providers |
| **Prompt ownership** | `.md` files, swap without code | Often in Python / templates | Often in Python / LCEL | Often in Python |
| **Stop condition** | Runner-enforced budgets | Agent / task config | Graph edges / limits | Your control flow |
| **Tool calls** | Fetcher JSON → deterministic executor | Framework tool calling | Tool nodes / bind_tools | Schema-validated tools |
| **LOC / surface** | Small, readable in an afternoon | Large product surface | Large ecosystem | Medium, typed |

Tradeoffs we accept:

- **More LLM calls per question** than a single agent (each role is cheap; total can be higher).
- **No provider-side function calling** — the fetcher emits JSON; you execute it.
- **Not a general agent framework** — four roles, one loop, local-first.

If you outgrow this, the durable assets are the role prompts and the
JSON contracts — porting those elsewhere is usually hours, not weeks.

## Why this exists

A single agent doing `THOUGHT → ACTION → OBSERVATION` works, but for tasks that involve *planning + retrieval + synthesis* it's fragile: the agent either burns the whole turn thinking, runs out of room for retrieval, or hallucinates because it can't fit both the search results and a sensible answer into one prompt.

Splitting the work into a small crew with separate prompts per role is one well-known fix. The boring half of "well-known fix" is:

1. Driving an Ollama chat call and getting back *strict* JSON when the model only wants to chat.
2. Counting rounds and stopping the crew before it spins.
3. Running the roles in sequence with deterministic budgets you can test.
4. Letting the user swap out any role's prompt without rewriting the runner.

`herding_cats` is the boring half.

### The three problems this is actually built to solve

The bullets above are symptoms; the underlying problems are three, and
each is a concrete reason this runtime is shaped the way it is. If
you only have thirty seconds, read these three sub-sections.

#### 1. Context management

A single-agent `THOUGHT → ACTION → OBSERVATION` loop has one context
window and four jobs to do in it: plan the search, issue the tool
call, read the tool result, write the answer. By the time the
thinker is done planning, half the window is gone; by the time the
tools come back, there's barely room for a coherent answer — and the
planner's intermediate scratch work is still sitting there polluting
the final response.

`herding-cats` gives each role **its own context window**. The thinker
sees the question and the rules of engagement; it does not see the
tool results. The fetcher sees only the search terms and the tool
schemas; it does not see the thinker's scratch. The finalist sees the
question, the tools' outputs, and the thinker's plan summary — and
*nothing else*. The orchestration state, the prior trace, and the
intermediate LLM chatter are kept in `CrewState` (in your code),
not in the prompt (in the LLM's context).

Concretely:

- `CrewState.trace` is a Python list of structured records, not a
  concatenated prompt string. The runner never copies it into a
  message — only a *rendered summary* of it goes to the LLM.
- Each role's `ChatRequest` is built from a tiny, role-specific
  template in `prompts/`. The templates know nothing about the other
  roles.
- `CrewState.seen_chunk_ids` is updated by the runner
  (`_record_chunk_ids` in `crew/runner.py`) as tool outputs come
  back. The IDs are passed to the fetcher prompt so the model can
  *avoid* re-reading them, but the dedup itself is enforced in code,
  not by the model — the runner extracts new IDs deterministically
  from tool outputs before storing them in the state.
- `CrewSession.turns` carries prior turns into the next run, but
  only as a compact summary capped at `max_prior_turns` (default 5).
  The raw chat history is *not* re-appended.

The result: even on a 7B model with a small context window, you can
run a multi-step retrieval task without the planner eating the
finalist's lunch.

#### 2. Reasoning depth

The runner does not ask the model to "think step by step and then
answer". It asks four different models of the same LLM to do four
different jobs, in sequence, with a budget per job.

- **Orchestrator** decides *what comes next*. Its prompt is small,
  its output is a one-or-two-word routing decision. It does not
  need to reason about the user's domain; it reasons about the
  *state machine*.
- **Thinker** is the only role that actually reasons about the
  user's question. It is given the question, the available tools, and
  the current trace; it returns a plan and (optionally) a reflection
  on whether the previous fetch was enough. The runner enforces
  `MAX_THINK` rounds so the thinker can't loop on "let me think
  more".
- **Fetcher** is *purely mechanical*: turn the plan into a JSON tool
  call. Its prompt lists the tools and their input schemas; its
  output is parsed, validated against the Pydantic schema, and
  executed. There is nothing to reason about here, which is why the
  prompt is so short.
- **Finalist** is the only role that writes for the user. It is given
  the question, the plan, the tool outputs, and (optionally) a
  reflection — and *only* those. It cannot decide to call another
  tool; if the search was insufficient, the orchestrator routes back
  to the thinker, not the finalist.

Splitting reasoning across roles lets each role use the full depth
of its context window for one job, instead of sharing a single
window across four. The tradeoff: more LLM calls per turn, each one
cheap. With a local 7B model on a modern CPU, a full
`think → fetch → think → final` cycle is sub-second per call.

#### 3. Swapping prompts without rewriting the runner

The runner never inlines a prompt. Every prompt lives in a
`.md` file under `src/herding_cats/prompts/`:

```text
prompts/
├── direct.md        ← smalltalk / direct-answer shortcut
├── orchestrator.md  ← routing rules + JSON contract
├── thinker.md       ← planning + reflection contract
├── fetcher.md       ← tool-pick + JSON contract
└── finalist.md      ← answer-writing rules + JSON contract
```

To swap any prompt, you do **not** touch the runner. Three ways, in
increasing order of override:

1. **Per-environment language**: set `HERDING_CATS_LANGUAGE=de` and
   put a German translation under `prompts_de/` (sibling tree) or
   `prompts/de/` (subdirectory — the loader accepts both). If the
   translation is missing, the loader **raises** rather than
   silently falling back to English.
2. **Per-call directory**: pass `Crew(prompts_dir="./my_prompts")`
   and drop your own `.md` files in there. The directory layout
   mirrors `prompts/` exactly.
3. **Per-call inline text**: pass
   `Crew(prompts={"orchestrator": "<text>", "thinker": "<text>"})`
   and the loader is bypassed entirely for the keys you provide.

The runner reads each prompt by name at call time; there is no
import-time coupling, no plugin registry, no decorator that pins the
prompt to a class. The Markdown files use `$name` / `${name}`
substitution (`string.Template`), so JSON example blocks inside a
prompt — which look like `{ "key": "value" }` — don't collide with
the variable syntax.

`scripts/check_prompts_isolated.py` enforces this property: it
greps the source for "You are the Orchestrator…" and similar
strings and exits non-zero if any role's prompt leaks into a `.py`
file. Run it locally with:

```bash
py scripts/check_prompts_isolated.py
```

## What you get

- **A typed `OllamaClient`** (sync + async) — chat, embed, streaming, JSON-mode, health checks. No surprises, no vendor lock-in beyond Ollama's HTTP shape.
- **A 4-role crew runtime**: orchestrator → thinker → fetcher → finalist, with a deterministic executor that enforces the fetcher's tool calls.
- **Pydantic tool schemas**: `ToolSpec` carries optional `input_schema` and `output_schema`; the executor validates both, and the fetcher's prompt is auto-generated from the schema.
- **Built-in tool registry** (`get_current_time`, `calculator`, `date_now`, `echo`, `web_search` stub) — drop-in sensible defaults.
- **Multi-turn sessions** with `CrewSession`: prior turns + seen chunk IDs are passed to the thinker/fetcher automatically.
- **HTTP server** (FastAPI, optional `server` extra): `/crew/run`, `/crew/step`, `/crew/stream`, plus session endpoints.
- **Budget-aware loops**: configurable max rounds, fetch rounds, think rounds, and per-call token caps. The runner enforces them whether the LLM likes it or not.
- **JSON contracts enforced from the outside**: every role's output is parsed and validated; malformed output triggers a bounded retry, not silent acceptance.
- **Pluggable tool surface**: pass any Python function as a tool; the fetcher picks from them by emitting a JSON tool plan.
- **Prompt localization built in**: ship English defaults under `prompts/` and one or more translations under `prompts_<lang>/` (sibling tree) or `prompts/<lang>/` (subdirectory). Set `HERDING_CATS_LANGUAGE=de` and the loader picks the German file. Missing translations are an error — no silent fallback to English.
- **Zero magic**: no framework DSL, no decorator spaghetti, no global state. One `Crew` class, one `CrewRunner.run(question)` call.

## When you should *not* use it

- If you only need a single-pass Q&A or one assistant turn. Use `ollama.Client` directly.
- If you need tool-use in the OpenAI function-calling sense with provider-side validation. `herding_cats` is intentionally smaller than that — the fetcher emits JSON, you execute it.
- If you want a hosted / cloud model by default. `herding_cats` is built for local Ollama. You can point it at any OpenAI-compatible endpoint with `HERDING_CATS_OLLAMA_URL`, but you give up the "local first" pitch.

## Python API (optional)

For scripts and examples, the public API is small. Prefer Docker for day-to-day
runs; use this when embedding the runner in another process (see
[CONTRIBUTING.md](CONTRIBUTING.md) for editable installs).

```python
from herding_cats import Crew, CrewRunner, OllamaClient

ollama = OllamaClient()  # HERDING_CATS_OLLAMA_URL or http://localhost:11434

crew = Crew(
    ollama=ollama,
    tools={
        "get_weather": lambda city: {"city": city, "temp_c": 17, "summary": "overcast"},
        "lookup_user": lambda user_id: {"id": user_id, "name": "Demo User", "role": "admin"},
    },
)

runner = CrewRunner(crew)
answer = runner.run("What's the weather in Berlin and who am I?")
print(answer)
```

### Why four roles, not one?

Because each role has a different *job to fail at*:

| Role | Job | Failure mode the runtime catches |
|---|---|---|
| **Orchestrator** | Decide what comes next | Stuck in `think` forever → runner kills at `MAX_LOOP` |
| **Thinker** | Plan the search before fetching | Skipped entirely → runner won't let `fetch` run without it |
| **Fetcher** | Emit one tool call as JSON | Bad JSON → bounded retry, then `final` with a warning |
| **Finalist** | Write the user-facing answer | Empty answer → caught by the contract |

If a single agent tried to do all four, every failure mode would compete for the same context window.

## Configuration

All settings are env vars; defaults are sane for a single-machine dev setup.
The new `HERDING_CATS_*` prefix is canonical; the `LOCALCREW_*` names from
earlier versions are accepted as a fallback for one release cycle.

| Variable | Default | Purpose |
|---|---|---|
| `HERDING_CATS_OLLAMA_URL` | `http://localhost:11434` | Ollama HTTP endpoint (fallback: `LOCALCREW_OLLAMA_URL`, then `OLLAMA_BASE_URL`) |
| `HERDING_CATS_MODEL` | `llama3.1:8b` | Chat model for all roles |
| `HERDING_CATS_EMBED_MODEL` | `nomic-embed-text` | Embedding model (used only if you call `ollama.embed`) |
| `HERDING_CATS_MAX_LOOP` | `12` | Hard cap on orchestrator rounds |
| `HERDING_CATS_MAX_FETCH` | `3` | Max fetch rounds per turn |
| `HERDING_CATS_MAX_THINK` | `4` | Max think rounds per turn |
| `HERDING_CATS_SEARCHES_PER_FETCH` | `2` | Tool calls per fetch round |
| `HERDING_CATS_TOKEN_BUDGET` | `4096` | Max output tokens per role call |
| `HERDING_CATS_PROMPTS_DIR` | `<pkg>/prompts/` | Override path for Markdown prompts |
| `HERDING_CATS_LANGUAGE` | `en` | Prompt language. Looks for `prompts_<lang>/<role>.md` (sibling) or `prompts/<lang>/<role>.md` (subdir). Missing translation = `FileNotFoundError`. |

You can also override per-`Crew` instance:

```python
crew = Crew(
    ollama=ollama,
    tools={"my_tool": my_tool},
    max_loop=8,
    max_fetch=2,
    max_think=3,
    model="qwen2.5:14b",
    prompts_dir="./my_prompts",  # or pass per-role strings
)
```

## Writing custom prompts

Each role's prompt is a Markdown file in `prompts/`. Override any one of them:

```python
from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.prompts import load_prompt

crew = Crew(
    ollama=OllamaClient(),
    tools={...},
    prompts={
        "orchestrator": load_prompt("./prompts/orchestrator.md"),
        "fetcher": load_prompt("./prompts/fetcher.md"),
        # thinker + finalist fall back to defaults
    },
)
```

A good role prompt has three sections:

1. **Role** — one sentence: who the model is in this step.
2. **Inputs** — what's in the context the model will see (be explicit).
3. **Output contract** — the JSON shape you want back, with an example.

The default prompts use this structure. Treat them as a template, not as gospel.

#### Variable substitution

The runtime substitutes **`$name`** placeholders (not `{name}`) so that
JSON example blocks in your prompts don't have to be brace-escaped:

```markdown
You will see:
- $question: the user's question
- $tools: a list of {name, description, input_hint}

Respond with {"answer": "...", "confidence": 0.0-1.0}
```

A literal `$` is `$$`. Unknown placeholders raise `KeyError` at render time so you catch them before the model runs.

## Writing tools

A tool is any async or sync callable taking a `dict` and returning a JSON-serializable value.

```python
def get_weather(input: dict) -> dict:
    """Return current weather for a city."""
    city = input["city"]
    return {"city": city, "temp_c": 17, "summary": "overcast"}

crew = Crew(
    ollama=ollama,
    tools={"get_weather": get_weather},
)
```

The fetcher's prompt should list the tool name and the expected `input` shape. The runtime does **not** validate the fetcher's tool plan against a schema by default — validation is the prompt's job, not the framework's. (We learned this the hard way: schemas drift, prompts drift faster, the prompt is the source of truth.)

## Streaming

For UI integrations, the runner exposes the same loop as `run_stream()`:

```python
for event in runner.run_stream("What's the weather in Berlin?"):
    match event.kind:
        case "orchestrator_decision":
            print("→", event.payload["next"], event.payload["reason"])
        case "tool_call":
            print("🔧", event.payload["tool"], event.payload["input"])
        case "tool_result":
            print("↳", event.payload["output"])
        case "final":
            print("\n" + event.payload["answer"])
```

The event kinds mirror the stages of the loop; consume them in any order, ignore the ones you don't care about.

## Per-step steering (UIs and HTTP)

The runner can be advanced one orchestrator decision at a time via
`CrewRunner.step(state)`. This is the basis of an HTTP `/crew/step`
endpoint, a pause/inspect/resume UI panel, or external steering from a
human-in-the-loop.

`CrewState` is a pydantic model and round-trips through JSON, so a UI
can mutate the state (cancel a fetch, switch tools, force-final, edit
notes) and hand it back to the runner to continue.

```python
from herding_cats import Crew, CrewRunner, CrewState, pipeline_status

crew = Crew(ollama=ollama, tools={...})
runner = CrewRunner(crew)

state = CrewState(question="What's the weather in Berlin?")

while True:
    result = runner.step(state)
    state = result.state
    print(f"→ {result.decision.next} ({result.decision.reason})")
    print(f"  pipeline: {pipeline_status(state, result.decision)}")
    if result.done:
        break
```

The `decision`, `state`, and `event` returned by `step()` describe exactly
what the runner did in one orchestrator cycle. See `examples/step_demo.py`
for a runnable demo.

## Short-circuiting known queries

Two ways to skip the crew entirely for queries you can answer without
an LLM call:

**1. Smalltalk** — built-in. When `Crew.short_circuit_smalltalk=True`
(the default) and the question is a greeting/pleasantry, the runner
goes straight to a lightweight "direct" prompt and skips the
orchestrator entirely.

**2. Shortcuts** — you provide a `{regex: handler}` map. When the
question matches a regex, the handler is called directly and the
answer is returned without any LLM call:

```python
def list_contacts(_input: dict) -> str:
    return "\n".join(f"- {c['name']}" for c in CONTACTS)

crew = Crew(
    ollama=ollama,
    shortcuts={
        r"^(list|show)\s+(all\s+)?contacts\??$": list_contacts,
    },
)
```

If a handler raises, the runner logs the error and falls through to
the regular crew path. See `examples/shortcuts_demo.py`.

## Anti-ping-pong

If a fetch returns hits, the rules-mode orchestrator asks for *exactly
one* reflection-think pass and then goes to `final`. Without this guard,
the crew can spin `think → fetch → think` forever on conversational
queries. The flag lives on `CrewState.reflection_think_done` and is
cleared automatically by the runner when the reflection completes.

The fetcher also tracks `state.queries_executed` and silently retries
with the next queued search term if the model loops to a duplicate
query. The search-term queue itself (`state.current.search_terms`) is
consumed FIFO via `state.current.search_term_index`, so a single term
isn't re-suggested to the model mid-reflection.

## Multi-turn sessions

`CrewSession` wraps a `CrewRunner` and remembers prior turns. Each new
question becomes a `CrewState` with `prior_turns`, `prior_context`, and
`seen_chunk_ids` populated from history, so the thinker can resolve
follow-up questions ("what about tomorrow?") and the fetcher can
avoid re-retrieving the same chunks.

```python
from herding_cats import Crew, CrewRunner, CrewSession, OllamaClient

crew = Crew(ollama=OllamaClient(), tools={...})
runner = CrewRunner(crew)
sess = CrewSession.with_in_memory(runner, prior_context="user is German")

sess.ask("What's the weather in Berlin?")
sess.ask("And tomorrow?")   # <-- prior turn visible to the thinker
```

Two storage backends ship:

* `InMemorySession` — process-local, lost on restart
* `JsonFileSession` — one JSON file per session, persists across restarts

## Built-in tools

Five ready-to-use tools with pydantic schemas ship in
`herding_cats.tools_builtin`:

| Tool | Purpose |
|---|---|
| `get_current_time` | Wall-clock in a timezone |
| `date_now` | Today's ISO date + weekday |
| `calculator` | Sandboxed arithmetic (math + abs/round/min/max) |
| `echo` | Repeat text N times |
| `web_search` | **Stub** — returns empty + a "replace me" note |

```python
from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.tools_builtin import builtin_tools

crew = Crew(ollama=OllamaClient(), tool_specs=builtin_tools())
print(CrewRunner(crew).run("What is 17 * 23?"))
```

To plug in a real web search, pass your own callable:

```python
crew = Crew(
    ollama=ollama,
    tools={"web_search": my_searxng_function},
    tool_specs=[s for s in builtin_tools() if s.name != "web_search"],
)
```

## Filesystem tools (`ls` / `cat`)

Two additional tools in `herding_cats.tools_filesystem` let the crew
locate and read files in a project directory:

| Tool | Purpose |
|---|---|
| `ls` | List a directory — returns name, kind, size, modified time |
| `cat` | Read a text file — capped at `max_bytes` with paging via `offset` |

Both tools are bound to a **data root** (default `./data/`, override
with `HERDING_CATS_DATA_DIR`). Any path that resolves outside the root is
rejected — the LLM cannot escape into `C:\Windows\System32` or
`/etc/`.

```python
from herding_cats import Crew, CrewRunner, OllamaClient
from herding_cats.tools_builtin import builtin_tools
from herding_cats.tools_filesystem import filesystem_tools, default_data_dir

crew = Crew(
    ollama=OllamaClient(),
    tool_specs=builtin_tools() + filesystem_tools(root=default_data_dir()),
)
print(CrewRunner(crew).run("What .md files are in the data directory?"))
```

The `docker-compose.yml` mounts `./data/` (on the host) into the
container at `/data` and sets `HERDING_CATS_DATA_DIR=/data`, so the same
crew works the same way whether you run it locally or in Docker.

See `examples/filesystem_demo.py`.

## Data directory layout

The `data/` directory is the safe root the filesystem tools can read.
`docker-compose.yml` mounts it as a volume, so files you drop in
`./data/` on the host show up at `/data` inside the container:

```yaml
services:
  ollama:
    volumes:
      - ./data:/data
    environment:
      HERDING_CATS_DATA_DIR: /data
```

The directory ships with a `.gitkeep` so the mount works on a fresh
clone, plus 5 fictional project briefs (`cobble.md`, `porthound.md`,
`sleepwell.md`, `manuscriptr.md`, `astrolabe.md`) used by the
grounding tests. Drop any extra files you want the crew to be able to
locate into `./data/` and they'll be visible to `ls` / `cat`.

## Grounding / hallucination tests

`tests/test_grounding.py` ships a parametrized test harness that:

* Asks the crew one question per fictional project.
* Classifies the answer against a hand-built **claim list** per project.
* Asserts both **grounded recall** (must mention the right facts) and
  **forbidden-hit-rate** (must NOT mention real-product facts that the
  brief explicitly denies).

The classifier is **negation-aware** — "Cobble is not a Stream Deck"
counts as *safe* even though the words "Stream Deck" appear in the
answer. The classifier only flags a forbidden phrase when it appears
without a nearby negation cue (`not`, `no`, `n't`, `never`, `without`).

```bash
# Fast suite (preferred)
docker compose --profile test run --rm test

# Slow grounding against real Ollama
HERDING_CATS_PIPELINE_MODE=all docker compose --profile test run --rm test
```

Each project has 4-5 grounded claims and 4-5 forbidden claims. With
`GROUNDING_THRESHOLD=0.8` (the default) the slow tests require ≥ 80%
on both axes per project. Tune the threshold up to make the test
stricter as models improve. We do **not** claim hallucination
tolerance — this harness measures grounding with a deterministic
classifier (no LLM judge).

## QA catalogue (eval / report)

Separate from the grounding classifier: a YAML question catalogue
graded **facts-first** (substring hits vs `min_fact_hits`), with an
optional **Ollama LLM judge** when facts miss. Markdown report under
`tests/reports/`. Still **not** a claim of hallucination tolerance.

```bash
# Fast: loader + judge unit tests (in the normal suite)
docker compose --profile test run --rm test

# Full report against a live Ollama-backed crew (host)
py tests/run_qa_report.py
py tests/run_qa_report.py --llm-judge   # or HERDING_CATS_QA_LLM_JUDGE=1
```

| Env | Default | Purpose |
|---|---|---|
| `HERDING_CATS_QA_LLM_JUDGE` | off | Enable LLM fallback for non-trivial rows |
| `HERDING_CATS_JUDGE_MODEL` | `HERDING_CATS_MODEL` / `llama3.1:8b` | Judge model |

Catalogue: `tests/question_catalogue.yaml`. Slow pytest twins:
`test_qa_catalogue_smoke_trivial` / `test_qa_catalogue_full_report`.

## HTTP server

With the `api` profile the server is already running on port `18000`.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Ollama liveness (`/healthz` alias) |
| `POST` | `/crew/run` | One-shot Q&A |
| `POST` | `/crew/step` | One orchestrator decision — for HTTP step APIs |
| `POST` | `/crew/stream` | Server-Sent Events stream |
| `POST` | `/sessions/{id}/ask` | Multi-turn: remembers history |
| `GET`  | `/sessions/{id}` | Get session history |
| `DELETE` | `/sessions/{id}` | Drop a session |
| `GET`  | `/sessions` | List session ids |

OpenAPI: `http://localhost:18000/docs` (when the API container is up).

Programmatic wiring (contributors / embedding):

```python
from herding_cats import Crew, OllamaClient
from herding_cats.server import create_app
import uvicorn

crew = Crew(ollama=OllamaClient(), tools={...})
app = create_app(crew=crew)
uvicorn.run(app, host="127.0.0.1", port=8765)
```

## Containerized runtime

| Profile | What it does | Port |
|---|---|---|
| `api` | Long-running FastAPI server backed by Ollama | `18000` |
| `test` | One-shot: `ruff check` + `pytest -m "not slow"` | — |

```bash
docker compose --profile api up -d --build
curl -fsS http://localhost:18000/health

docker compose --profile test run --rm test
```

Set `HERDING_CATS_PIPELINE_MODE=all` in `.env` to also run the slow
grounding suite against real Ollama (adds many minutes).

Image `CMD` is `herding-cats-api`.

## Testing

### Results (last local run)

| Suite | Command | Result | Coverage |
|---|---|---|---|
| Fast | `pytest -m "not slow"` / Compose `test` profile | **150 passed**, 9 deselected | **not measured yet** |
| Lint | `ruff check .` | clean | — |
| Prompt isolation | `py scripts/check_prompts_isolated.py` | OK | — |
| Slow grounding | `pytest -m slow` (needs Ollama) | optional | not measured yet |

CI runs lint + the fast suite on every push/PR (see `.github/workflows/ci.yml`).
Slow / Ollama jobs are **not** required in CI.

### How to run

```bash
# Preferred — same path as users
docker compose --profile test run --rm test

# Contributors — see CONTRIBUTING.md
pytest -q -m "not slow"
```

We do **not** claim "hallucination tolerance" — the grounding harness
measures recall / forbidden-hit rate with a deterministic classifier.

## Benchmark: single-agent vs crew

`scripts/benchmark_single_vs_crew.py` times one Ollama chat call against a
full crew turn on the same question (default: Cobble from `data/`).

```bash
docker compose --profile api up -d
python scripts/benchmark_single_vs_crew.py
```

Honest expectations (re-run the script on your machine):

| Mode | What it measures | Typical shape |
|---|---|---|
| **Single-agent** | One chat completion | Lower latency, weaker structure |
| **Crew** | Full orchestrator → … → final loop | Higher latency, tool hits constrain the finalist |

The script prints wall times and a short grounding note using the same
claim lists as `tests/test_grounding.py` (string classifier, not an LLM
judge). Treat it as a worksheet, not a leaderboard.

## Project layout

```
herding-cats/
├── src/herding_cats/          # Python package (underscore)
├── data/                      # ls/cat root (Docker mount)
├── examples/
├── tests/
├── scripts/
│   ├── check_prompts_isolated.py
│   └── benchmark_single_vs_crew.py
├── .github/workflows/ci.yml   # ruff + pytest (not slow)
├── docker-compose.yml
├── Dockerfile
├── CONTRIBUTING.md
├── .env.example
├── pyproject.toml
├── LICENSE
└── README.md
```

The package under `src/herding_cats/` holds crew roles, prompts,
`prompts_de/`, HTTP server, and helpers.

## License

MIT. See `LICENSE`.

## Acknowledgements

This pattern came out of building an offline-first office assistant where we couldn't trust the model to plan, fetch, and answer in a single context window without drifting. The four-role crew with explicit JSON contracts and budget enforcement made it tractable on a small local model. If you ship a `herding-cats` of your own, please open an issue — we'd love to link to it from the README.
