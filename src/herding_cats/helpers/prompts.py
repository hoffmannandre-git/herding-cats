"""Prompt loader.

Each role's prompt is a Markdown file under `prompts/`. Override per-role
with `Crew(prompts={"orchestrator": "<text>", ...})` or
`Crew(prompts_dir="./my_prompts")`.

Substitution syntax: `$name` and `${name}` (from `string.Template`).
We deliberately do NOT use `{name}` because prompts often contain JSON
example blocks (which look like `{ "key": "value" }`) and a brace-based
substitution would have to escape them everywhere.

Variables must be stringifiable. For complex values (lists, dicts,
pydantic models), pre-render them with `stringify()` before passing to
`render()`. The fetcher's `$tools` block, for instance, is a list of
schema-shaped dicts that get rendered via `stringify(tools)`.
"""

from __future__ import annotations

import json
import string
from functools import lru_cache
from pathlib import Path
from typing import Any

from herding_cats.helpers.env import get_env


@lru_cache(maxsize=1)
def default_prompts_dir() -> Path:
    """Return the package's bundled prompts directory."""
    return Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str, *, base_dir: Path | None = None) -> str:
    """Load one prompt file by role name.

    `name` is one of: `orchestrator`, `thinker`, `fetcher`, `finalist`, `direct`.
    Optionally suffixed with `.md`.

    Lookup order:

    1. `<sibling>/<lang>/<name>` — the "sibling tree" convention, e.g.
       ``prompts_de/orchestrator.md`` next to ``prompts/``. This is
       what the bundled German translations ship as.
    2. `<base>/<lang>/<name>` — the "subdirectory" convention, e.g.
       ``prompts/de/orchestrator.md`` inside the prompts dir. This
       is what user-supplied translation trees tend to look like.
    3. ``<base>/<name>`` — the language-neutral default (English in
       the bundled prompts).
    4. If none exists, ``FileNotFoundError`` listing all tried paths.

    The loader does **not** silently fall back from a found localized
    file to the default: if ``HERDING_CATS_LANGUAGE=de`` and the
    German file is missing, that is an error — operators should know
    they have an incomplete translation, not a silently-wrong
    language.
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    base = base_dir or Path(get_env("PROMPTS_DIR") or default_prompts_dir())
    lang = (get_env("LANGUAGE") or "en").lower()
    tried: list[Path] = []
    if lang != "en":
        # Convention 1: sibling tree (`<base_parent>/<dir_stem>_<lang>/<name>`).
        sibling = base.parent / f"{base.name}_{lang}" / name
        tried.append(sibling)
        if sibling.exists():
            return sibling.read_text(encoding="utf-8")
        # Convention 2: subdirectory inside the prompts tree.
        sub = base / lang / name
        tried.append(sub)
        if sub.exists():
            return sub.read_text(encoding="utf-8")
        # No translation found — strict mode: do NOT silently fall back
        # to English. Operators should know they have an incomplete
        # translation rather than getting a silently-wrong language.
        raise FileNotFoundError(
            f"HERDING_CATS_LANGUAGE={lang!r} but no {lang} translation "
            f"found for prompt {name!r}; tried: "
            f"{', '.join(str(p) for p in tried)}"
        )
    default = base / name
    tried.append(default)
    if default.exists():
        return default.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"prompt not found; tried: {', '.join(str(p) for p in tried)}"
    )


def stringify(value: Any, *, indent: int | None = 2) -> str:
    """Render a Python value as a string suitable for prompt substitution.

    Strings pass through unchanged. Everything else goes through
    `json.dumps` with sorted keys and stable indentation.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=indent, default=str, sort_keys=True)


def render(template: str, **vars: Any) -> str:
    """Substitute `$name` / `${name}` placeholders with `vars`.

    A `$$` literal becomes a single `$`. Unknown placeholders raise
    `KeyError` so missing variables are caught at render time, not later
    when the model misbehaves.

    Non-string values are stringified via `stringify()`.
    """
    safe_vars: dict[str, str] = {k: stringify(v) for k, v in vars.items()}
    try:
        return string.Template(template).substitute(safe_vars)
    except KeyError as exc:
        raise KeyError(f"prompt template missing variable: {exc.args[0]!r}") from exc
