"""Robust JSON object extraction from chat output.

Local models love to wrap JSON in prose, code fences, or leading
chatter. We do three things, in order:

1. If the whole string parses as JSON, return it.
2. If a fenced ```json ... ``` block is present, try that.
3. Otherwise, scan for the first balanced `{...}` block that parses.

We never raise; on failure we return `None` so the caller can record a
warning and continue.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try a fenced block.
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Last resort: find the first balanced { ... } block.
    return _first_balanced_object(text)


def _first_balanced_object(text: str) -> dict[str, Any] | None:
    start = -1
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = text[start : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        # Try the next opening brace from here.
                        start = -1
    return None
