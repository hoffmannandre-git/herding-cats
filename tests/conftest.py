"""Shared test fixtures."""

from __future__ import annotations

import os

# Ensure tests can find the package even without an editable install.
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def ollama_url() -> str:
    return OLLAMA_URL


@pytest.fixture(scope="session")
def skip_if_no_ollama() -> None:
    """Skip a test if no Ollama is reachable."""
    import httpx

    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(f"Ollama not healthy at {OLLAMA_URL}")
    except httpx.HTTPError:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL}")
