"""Tests for the prompt loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from herding_cats.helpers.prompts import default_prompts_dir, load_prompt, render


def test_default_prompts_dir_exists() -> None:
    assert default_prompts_dir().is_dir()


def test_load_all_default_roles() -> None:
    for role in ("orchestrator", "thinker", "fetcher", "finalist", "direct"):
        text = load_prompt(role)
        assert isinstance(text, str)
        assert len(text) > 20


def test_load_supports_explicit_md() -> None:
    text = load_prompt("orchestrator.md")
    assert isinstance(text, str)


def test_load_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does-not-exist")


def test_load_with_no_language_env_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HERDING_CATS_LANGUAGE` unset -> the English default is loaded."""
    monkeypatch.delenv("HERDING_CATS_LANGUAGE", raising=False)
    monkeypatch.delenv("LOCALCREW_LANGUAGE", raising=False)
    text = load_prompt("orchestrator")
    assert "You are the Orchestrator" in text


def test_load_with_sibling_convention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`prompts_de/orchestrator.md` (sibling tree) is found."""
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "orchestrator.md").write_text("EN default", encoding="utf-8")
    sibling = tmp_path / "prompts_de"
    sibling.mkdir()
    (sibling / "orchestrator.md").write_text("DE translation", encoding="utf-8")
    monkeypatch.setenv("HERDING_CATS_LANGUAGE", "de")
    text = load_prompt("orchestrator", base_dir=base)
    assert text == "DE translation"


def test_load_with_subdirectory_convention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`prompts/de/orchestrator.md` (subdirectory) is found."""
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "orchestrator.md").write_text("EN default", encoding="utf-8")
    sub = base / "de"
    sub.mkdir()
    (sub / "orchestrator.md").write_text("DE subdir translation", encoding="utf-8")
    monkeypatch.setenv("HERDING_CATS_LANGUAGE", "de")
    text = load_prompt("orchestrator", base_dir=base)
    assert text == "DE subdir translation"


def test_load_missing_translation_raises_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing translation = `FileNotFoundError`, NOT a silent fallback."""
    base = tmp_path / "prompts"
    base.mkdir()
    (base / "orchestrator.md").write_text("EN default", encoding="utf-8")
    monkeypatch.setenv("HERDING_CATS_LANGUAGE", "fr")
    with pytest.raises(FileNotFoundError, match="but no fr translation"):
        load_prompt("orchestrator", base_dir=base)


def test_load_bundled_german_prompts_exist() -> None:
    """The shipped `prompts_de/` tree has a translation for every role."""
    base = default_prompts_dir()
    sibling = base.parent / f"{base.name}_de"
    for role in ("orchestrator", "thinker", "fetcher", "finalist", "direct"):
        path = sibling / f"{role}.md"
        assert path.exists(), f"missing bundled DE prompt: {path}"


def test_render_substitutes_vars() -> None:
    out = render("hello $name", name="world")
    assert out == "hello world"


def test_render_does_not_touch_json_braces() -> None:
    """The big reason we use $name instead of {name}: prompts have JSON examples."""
    out = render(
        'Respond with {"answer": "$value"}',
        value="yes",
    )
    assert out == 'Respond with {"answer": "yes"}'


def test_render_missing_var_raises() -> None:
    with pytest.raises(KeyError):
        render("hello $name")


def test_render_double_dollar_is_literal_dollar() -> None:
    out = render("price: $$5")
    assert out == "price: $5"
