"""Tests for the filesystem tools (ls, cat)."""

from __future__ import annotations

from pathlib import Path

import pytest

from herding_cats.tools_filesystem import (
    CatInput,
    LsInput,
    LsOutput,
    _resolve_within,
    cat,
    default_data_dir,
    filesystem_tools,
    ls,
)

# ----- fixtures ------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Create a sandbox directory tree for the tools to operate on."""
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# Title\n\nbody", encoding="utf-8")
    (tmp_path / "sub" / "binary.bin").write_bytes(b"\x00\x01\xff")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    return tmp_path


# ----- default_data_dir ---------------------------------------------------


def test_default_data_dir_reads_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERDING_CATS_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path.resolve()


def test_default_data_dir_relative_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERDING_CATS_DATA_DIR", "subdir")
    result = default_data_dir()
    assert result == (tmp_path / "subdir").resolve()


# ----- _resolve_within (the safety net) ------------------------------------


def test_resolve_within_accepts_relative(sandbox: Path) -> None:
    p = _resolve_within(sandbox, "a.txt")
    assert p == (sandbox / "a.txt").resolve()


def test_resolve_within_rejects_absolute(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="absolute paths"):
        _resolve_within(sandbox, str(sandbox / "a.txt"))


def test_resolve_within_rejects_traversal(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="escapes the data root"):
        _resolve_within(sandbox, "../etc/passwd")


def test_resolve_within_rejects_double_dot_in_middle(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="escapes the data root"):
        _resolve_within(sandbox, "sub/../../etc/passwd")


def test_resolve_within_rejects_empty(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="path is empty"):
        _resolve_within(sandbox, "")


# ----- ls -----------------------------------------------------------------


def test_ls_lists_root(sandbox: Path) -> None:
    out = ls(LsInput(), root=sandbox)
    assert isinstance(out, LsOutput)
    names = {e.name for e in out.entries}
    assert {"a.txt", "sub", "empty.txt"} <= names


def test_ls_lists_subdirectory(sandbox: Path) -> None:
    out = ls(LsInput(directory="sub"), root=sandbox)
    names = {e.name for e in out.entries}
    assert {"b.md", "binary.bin"} <= names


def test_ls_pattern_filter(sandbox: Path) -> None:
    out = ls(LsInput(pattern="*.md"), root=sandbox)
    assert {e.name for e in out.entries} == set()
    out = ls(LsInput(pattern="*.md", directory="sub"), root=sandbox)
    assert {e.name for e in out.entries} == {"b.md"}


def test_ls_handles_missing_directory(sandbox: Path) -> None:
    out = ls(LsInput(directory="nope"), root=sandbox)
    assert out.entries == []
    assert "does not exist" in out.note


def test_ls_reports_file_kinds_and_sizes(sandbox: Path) -> None:
    out = ls(LsInput(directory="."), root=sandbox)
    by_name = {e.name: e for e in out.entries}
    assert by_name["a.txt"].kind == "file"
    assert by_name["a.txt"].size == 11  # "hello world"
    assert by_name["sub"].kind == "dir"
    assert by_name["sub"].size >= 0


def test_ls_respects_max_entries(sandbox: Path) -> None:
    for i in range(10):
        (sandbox / f"f{i}.txt").write_text("x", encoding="utf-8")
    out = ls(LsInput(max_entries=3), root=sandbox)
    assert len(out.entries) == 3
    assert "truncated" in out.note


def test_ls_rejects_traversal(sandbox: Path) -> None:
    # The pydantic executor converts ValueError into validation_error.
    with pytest.raises(ValueError):
        ls(LsInput(directory=".."), root=sandbox)


# ----- cat ----------------------------------------------------------------


def test_cat_reads_text(sandbox: Path) -> None:
    out = cat(CatInput(path="a.txt"), root=sandbox)
    assert out.content == "hello world"
    assert out.size == 11
    assert out.truncated is False
    assert out.encoding == "utf-8"


def test_cat_reads_empty(sandbox: Path) -> None:
    out = cat(CatInput(path="empty.txt"), root=sandbox)
    assert out.content == ""
    assert out.size == 0


def test_cat_handles_missing(sandbox: Path) -> None:
    out = cat(CatInput(path="nope.txt"), root=sandbox)
    assert out.note  # has a message
    assert out.content == ""


def test_cat_truncates(sandbox: Path) -> None:
    out = cat(CatInput(path="a.txt", max_bytes=5), root=sandbox)
    assert out.content == "hello"
    assert out.truncated is True
    assert "truncated" in out.note.lower()


def test_cat_offset_paging(sandbox: Path) -> None:
    out = cat(CatInput(path="a.txt", offset=6), root=sandbox)
    assert out.content == "world"


def test_cat_rejects_traversal(sandbox: Path) -> None:
    with pytest.raises(ValueError):
        cat(CatInput(path="../../../etc/passwd"), root=sandbox)


def test_cat_rejects_absolute(sandbox: Path) -> None:
    with pytest.raises(ValueError):
        cat(CatInput(path=str(sandbox / "a.txt")), root=sandbox)


def test_cat_handles_binary_via_latin1(sandbox: Path) -> None:
    out = cat(CatInput(path="sub/binary.bin"), root=sandbox)
    # latin-1 decodes every byte; content is the byte string as chars.
    assert out.encoding in {"latin-1", "utf-8"}  # some bytes are valid utf-8 too
    assert len(out.content) == 3


def test_cat_directory_rejected(sandbox: Path) -> None:
    out = cat(CatInput(path="sub"), root=sandbox)
    assert "not a regular file" in out.note


# ----- registry helpers --------------------------------------------------


def test_filesystem_tools_returns_two_specs() -> None:
    specs = filesystem_tools()
    assert {s.name for s in specs} == {"ls", "cat"}
    for s in specs:
        assert s.input_schema is not None
        assert s.output_schema is not None


def test_filesystem_tools_uses_explicit_root(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("data", encoding="utf-8")
    specs = filesystem_tools(root=tmp_path)
    # Pull the fn and run it; it should see tmp_path, not cwd.
    ls_spec = next(s for s in specs if s.name == "ls")
    out = ls_spec.fn(LsInput())
    names = {e.name for e in out.entries}
    assert "x.txt" in names


def test_registry_describe_for_filesystem_tools(tmp_path: Path) -> None:
    specs = filesystem_tools(root=tmp_path)
    from herding_cats.crew.tools import ToolRegistry

    reg = ToolRegistry()
    for s in specs:
        reg.specs.append(s)
    desc = reg.describe()
    by_name = {d["name"]: d for d in desc}
    assert by_name["ls"]["input"]["type"] == "object"
    assert "directory" in by_name["ls"]["input"]["properties"]
    assert "path" in by_name["cat"]["input"]["properties"]
