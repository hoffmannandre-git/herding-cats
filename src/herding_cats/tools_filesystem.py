"""Filesystem tools: `ls` and `cat`.

These are the bare-bones counterparts of the shell utilities, packaged as
pydantic-typed `ToolSpec`s so the fetcher's prompt renders the exact
input/output shape.

* `ls(directory=".")`  — list a directory under the data root. Returns
  one entry per line: name, kind (file/dir), size, modified ISO timestamp.
* `cat(path)`         — read a text file under the data root. Returns the
  first `max_bytes` characters of the file (default 16 KiB) plus a
  flag indicating whether it was truncated.

Both tools are bound to a **root directory** (default: `HERDING_CATS_DATA_DIR`
or `Path.cwd()`). Any path that resolves outside the root is rejected
with a `ValueError` — this is the safety net that keeps the LLM from
reading `C:\\Windows\\System32\\config\\SAM` or `/etc/shadow`.

The root resolution is symlink-aware: if `/data` is a symlink to
`/var/data`, paths under both still resolve correctly. But the
*resolved* path is what we compare against, so `cat("data/../etc/passwd")`
still gets rejected.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from herding_cats.crew.tools import ToolSpec
from herding_cats.helpers.env import get_env

# ----- defaults ------------------------------------------------------------


def default_data_dir() -> Path:
    """Return the data root the filesystem tools are bound to.

    Resolution order:
    1. `HERDING_CATS_DATA_DIR` env var (absolute or relative; resolved to
       absolute against `cwd()`).
    2. `./data/` next to the current working directory.
    3. The current working directory itself (last resort, less safe).
    """
    env = get_env("DATA_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    cwd_data = (Path.cwd() / "data").resolve()
    if cwd_data.exists():
        return cwd_data
    return Path.cwd().resolve()


# ----- input / output schemas ---------------------------------------------


class LsInput(BaseModel):
    directory: str = Field(
        default=".",
        description=(
            "Directory to list, relative to the data root. Use '.' for the root. "
            "Absolute paths and parent references are rejected."
        ),
    )
    pattern: str = Field(
        default="*",
        description="Glob pattern to filter entries, e.g. '*.md' or '**/*.txt'.",
    )
    max_entries: int = Field(
        default=200, ge=1, le=5000,
        description="Safety cap on the number of entries returned.",
    )


class LsEntry(BaseModel):
    name: str
    path: str
    kind: Literal["file", "dir", "other"]
    size: int
    modified: str


class LsOutput(BaseModel):
    directory: str
    pattern: str
    entries: list[LsEntry]
    note: str = ""


class CatInput(BaseModel):
    path: str = Field(
        description=(
            "File to read, relative to the data root. Absolute paths and "
            "parent references are rejected."
        ),
    )
    max_bytes: int = Field(
        default=16 * 1024, ge=1, le=1024 * 1024,
        description="Read at most this many bytes (default 16 KiB).",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Skip this many bytes before reading (for paging).",
    )


class CatOutput(BaseModel):
    path: str
    content: str
    size: int
    truncated: bool
    encoding: str
    note: str = ""


# ----- safety helpers ------------------------------------------------------


def _resolve_within(root: Path, requested: str) -> Path:
    """Resolve `requested` against `root` and verify it stays inside.

    Raises `ValueError` if the resolved path escapes `root`. The
    comparison is done on the *resolved* paths, so symlinks that point
    outside `root` are also rejected.
    """
    root_resolved = root.resolve()
    if not requested:
        raise ValueError("path is empty")
    # If the user passed an absolute path, reject outright — the docs
    # say "relative to the data root".
    candidate = Path(requested)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {requested!r}")
    joined = (root_resolved / candidate).resolve()
    try:
        joined.relative_to(root_resolved)
    except ValueError as e:
        raise ValueError(
            f"path escapes the data root: {requested!r}"
        ) from e
    return joined


# ----- implementations ----------------------------------------------------


def _system_ls_long(directory: Path) -> str | None:
    """Try the platform-native `ls -la` for richer info on Unix.

    Returns the raw text on success, or `None` on any failure (Windows,
    missing binary, permission error, etc.). The structured Python
    listing is the source of truth; this is a nice-to-have.
    """
    if platform.system() == "Windows":
        return None
    try:
        result = subprocess.run(
            ["ls", "-la", str(directory)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def _file_kind(entry: Path) -> Literal["file", "dir", "other"]:
    if entry.is_dir():
        return "dir"
    if entry.is_file():
        return "file"
    return "other"


def ls(input: LsInput, *, root: Path) -> LsOutput:
    """List `input.directory` (under `root`) matching `input.pattern`."""
    target = _resolve_within(root, input.directory)
    if not target.exists():
        return LsOutput(
            directory=input.directory,
            pattern=input.pattern,
            entries=[],
            note=f"directory does not exist: {input.directory}",
        )
    if not target.is_dir():
        return LsOutput(
            directory=input.directory,
            pattern=input.pattern,
            entries=[],
            note=f"not a directory: {input.directory}",
        )

    raw = sorted(target.glob(input.pattern))
    entries: list[LsEntry] = []
    for p in raw:
        if len(entries) >= input.max_entries:
            break
        try:
            stat = p.stat()
            entries.append(
                LsEntry(
                    name=p.name,
                    path=str(p.relative_to(root)),
                    kind=_file_kind(p),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                )
            )
        except OSError:
            # Skip unreadable entries silently; the listing is best-effort.
            continue

    note_parts: list[str] = []
    if len(raw) > input.max_entries:
        note_parts.append(
            f"truncated to first {input.max_entries} of {len(raw)} matches"
        )
    native = _system_ls_long(target)
    if native:
        note_parts.append("native `ls -la` output included for reference")

    return LsOutput(
        directory=input.directory,
        pattern=input.pattern,
        entries=entries,
        note="; ".join(note_parts),
    )


def cat(input: CatInput, *, root: Path) -> CatOutput:
    """Read a text file under `root`, capped at `input.max_bytes`."""
    target = _resolve_within(root, input.path)
    if not target.exists():
        return CatOutput(
            path=input.path,
            content="",
            size=0,
            truncated=False,
            encoding="",
            note=f"file does not exist: {input.path}",
        )
    if not target.is_file():
        return CatOutput(
            path=input.path,
            content="",
            size=0,
            truncated=False,
            encoding="",
            note=f"not a regular file: {input.path}",
        )

    size = target.stat().st_size
    truncated = False
    try:
        # Read as bytes, slice, then decode with a forgiving codec.
        with target.open("rb") as f:
            if input.offset:
                f.seek(input.offset)
            data = f.read(input.max_bytes + 1)
        if len(data) > input.max_bytes:
            truncated = True
            data = data[: input.max_bytes]
        # Try utf-8 first, fall back to latin-1 (lossless for any byte).
        try:
            text = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
            encoding = "latin-1"
    except PermissionError as e:
        return CatOutput(
            path=input.path,
            content="",
            size=size,
            truncated=False,
            encoding="",
            note=f"permission denied: {e}",
        )
    except OSError as e:
        return CatOutput(
            path=input.path,
            content="",
            size=size,
            truncated=False,
            encoding="",
            note=f"read error: {e}",
        )

    return CatOutput(
        path=input.path,
        content=text,
        size=size,
        truncated=truncated,
        encoding=encoding,
        note=("truncated to max_bytes; use offset to page" if truncated else ""),
    )


# ----- registry helpers ---------------------------------------------------


def filesystem_tools(*, root: Path | None = None) -> list[ToolSpec]:
    """Return the `ls` and `cat` `ToolSpec`s bound to `root`.

    Pass `root` to bind to a specific directory (e.g. your docker mount).
    Default: `herding_cats.tools_filesystem.default_data_dir()`.
    """
    bound_root = (root or default_data_dir()).resolve()

    def _ls(input: LsInput) -> LsOutput:
        return ls(input, root=bound_root)

    def _cat(input: CatInput) -> CatOutput:
        return cat(input, root=bound_root)

    return [
        ToolSpec(
            name="ls",
            description=(
                "List files in a directory. Returns name, kind, size, "
                "modified time. Bound to the data root; cannot escape it."
            ),
            fn=_ls,
            input_schema=LsInput,
            output_schema=LsOutput,
        ),
        ToolSpec(
            name="cat",
            description=(
                "Read a text file. Returns up to max_bytes of content plus "
                "size/truncation metadata. Bound to the data root; cannot "
                "escape it."
            ),
            fn=_cat,
            input_schema=CatInput,
            output_schema=CatOutput,
        ),
    ]


def all_builtin_tools(*, root: Path | None = None) -> list[ToolSpec]:
    """Convenience: return the 5 general tools + the 2 filesystem tools.

    `root` is forwarded to the filesystem tools; other tools ignore it.
    """
    from herding_cats.tools_builtin import builtin_tools

    return builtin_tools() + filesystem_tools(root=root)
