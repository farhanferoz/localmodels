"""Atomic writes: a reader sees the old file or directory, or the new one, never a half-written one."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def write_text_atomically(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` through a temporary file beside it, then rename it over ``path``.

    The temporary name ends in ``.tmp``, so a reader globbing for the final suffix (the lease directory's
    ``*.json``) never picks it up; it is removed if the write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temp).replace(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return path


def replace_dir_atomically(target: Path, fill: Callable[[Path], None], marker: str) -> None:
    """Make the directory ``target`` from what ``fill`` writes into an empty staging directory beside it.

    ``marker`` names the file a complete directory holds. A ``target`` that already holds it is kept as it is,
    and losing a race to another writer of the same directory is not an error.
    """
    if (target / marker).is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=".tmp-"))
    try:
        fill(staging)
        staging.replace(target)
    except OSError:
        if not (target / marker).is_file():
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
