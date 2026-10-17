from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from localmodels.atomic import replace_dir_atomically, write_text_atomically

if TYPE_CHECKING:
    from pathlib import Path

MARKER = "manifest.json"


def test_a_directory_is_built_once_and_a_complete_one_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "inputs" / "abc"

    def fill(staging: Path) -> None:
        (staging / "x").write_text("1")
        (staging / MARKER).write_text("{}")

    def refill(_staging: Path) -> NoReturn:
        raise AssertionError("a complete directory was built again")

    replace_dir_atomically(target, fill, MARKER)
    replace_dir_atomically(target, refill, MARKER)
    assert (target / "x").read_text() == "1"
    assert [p.name for p in target.parent.iterdir()] == ["abc"]  # no staging directory left


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    write_text_atomically(path, "old")
    with pytest.raises(UnicodeEncodeError):
        write_text_atomically(path, "\ud800")  # a lone surrogate cannot be written as UTF-8
    assert path.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
