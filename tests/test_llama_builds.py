"""`localmodels llama ...`: the CMake options of a build, switching `current`, and pruning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from localmodels.cli import main
from localmodels.llama_builds import cmake_options, prune_builds, switch

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

CACHE = """\
//Choose the type of build
CMAKE_BUILD_TYPE:STRING=Release

//No help, variable specified on the command line.
CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=121

//CUDA compiler
CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc

//ggml: use CUDA
GGML_CUDA:BOOL=ON

//ggml: use HIP
GGML_HIP:BOOL=OFF

//No help, variable specified on the command line.
LLAMA_CURL:UNINITIALIZED=ON

CMAKE_COMMAND:INTERNAL=/usr/bin/cmake
"""


def make_build(home: Path, commit: str, *, finished: bool = True) -> Path:
    folder = home / "builds" / commit
    (folder / "bin").mkdir(parents=True)
    if finished:
        (folder / "bin" / "llama-server").touch()
    return folder


def test_build_reproduces_the_command_line_options_and_not_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "CMakeCache.txt").write_text(CACHE, encoding="utf-8")
    assert cmake_options(tmp_path / "CMakeCache.txt") == [
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=121",
        "-DCMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc",
        "-DGGML_CUDA:BOOL=ON",
        "-DLLAMA_CURL=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
    ]
    home = tmp_path / "home"
    args = ["llama", "--home", str(home), "build", "--commit", "abc1234", "--dry-run"]
    assert main([*args, "--reference", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"git clone --no-checkout https://github.com/ggml-org/llama.cpp {home / 'src'}"
    assert f"-B {home / 'builds' / 'abc1234'} -DCMAKE_BUILD_TYPE:STRING=Release" in out[3]
    assert not (home / "builds").exists()


def test_switch_is_refused_until_a_comparison_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_build(tmp_path, "abc1234")
    base = ["llama", "--home", str(tmp_path), "switch", "abc1234"]
    assert main(base) == 1
    assert "comparison.json does not exist" in capsys.readouterr().err
    comparison = tmp_path / "builds" / "abc1234" / "comparison.json"
    comparison.write_text(json.dumps({"passed": False, "slowdown": 0.2}), encoding="utf-8")
    assert main(base) == 1
    assert "does not record passed: true" in capsys.readouterr().err
    assert not (tmp_path / "builds" / "current").exists()
    comparison.write_text(json.dumps({"passed": True}), encoding="utf-8")
    comparison.rename(comparison.with_name("kept.json"))
    problem = switch(
        tmp_path, "abc1234"
    )  # the function the refresh job calls: the gate, as a returned reason
    assert problem is not None
    assert "comparison.json does not exist" in problem
    comparison.with_name("kept.json").rename(comparison)
    assert main(base) == 0
    assert (tmp_path / "builds" / "current").readlink().name == "abc1234"


def test_prune_keeps_current_and_the_builds_projects_name(tmp_path: Path) -> None:
    for commit in ("cur0000", "pinned1", "followed", "unused1", "unused2"):
        make_build(tmp_path, commit)
    (tmp_path / "builds" / "current").symlink_to("cur0000")
    (tmp_path / "builds" / "users.toml").write_text(
        '[projects]\nexperiment = "pinned1"\nlong = "followed0123456789"\n',
        encoding="utf-8",
    )
    assert main(["llama", "--home", str(tmp_path), "prune", "--dry-run"]) == 0
    assert (tmp_path / "builds" / "unused1").exists()
    kept = prune_builds(
        tmp_path, keep=["unused2"], dry_run=True
    )  # the refresh job keeps the previous current
    assert [folder.name for folder in kept] == ["unused1"]
    assert main(["llama", "--home", str(tmp_path), "prune"]) == 0
    assert sorted(p.name for p in (tmp_path / "builds").iterdir()) == [
        "cur0000",
        "current",
        "followed",
        "pinned1",
        "users.toml",
    ]
