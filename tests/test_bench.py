"""Unit tests for llama-bench invocation and JSON-row parsing.

Row schema (``n_prompt``, ``n_gen``, ``avg_ts``, ...) is verified against llama.cpp's own
printer, not guessed: ``tools/llama-bench/llama-bench.cpp`` ``benchmark_params_table::
get_fields()``/``get_field_type()`` (read 2026-09-25). With ``-p 512 -n 128 -r 3`` the tool
already averages the 3 repetitions into ``avg_ns``/``avg_ts`` and prints two rows -- one
prompt-processing row (``n_prompt=512, n_gen=0``) and one generation row (``n_prompt=0,
n_gen=128``) -- never 3x the row count.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import localmodels.bench as bench_mod
from localmodels.bench import BenchError, run_bench

_TWO_ROW_JSON = """[
  {"n_prompt": 512, "n_gen": 0, "avg_ns": 100000000, "avg_ts": 2943.6, "stddev_ts": 12.1},
  {"n_prompt": 0, "n_gen": 128, "avg_ns": 200000000, "avg_ts": 92.12, "stddev_ts": 1.4}
]"""


def _fake_run(monkeypatch, *, stdout: str = "", returncode: int = 0, raises: Exception | None = None):
    calls: list[list[str]] = []

    def _run(cmd, **_k):
        calls.append(cmd)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(bench_mod.subprocess, "run", _run)
    return calls


def test_run_bench_parses_prompt_and_decode_rows(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout=_TWO_ROW_JSON)
    result = run_bench(Path("/fake/llama-bench"), Path("/fake/model.gguf"))
    assert result.prompt_tok_s == pytest.approx(2943.6)
    assert result.decode_tok_s == pytest.approx(92.12)
    assert len(result.raw_rows) == 2


def test_run_bench_passes_the_documented_flags(monkeypatch) -> None:
    calls = _fake_run(monkeypatch, stdout=_TWO_ROW_JSON)
    run_bench(Path("/fake/llama-bench"), Path("/m.gguf"), n_prompt=512, n_gen=128, repetitions=3)
    cmd = calls[0]
    assert cmd[0] == "/fake/llama-bench"
    assert "-m" in cmd
    assert "/m.gguf" in cmd
    assert cmd[cmd.index("-p") + 1] == "512"
    assert cmd[cmd.index("-n") + 1] == "128"
    assert cmd[cmd.index("-r") + 1] == "3"
    assert cmd[cmd.index("-o") + 1] == "json"


def test_run_bench_raises_on_nonzero_exit(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="", returncode=1)
    with pytest.raises(BenchError):
        run_bench(Path("/fake/llama-bench"), Path("/m.gguf"))


def test_run_bench_raises_on_unparseable_output(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout="not json")
    with pytest.raises(BenchError):
        run_bench(Path("/fake/llama-bench"), Path("/m.gguf"))


def test_run_bench_raises_when_subprocess_cannot_start(monkeypatch) -> None:
    _fake_run(monkeypatch, raises=OSError("no such file"))
    with pytest.raises(BenchError):
        run_bench(Path("/fake/llama-bench"), Path("/m.gguf"))


def test_run_bench_raises_on_timeout(monkeypatch) -> None:
    _fake_run(monkeypatch, raises=subprocess.TimeoutExpired(cmd="llama-bench", timeout=1))
    with pytest.raises(BenchError):
        run_bench(Path("/fake/llama-bench"), Path("/m.gguf"))


def test_parse_rows_tolerates_missing_avg_ts() -> None:
    result = bench_mod._parse_rows([{"n_prompt": 512, "n_gen": 0}])
    assert result.prompt_tok_s is None
    assert result.decode_tok_s is None
