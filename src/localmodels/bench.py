"""llama-bench invocation and JSON-row parsing, torch-free.

The caller holds the GPU lease before calling ``run_bench`` -- this module never touches the
lease itself, so it stays reusable by any caller that already holds it for another reason
(``setup.py``'s bench step is the one that acquires/releases it around this call).

Row schema (``n_prompt``, ``n_gen``, ``avg_ts``, ...) verified against llama.cpp's own printer
(``tools/llama-bench/llama-bench.cpp`` ``get_fields()``/``get_field_type()``, read 2026-09-25):
with ``-p 512 -n 128 -r 3`` the tool already averages the 3 repetitions into a single
``avg_ns``/``avg_ts`` per row and prints two rows -- one prompt-processing row
(``n_prompt=512, n_gen=0``) and one generation row (``n_prompt=0, n_gen=128``) -- never 3x the
row count.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from .lease import QuietCard

_DEFAULT_TIMEOUT_S = 300.0


class BenchError(RuntimeError):
    """llama-bench failed to run, exited non-zero, or its output could not be parsed."""


class BenchResult(BaseModel):
    """Measured throughput from one llama-bench run; ``None`` fields mean that row was absent."""

    prompt_tok_s: float | None
    decode_tok_s: float | None
    raw_rows: list[dict[str, Any]]


def _parse_rows(rows: list[dict[str, Any]]) -> BenchResult:
    prompt_tok_s: float | None = None
    decode_tok_s: float | None = None
    for row in rows:
        n_prompt = row.get("n_prompt", 0)
        n_gen = row.get("n_gen", 0)
        avg_ts = row.get("avg_ts")
        if avg_ts is None:
            continue
        if n_prompt and not n_gen:
            prompt_tok_s = float(avg_ts)
        elif n_gen and not n_prompt:
            decode_tok_s = float(avg_ts)
    return BenchResult(prompt_tok_s=prompt_tok_s, decode_tok_s=decode_tok_s, raw_rows=rows)


def bench_command(
    bench_bin: Path,
    model_path: Path,
    *,
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 3,
    extra_flags: tuple[str, ...] = (),
) -> list[str]:
    """``llama-bench -m model_path -p n_prompt -n n_gen -r repetitions -o json`` and ``extra_flags``."""
    return [
        str(bench_bin),
        *("-m", str(model_path), "-p", str(n_prompt), "-n", str(n_gen), "-r", str(repetitions)),
        *("-o", "json"),
        *extra_flags,
    ]


def parse_bench_output(stdout: str) -> BenchResult:
    """The throughput rows of llama-bench's ``-o json`` output.

    Raises:
        BenchError: the output is not llama-bench's JSON.
    """
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = f"could not parse llama-bench JSON output: {exc}"
        raise BenchError(msg) from exc
    return _parse_rows(rows)


def run_bench(
    bench_bin: Path,
    model_path: Path,
    *,
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 3,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    extra_flags: tuple[str, ...] = (),
    quiet_card: QuietCard | None = None,
) -> BenchResult:
    """Run ``llama-bench -m model_path -p n_prompt -n n_gen -r repetitions -o json`` once.

    ``quiet_card``, when given, is held around the run so that stay-up services have unloaded first.
    """
    cmd = bench_command(
        bench_bin,
        model_path,
        n_prompt=n_prompt,
        n_gen=n_gen,
        repetitions=repetitions,
        extra_flags=extra_flags,
    )
    try:
        with quiet_card if quiet_card is not None else contextlib.nullcontext():
            proc = subprocess.run(  # noqa: S603 — bench_bin is caller-resolved, not user input
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"failed to run {' '.join(cmd)}: {exc}"
        raise BenchError(msg) from exc
    if proc.returncode != 0:
        msg = f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
        raise BenchError(msg)
    return parse_bench_output(proc.stdout)
