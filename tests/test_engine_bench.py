"""The measuring harness against a stub engine: every phase, both teacher-forced shapes, the pinned inputs."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import cli, engine_bench
from localmodels.engine_bench import (
    TINY,
    Phase,
    PhaseError,
    PhaseStatus,
    PromptFileError,
    PromptPins,
    ScoringError,
    ScoringPath,
)

from .stub_engine import StubEngine, expected_logprobs, expected_top_tokens

if TYPE_CHECKING:
    from collections.abc import Iterator

BENCHY_JSON = {  # the shape of a real llama-benchy 0.4.0 file (sparkrun run of 2026-11-08), figures shortened
    "model": "m",
    "max_concurrency": 1,
    "benchmarks": [
        {
            "concurrency": 1,
            "context_size": 0,
            "prompt_size": 2048,
            "response_size": 32,
            "is_context_prefill_phase": False,
            "pp_throughput": {"mean": 3521.6, "std": 55.9, "values": [3577.6, 3465.7]},
            "tg_throughput": {"mean": 60.9, "std": 0.8, "values": [61.7, 60.1]},
            "ttfr": {"mean": 582.6, "std": 9.2, "values": [573.4, 591.9]},
        }
    ],
}


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.fixture
def prompts(tmp_path: Path) -> tuple[Path, PromptPins]:
    """A tiny pinned-inputs folder and the pins that match it."""
    folder = tmp_path / "prompts"
    folder.mkdir()
    judge = (
        "\n".join(json.dumps({"prompt": f"judge prompt number {i} " + "word " * 30}) for i in range(6)) + "\n"
    )
    text = " ".join(f"w{i}" for i in range(2000))
    _write(folder / engine_bench.JUDGE_FILE, judge)
    _write(folder / engine_bench.TEXT_FILE, text)
    pins = PromptPins(judge=engine_bench._sha256(judge.encode()), text=engine_bench._sha256(text.encode()))
    return folder, pins


@pytest.fixture
def stub() -> Iterator[StubEngine]:
    with StubEngine(delay=0.001) as server:
        yield server


def _fake_benchy(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    Path(cmd[cmd.index("--save-result") + 1]).write_text(json.dumps(BENCHY_JSON))
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _measure(stub: StubEngine, prompts: tuple[Path, PromptPins], engine: str = "vllm", **kw: Any) -> Any:
    folder, pins = prompts
    return engine_bench.measure(
        stub.url, engine, sizes=TINY, prompts_folder=folder, pins=pins, runner=_fake_benchy, **kw
    )


def test_every_phase_is_measured_against_the_stub(stub: StubEngine, prompts: tuple[Path, PromptPins]) -> None:
    record = _measure(
        stub, prompts, stamp={"image": "sha256:abc"}, startup_s=12.5, extra={"cache_prompt": False}
    )

    assert {p: o.status for p, o in record.phases.items()} == dict.fromkeys(Phase, PhaseStatus.MEASURED)
    assert (record.model, record.stamp, record.startup_s) == ("stub-model", {"image": "sha256:abc"}, 12.5)
    assert len(record.greedy_outputs) == TINY.quality_n
    assert all(len(t.split()) == 8 for t in record.greedy_outputs)  # stopped at the stub's end of turn
    f = record.figures
    assert all(f[k] for k in ("write_tps_1_decode_phase", "write_tps_1_at_once", "write_tps_2_at_once"))
    assert all(f[f"read_tps_prefill_{n}"] for n in TINY.prefill_tokens)
    assert all(f[k] for k in ("judge_total_s", "judge_write_tps", "judge_ttft_max_s"))
    assert (record.benchy.read_tps, record.benchy.write_tps, record.benchy.runs) == (3521.6, 60.9, 2)
    # the extra request fields travel as data
    completions = [b for path, b in stub.requests if path == "/v1/completions" and "stream" in b]
    assert completions
    assert all(b["cache_prompt"] is False for b in completions)
    # the record round-trips
    assert engine_bench.EngineRun.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("engine", "path", "field"),
    [("vllm", "/v1/completions", "prompt_logprobs"), ("sglang", "/generate", "return_logprob")],
)
def test_teacher_forced_scores_the_stub_on_both_shapes(
    stub: StubEngine, prompts: tuple[Path, PromptPins], engine: str, path: str, field: str
) -> None:
    record = _measure(stub, prompts, engine, phases=[Phase.TEACHER_FORCED])

    sent = [b for p, b in stub.requests if p == path and field in b]
    assert len(sent) == TINY.tf_chunks
    prompts_sent = [b.get("prompt") or b["text"] for b in sent]
    first_lines = [p.split("\n", 1)[0] for p in prompts_sent]
    assert len(set(first_lines)) == len(
        first_lines
    )  # a unique first line per chunk: no prefix cache can hide one
    expected = [lp for p in prompts_sent for lp in expected_logprobs(len(p.split()))]
    scored = record.teacher_forced
    assert scored.path is (
        ScoringPath.VLLM_PROMPT_LOGPROBS if engine == "vllm" else ScoringPath.SGLANG_GENERATE
    )
    assert scored.positions == len(expected)
    assert scored.perplexity == pytest.approx(math.exp(-sum(expected) / len(expected)))
    assert scored.top_tokens == [expected_top_tokens(len(p.split())) for p in prompts_sent]
    if engine == "sglang":
        assert (sent[0]["logprob_start_len"], sent[0]["top_logprobs_num"]) == (0, 1)
    else:
        assert sent[0]["prompt_logprobs"] == 1


def test_parsers_read_the_documented_shapes_and_refuse_others() -> None:
    vllm = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"5": {"logprob": -2.0, "rank": 3}, "9": {"logprob": -0.2, "rank": 1}},
                ]
            }
        ]
    }
    got = engine_bench.parse_vllm_prompt_logprobs(vllm)
    assert (got.logprobs, got.top_tokens) == ([-2.0], [9])  # own token = highest rank, top token = rank 1
    sglang = {
        "meta_info": {
            "input_token_logprobs": [[None, 1, None], [-1.0, 4, None]],
            "input_top_logprobs": [None, [[-0.3, 8, None], [-0.9, 4, None]]],
        }
    }
    got = engine_bench.parse_sglang_input_logprobs(sglang)
    assert (got.logprobs, got.top_tokens) == ([-1.0], [8])
    for parser, bad in (
        (engine_bench.parse_vllm_prompt_logprobs, {"choices": [{"text": "x"}]}),
        (engine_bench.parse_vllm_prompt_logprobs, {"choices": [{"prompt_logprobs": [None]}]}),
        (engine_bench.parse_sglang_input_logprobs, {"meta_info": {}}),
        (
            engine_bench.parse_sglang_input_logprobs,
            {
                "meta_info": {
                    "input_token_logprobs": [[None, 1, None], [-1.0, 4, None]],
                    "input_top_logprobs": [None, [[-0.3, 8, None]], [[-0.3, 8, None]]],
                }
            },
        ),
    ):
        with pytest.raises(ScoringError):
            parser(bad)


def test_a_failing_phase_is_recorded_as_not_measured(
    stub: StubEngine, prompts: tuple[Path, PromptPins]
) -> None:
    failing = StubEngine(fail_paths=("logprobs",))
    with failing:
        record = _measure(failing, prompts, "vllm", phases=[Phase.WARM, Phase.TEACHER_FORCED, Phase.DECODE])
    tf = record.phases[Phase.TEACHER_FORCED]
    assert (tf.status, record.teacher_forced) == (PhaseStatus.NOT_MEASURED, None)
    assert "400" in tf.reason
    assert record.phases[Phase.DECODE].status is PhaseStatus.MEASURED  # the other phases are unaffected

    no_path = _measure(stub, prompts, "llama.cpp", phases=[Phase.TEACHER_FORCED])
    assert no_path.phases[Phase.TEACHER_FORCED].status is PhaseStatus.NOT_MEASURED
    assert "no teacher-forced scoring" in no_path.phases[Phase.TEACHER_FORCED].reason


def test_a_server_that_cannot_answer_skips_the_slow_phases(prompts: tuple[Path, PromptPins]) -> None:
    with StubEngine(fail_paths=("/v1/completions",)) as dead:
        record = _measure(dead, prompts, "vllm", phases=[Phase.WARM, Phase.PREFILL, Phase.TEACHER_FORCED])
    assert {o.status for o in record.phases.values()} == {PhaseStatus.NOT_MEASURED}
    assert "warm-up request failed" in record.phases[Phase.PREFILL].reason
    unreachable = engine_bench.measure(
        "http://127.0.0.1:9", "vllm", sizes=TINY, phases=[Phase.WARM, Phase.DECODE]
    )
    assert {o.status for o in unreachable.phases.values()} == {PhaseStatus.NOT_MEASURED}


def test_a_changed_prompt_file_is_refused_by_its_hash(
    stub: StubEngine, prompts: tuple[Path, PromptPins], tmp_path: Path
) -> None:
    folder, pins = prompts
    _write(folder / engine_bench.TEXT_FILE, "w0 w1 changed")
    with pytest.raises(PromptFileError, match="not the pinned file"):
        engine_bench.load_prompts(folder, pins)
    record = _measure(stub, prompts, phases=[Phase.WARM, Phase.PREFILL, Phase.JUDGE, Phase.DECODE])
    assert record.phases[Phase.PREFILL].status is PhaseStatus.NOT_MEASURED
    assert "not the pinned file" in record.phases[Phase.PREFILL].reason
    assert record.phases[Phase.DECODE].status is PhaseStatus.MEASURED  # needs no pinned input

    docs, judge = tmp_path / "docs", tmp_path / "judge.jsonl"
    docs.mkdir()
    _write(docs / "a.md", "doc a")
    _write(judge, '{"prompt": "p"}\n')
    with pytest.raises(PromptFileError, match="nothing installed"):  # a source that is not what D16 used
        engine_bench.install_prompts(tmp_path / "out", docs_dir=docs, judge_source=judge)
    assert not (tmp_path / "out").exists()
    good = PromptPins(judge=engine_bench._sha256(judge.read_bytes()), text=engine_bench._sha256(b"doc a"))
    written = engine_bench.install_prompts(tmp_path / "out", docs_dir=docs, judge_source=judge, pins=good)
    assert sorted(p.name for p in written) == sorted([engine_bench.JUDGE_FILE, engine_bench.TEXT_FILE])
    assert engine_bench.load_prompts(tmp_path / "out", good).judge == ("p",)


def test_benchy_command_is_pinned_and_its_json_read(tmp_path: Path) -> None:
    cmd = engine_bench.benchy_command("http://h:1", "m", tmp_path / "r.json")
    assert cmd[:2] == ["uvx", "llama-benchy==0.4.0"]
    flags = {cmd[i]: cmd[i + 1] for i in range(2, len(cmd), 2)}
    assert flags == {
        "--base-url": "http://h:1",
        "--model": "m",
        "--pp": "2048",
        "--tg": "32",
        "--runs": "3",
        "--format": "json",
        "--save-result": str(tmp_path / "r.json"),
    }
    seen: list[list[str]] = []

    def failing(c: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(c)
        return subprocess.CompletedProcess(c, 2, "", "no such host")

    with pytest.raises(PhaseError, match="exited 2: no such host"):
        engine_bench.phase_benchy("http://h:1", "m", failing)
    assert seen[0][1] == "llama-benchy==0.4.0"
    with pytest.raises(PhaseError, match="not in the expected shape"):
        engine_bench.parse_benchy({"benchmarks": [{"concurrency": 4}]})


def test_command_line_measures_and_installs(
    stub: StubEngine,
    prompts: tuple[Path, PromptPins],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder, pins = prompts
    state_prompts = tmp_path / "state" / "engines" / "prompts"
    shutil.copytree(folder, state_prompts)
    monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(engine_bench, "PINS", pins)
    out = tmp_path / "run.json"
    code = cli.main(
        [
            "engines",
            "measure",
            "--url",
            stub.url,
            "--engine",
            "sglang",
            "--out",
            str(out),
            "--tiny",
            "--phases",
            "warm,decode,teacher_forced",
            "--stamp",
            "image=sha256:abc",
            "--startup-s",
            "3",
        ]
    )
    assert code == 0
    record = engine_bench.EngineRun.model_validate_json(out.read_text())
    assert (record.stamp, record.startup_s, record.teacher_forced is not None) == (
        {"image": "sha256:abc"},
        3.0,
        True,
    )
    assert "teacher_forced: measured" in capsys.readouterr().out
