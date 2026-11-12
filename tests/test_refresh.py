"""`localmodels refresh`: the verdict, the newest tag, `run`, `build`, `measure` and the digest line.

No llama.cpp program runs here: a build folder holds wrappers around ``fixtures/refresh/fake_llama.py``, which
prints recorded output. Slurm is a script that records its arguments; the lease folder is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pytest

from localmodels import digest, engine_bench, engine_refresh, refresh, rules
from localmodels.atomic import read_model, write_model, write_text_atomically
from localmodels.cli import main
from localmodels.container_run import ContainerStartError, MemoryNeed
from localmodels.digest import Context
from localmodels.engines import (
    GateVerdict,
    Publisher,
    Response,
    TagListing,
    VersionRecord,
    link_target,
    read_record,
    tried_tags,
    versions_dir,
    write_record,
)
from localmodels.launcher import NotEnoughMemoryError
from localmodels.llama_tune import KlStats
from localmodels.stack import Engine

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FAKE = Path(__file__).parent / "fake_llama.py"
HOST = {
    "driver": "590.2",
    "cuda": "13.0",
    "kernel": "7.0",
    "dgx_os": "7.6.0",
    "localmodels": "0.10.0",
    "llama_cpp": "a1b2c3d4e",
}
CURRENT = "a1b2c3d4e"
CANDIDATE = "f00dcafe1"
CANDIDATE_SHA = CANDIDATE + "0" * 31
TAGS = f"{'1' * 40}\trefs/tags/b11\n{'2' * 40}\trefs/tags/b12\n{CANDIDATE_SHA}\trefs/tags/b12^{{}}\n"
SAME = KlStats(ppl_ratio=1.0, mean_kl=0.00005, same_top=1.0)
OCTOBER = KlStats(ppl_ratio=1.0029, mean_kl=0.012, same_top=0.9868)


def _rounds(**figures: list[float]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Two rounds of baseline and candidate figures: every figure 100 unless given as [r1, r2]."""
    base = {name: [100.0, 100.0] for name in refresh.SPEED_FIGURES}
    cand = {name: [100.0, 100.0] for name in refresh.SPEED_FIGURES}
    for key, values in figures.items():
        (cand if key.startswith("c_") else base)[key[2:]] = values
    return (
        [{n: v[i] for n, v in base.items()} for i in range(2)],
        [{n: v[i] for n, v in cand.items()} for i in range(2)],
    )


def test_verdict_applies_the_speed_tolerance_the_kl_limits_and_fails_what_was_not_measured() -> None:
    base, cand = _rounds(b_small_tg128=[65.9, 64.9], c_small_tg128=[64.3, 65.0])  # October: 1.1 % slower
    october = refresh.judge_candidate(base, cand, SAME, OCTOBER, {})
    assert october.passed
    assert october.slowdown == pytest.approx(0.011, abs=0.001)
    base, cand = _rounds(c_small_tg128=[95.0, 95.0])  # 5 % slower, no spread
    slow = refresh.judge_candidate(base, cand, SAME, OCTOBER, {})
    assert not slow.passed
    assert slow.speed_failures[0].startswith("small_tg128 5.0 % slower")
    base, cand = _rounds(b_small_tg128=[100.0, 104.0], c_small_tg128=[98.0, 98.0])  # 3.9 % slower, spread 4 %
    assert refresh.judge_candidate(base, cand, SAME, OCTOBER, {}).passed
    base, cand = _rounds()
    noisy = KlStats(ppl_ratio=1.0, mean_kl=0.04, same_top=0.99)
    assert refresh.judge_candidate(base, cand, SAME, noisy, {}).quality_failures  # KL 0.04 is over 0.03
    accepted = KlStats(
        ppl_ratio=1.0014, mean_kl=0.0036, same_top=0.9763
    )  # October's switch on this text (2159)
    assert refresh.judge_candidate(base, cand, SAME, accepted, {}).passed
    flips = KlStats(ppl_ratio=1.0, mean_kl=0.005, same_top=0.965)
    assert refresh.judge_candidate(base, cand, SAME, flips, {}).quality_failures  # 96.5 % is under 97 %
    missing = refresh.judge_candidate(base, cand, None, None, {"cmp-k3-control": "exited 3"})
    assert missing.quality_failures == ["not measured: cmp-k3-control: exited 3"]
    assert not refresh.judge_candidate(base, cand[:1], SAME, OCTOBER, {}).passed  # a missing round
    base, cand = _rounds(c_small_tg128=[float("nan"), float("nan")])
    assert not refresh.judge_candidate(base, cand, SAME, OCTOBER, {}).passed  # NaN is no measurement


def test_newest_tag_prefers_the_peeled_line_and_ignores_other_tags() -> None:
    text = f"{'3' * 40}\trefs/tags/backup/x\n{'4' * 40}\trefs/tags/b9999-rc\n{TAGS}"
    assert refresh.newest_tag(text) == refresh.Candidate("b12", CANDIDATE_SHA)
    assert refresh.newest_tag(f"{'5' * 40}\trefs/tags/b7") == refresh.Candidate("b7", "5" * 40)
    assert refresh.newest_tag("garbage") is None


class World:
    """A llama home with a current build, a Hub cache with both models, state folders and a fake slurm."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.patch = monkeypatch
        self.home = root / "llama"
        self.run_dir = root / "refresh" / "2026-11-01T0900"
        self.submitted = root / "submitted.txt"
        script = root / "submit.py"
        script.write_text(
            "import pathlib, sys\n"
            "workdir = pathlib.Path(sys.argv[sys.argv.index('--workdir') + 1])\n"
            # sbatch --chdir needs the folder to exist when the job is queued
            "if not workdir.is_dir():\n    sys.exit('the --workdir does not exist')\n"
            f"log = pathlib.Path({str(self.submitted)!r})\n"
            "n = len(log.read_text().splitlines()) if log.exists() else 0\n"
            "log.open('a').write(' '.join(sys.argv[1:]) + '\\n')\n"
            "print(101 + n)\n"
        )
        monkeypatch.setenv("LOCALMODELS_SUBMIT", str(script))
        monkeypatch.setenv("LOCALMODELS_LLAMA_HOME", str(self.home))
        monkeypatch.setenv("LOCALMODELS_STATE_DIR", str(root / "state"))
        monkeypatch.setenv("LOCALMODELS_REFRESH_DIR", str(root / "refresh"))
        monkeypatch.setenv("HF_HUB_CACHE", str(root / "hub"))
        monkeypatch.setattr(refresh, "DEFAULT_LEASE_DIR", root / "lease")
        monkeypatch.setattr(rules, "host_stamp", lambda _binary: HOST.copy())
        self.cancelled: list[str] = []
        monkeypatch.setattr(refresh, "_refresh_queue", list)
        monkeypatch.setattr(refresh, "_scancel", lambda job: self.cancelled.append(job) or "")
        monkeypatch.setattr(refresh, "_ls_remote", lambda: TAGS)
        from localmodels.pinned import snapshot_dir  # noqa: PLC0415

        for pin in (refresh.SMALL_MODEL, refresh.PRODUCTION_MODEL):
            model = snapshot_dir(pin.repository, pin.revision, root / "hub") / pin.filename
            model.parent.mkdir(parents=True)
            model.touch()
        self.make_build(CURRENT)
        (self.home / "builds" / "current").symlink_to(CURRENT)

    def make_build(self, name: str, **config: object) -> Path:
        build = self.home / "builds" / name
        (build / "bin").mkdir(parents=True)
        for program in refresh.Tool:
            wrapper = build / "bin" / program
            wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE} {program} "$0" "$@"\n')
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        (build / "bin" / "llama-server").touch()
        (build / "fake.json").write_text(json.dumps(config))
        return build

    def configure(self, name: str, **config: object) -> None:
        (self.home / "builds" / name / "fake.json").write_text(json.dumps(config))

    def build_done(self, name: str = CANDIDATE) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        done = refresh.BuildResult(tag="b12", commit=name, ok=True)
        (self.run_dir / refresh.BUILD_FILE).write_text(done.model_dump_json())

    def measure(self) -> int:
        return main(["refresh", "--home", str(self.home), "measure", "--run", str(self.run_dir)])

    def current(self) -> str | None:
        return (self.home / "builds" / "current").resolve().name

    def result(self) -> refresh.MeasureResult:
        return refresh.MeasureResult.model_validate_json((self.run_dir / refresh.RESULT_FILE).read_text())


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def test_run_submits_build_then_measure_after_it_and_skips_when_nothing_is_due(  # noqa: PLR0915 - one run after another
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    world.make_build("aaaaaaaaa")  # not current, not previous, not named: removed
    world.make_build("bbbbbbbbb")  # the previous current: kept for a rollback
    refresh.write_state(refresh.RefreshState(previous_current="bbbbbbbbb"))
    args = ["refresh", "--home", str(world.home), "run"]
    assert main([*args, "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert "would remove" in dry
    assert "refresh-build" in dry
    assert "--after '<build job id>'" in dry
    assert not world.submitted.exists()
    assert (world.home / "builds" / "aaaaaaaaa").exists()  # a dry run changes nothing

    assert main(args) == 0
    run_dir = refresh.newest_run()  # named by the clock
    assert run_dir is not None
    build, measure, same_client = world.submitted.read_text().splitlines()
    assert "--class resume --resume-ok --no-anywhere --cpus 8 --mem 8G" in build
    assert "--name refresh-build" in build
    assert f"refresh --home {world.home} build --run {run_dir} --commit b12" in build
    assert "--class restart --gpu --mem 52G --after 101" in measure
    assert f"--workdir {run_dir}" in measure
    assert "timeout --kill-after=30 1200" in measure
    assert (
        "--name refresh-same-client" in same_client
    )  # llama.cpp's current may change: measured on the client
    assert "--class restart --gpu --mem 52G --after 102" in same_client
    assert "timeout --kill-after=180 600" in same_client
    record = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert [job.job for job in record.jobs] == ["101", "102", "103"]
    assert record.candidate_commit == CANDIDATE_SHA
    assert not (world.home / "builds" / "aaaaaaaaa").exists()
    assert (world.home / "builds" / "bbbbbbbbb").exists()
    assert record.removed_builds == [str(world.home / "builds" / "aaaaaaaaa")]

    world.submitted.unlink()
    world.patch.setattr(refresh, "_refresh_queue", lambda: [("900", "None")])
    assert main(args) == 0
    assert not world.submitted.exists()
    assert "900 still queued" in capsys.readouterr().out
    kept = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert (kept.skipped, len(kept.jobs)) == ("", 3)  # a skipped run writes nothing over the run in progress
    world.patch.setattr(refresh, "_refresh_queue", lambda: None)
    assert main(args) == 0
    assert "squeue did not answer" in capsys.readouterr().out
    assert not world.submitted.exists()
    refresh.state_path().write_text("{not json")
    world.patch.setattr(refresh, "_refresh_queue", list)
    assert main(args) == 0
    assert "cannot be read" in capsys.readouterr().out
    assert not world.submitted.exists()
    refresh.state_path().unlink()
    world.patch.setattr(refresh, "_refresh_queue", lambda: [("901", "DependencyNeverSatisfied")])

    world.patch.setattr(refresh, "_ls_remote", lambda: f"{CURRENT}{'0' * 31}\trefs/tags/b13")
    seen = engine_bench.EngineRun(
        engine="llama.cpp", model="m", url="u", started="t"
    )  # same-client is answered
    gguf = engine_refresh.SAME_CLIENT_FILES[engine_refresh.MEASUREMENT_MODEL]
    write_model(engine_refresh.same_client_path(CURRENT, gguf), seen)
    for kind in refresh.REMEASURED_KINDS:  # answered: a holding record on this stack
        rules.write_record(
            rules.RuleRecord(
                kind=kind.value, measured_at="2026-11-01", stamp=dict(HOST), figures={}, holds=True
            )
        )
    assert main(args) == 0
    assert not world.submitted.exists()
    assert "nothing to do" in capsys.readouterr().out
    assert world.cancelled == ["901"]  # it waited on a build job that died: it could never start


def test_run_measures_a_candidate_built_ahead_and_urgent_goes_through_urgent_py(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["refresh", "--home", str(world.home), "run"]
    urgent_log = world.root / "urgent.txt"
    (world.root / "urgent.py").write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(urgent_log)!r}).write_text(' '.join(sys.argv[1:]))\n"
        "print('[urgent] memory: fits')\nprint('[urgent] job 777 submitted; log x')\n"
    )
    assert (
        main([*args, "--urgent"]) == 1
    )  # the newest candidate (b12) is not built: refused, nothing submitted
    assert "--urgent needs b12 built first" in capsys.readouterr().out
    assert not world.submitted.exists()
    assert not urgent_log.exists()

    world.make_build(CANDIDATE)  # built ahead of the run, never compared, not named: kept, not rebuilt
    assert main([*args, "--urgent"]) == 0
    assert (world.home / "builds" / CANDIDATE).exists()
    # no build job, and the measure job went to urgent.py: only the same-client job went to submit.py
    assert "--name refresh-same-client" in world.submitted.read_text()
    assert "--after 777" in world.submitted.read_text()
    world.submitted.unlink()
    urgent = urgent_log.read_text()
    assert "--name refresh-measure" in urgent
    assert "--time 1:30:00 --gpu --mem 52G --wait" in urgent
    assert "--after" not in urgent
    run_dir = refresh.newest_run()
    assert run_dir is not None
    built = refresh.BuildResult.model_validate_json((run_dir / refresh.BUILD_FILE).read_text())
    assert (built.tag, built.commit, built.ok) == ("b12", CANDIDATE, True)
    record = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert [(job.name, job.job) for job in record.jobs] == [
        ("refresh-measure", "777"),
        ("refresh-same-client", "101"),
    ]

    def offline() -> str:
        raise OSError("no network")

    world.make_build(
        "eeeeeeeee"
    )  # unnamed: removed by a run that read the release list, kept by one that did not
    world.patch.setattr(refresh, "_ls_remote", offline)
    assert main([*args, "--force-measure"]) == 0
    assert (world.home / "builds" / "eeeeeeeee").exists()
    world.patch.setattr(refresh, "_ls_remote", lambda: TAGS + f"{'9' * 40}\trefs/tags/b13\n")
    assert main([*args, "--dry-run", "--candidate", "b12"]) == 0
    assert "b12 is already built" in capsys.readouterr().out


def test_build_exits_zero_when_a_step_fails_and_says_so_in_build_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    script = "#!/bin/sh\n{}\n"
    (tools / "git").write_text(
        script.format(
            '[ "$1" = clone ] && exit "${GIT_FAIL:-0}"; [ "$3" = rev-parse ] && echo abc1234; exit 0'
        )
    )
    (tools / "cmake").write_text(  # configure is `-S src -B build ...`, compile is `--build build ...`
        script.format(
            '[ "$1" = -S ] && mkdir -p "$4/bin"; [ -n "$CMAKE_FAIL" ] && exit 1; '
            '[ "$1" = --build ] && touch "$2/bin/llama-server"; exit 0'
        )
    )
    for tool in tools.iterdir():
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    (tmp_path / "CMakeCache.txt").write_text("GGML_CUDA:BOOL=ON\n")
    cases = (({}, True), ({"GIT_FAIL": "1"}, False), ({"CMAKE_FAIL": "1"}, False))
    for number, (fail, ok) in enumerate(cases):
        for key, value in fail.items():
            monkeypatch.setenv(key, value)
        run = tmp_path / f"run-{number}"
        run.mkdir()
        args = ["refresh", "--home", str(tmp_path / f"home-{number}"), "build", "--reference", str(tmp_path)]
        assert main([*args, "--run", str(run), "--commit", "b12"]) == 0
        built = refresh.BuildResult.model_validate_json((run / refresh.BUILD_FILE).read_text())
        assert built.ok is ok
        assert bool(built.error) is not ok
        monkeypatch.delenv("GIT_FAIL", raising=False)
        monkeypatch.delenv("CMAKE_FAIL", raising=False)
        if ok:
            assert built.commit == "abc1234"
    gone = tmp_path / "CMakeCache.txt" / "run"  # a file, so build.json cannot be written: the one error exit
    assert main([*args, "--run", str(gone), "--commit", "b12"]) == 1


def test_measure_passing_candidate_switches_records_and_deletes_the_base(world: World) -> None:
    world.configure(CURRENT, speed=1.0)
    world.make_build(CANDIDATE, speed=0.995, mean_kl=0.012, same_top=98.68, ppl_ratio=1.0029)
    world.build_done()
    assert world.measure() == 0
    assert world.current() == CANDIDATE
    comparison = json.loads((world.home / "builds" / CANDIDATE / "comparison.json").read_text())
    assert comparison["passed"] is True
    assert comparison["baseline_commit"] == CURRENT
    assert comparison["candidate_tag"] == "b12"
    assert set(comparison["figures"]) == set(refresh.SPEED_FIGURES)
    assert (world.run_dir / "comparison.json").is_file()
    state = refresh.read_state()
    assert state.previous_current == CURRENT
    assert state.tried[CANDIDATE].passed
    result = world.result()
    assert result.switched
    assert result.records == ["K1", "K2", "K3"]
    assert set(result.steps.values()) == {"ok"}
    assert not (world.run_dir / refresh.KL_BASE_FILE).exists()
    k1, k2, k3 = (rules.read_record(kind) for kind in ("K1", "K2", "K3"))
    assert k1 is not None
    assert k1.holds
    assert k1.stamp["llama_cpp"] == CURRENT  # the patched host stamp; the real one reads the final current
    assert k2 is not None
    assert k2.holds
    assert k2.figures["prod_gain_16_over_4"] > 1
    assert k3 is not None
    assert k3.holds
    assert k3.figures["mean_kl"] == pytest.approx(0.00005)  # the new current repeated against its own base
    view = refresh.newest_view()
    assert view is not None
    assert refresh.headline(view).startswith(
        "refresh 2026-11-01: llama.cpp b12 (f00dcafe1) switched (slowdown 0.5 %"
    )
    assert refresh.rules_line(view) == "rules re-measured: K1 K2 K3; with a reason: K5 K6 K8 K9"
    lines = digest.check_refresh(Context(None, ""))  # type: ignore[arg-type]
    assert [f.text for f in lines] == [refresh.headline(view), refresh.rules_line(view)]


def test_measure_failing_candidate_keeps_current_and_failed_build_skips_the_comparison(world: World) -> None:
    world.configure(CURRENT, speed=1.0)
    world.make_build(CANDIDATE, speed=1.0, tg=0.9)  # generation 10 % slower
    world.build_done()
    assert world.measure() == 0
    assert world.current() == CURRENT
    comparison = world.result().comparison
    assert comparison is not None
    assert not comparison.passed
    assert comparison.speed_failures[0].startswith("small_tg128 10.0 % slower")
    assert refresh.read_state().tried[CANDIDATE].passed is False
    assert refresh.read_state().previous_current is None
    assert not (world.run_dir / refresh.KL_BASE_FILE).exists()
    view = refresh.newest_view()
    assert view is not None
    assert "kept back: small_tg128 10.0 % slower" in refresh.headline(view)

    failed = world.run_dir.parent / "2026-12-01T0900"
    failed.mkdir()
    (failed / refresh.BUILD_FILE).write_text(
        refresh.BuildResult(tag="b13", ok=False, error="nvcc died").model_dump_json()
    )
    assert main(["refresh", "--home", str(world.home), "measure", "--run", str(failed)]) == 0
    result = refresh.MeasureResult.model_validate_json((failed / refresh.RESULT_FILE).read_text())
    assert result.comparison is None
    assert result.records == ["K1", "K2", "K3"]  # the rules are still re-measured on current
    assert (
        refresh.headline(refresh.newest_view()) == "refresh 2026-12-01: llama.cpp b13 build failed: nvcc died"
    )  # type: ignore[arg-type]


def test_measure_records_a_step_that_failed_and_the_rule_that_no_longer_holds(world: World) -> None:
    world.configure(CURRENT, speed=1.0, fail=["llama-batched-bench"])
    world.run_dir.mkdir(parents=True)
    assert world.measure() == 0
    steps = world.result().steps
    assert steps["r1-current-batched"] == "not measured: exited 3"
    assert steps["r1-current-bench"] == "ok"
    k2 = rules.read_record("K2")
    assert k2 is not None
    assert not k2.holds
    assert "not measured" in k2.detail
    assert (world.run_dir / "r1-current-batched.err").read_text().strip() == "fake failure"

    world.configure(
        CURRENT, speed=1.0
    )  # a comparison whose candidate cannot score: the verdict names the step
    world.make_build(CANDIDATE, speed=1.0, fail=["llama-perplexity"])
    world.build_done()
    assert world.measure() == 0
    comparison = world.result().comparison
    assert comparison is not None
    assert comparison.quality_failures == ["not measured: cmp-k3-candidate: exited 3"]
    assert world.current() == CURRENT


def test_measure_stops_when_the_card_does_not_go_quiet(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "still loaded, so not measuring"
        raise refresh.QuietCardTimeoutError(msg)

    monkeypatch.setattr(refresh.QuietCard, "__enter__", refuse)
    world.run_dir.mkdir(parents=True)
    assert world.measure() == 1
    assert world.result().not_measured == "still loaded, so not measuring"
    assert rules.read_record("K1") is None


def test_a_broken_engine_walk_never_stops_the_llama_cpp_refresh(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_self: object) -> None:
        msg = "registry layout changed"
        raise RuntimeError(msg)

    monkeypatch.setattr(engine_refresh.EngineWalk, "decide", broken)
    assert main(["refresh", "--home", str(world.home), "run"]) == 0
    build, measure = world.submitted.read_text().splitlines()  # llama.cpp's jobs went; no engine job
    assert ("--name refresh-build" in build, "--name refresh-measure" in measure) == (True, True)
    run_dir = refresh.newest_run()
    assert run_dir is not None
    record = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert record.engine_reports == [
        "engine walk failed while deciding: RuntimeError: registry layout changed"
    ]


def test_status_prints_the_newest_run(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["refresh", "status"]) == 0
    assert "no refresh run" in capsys.readouterr().out
    world.run_dir.mkdir(parents=True)
    (world.run_dir / refresh.RUN_FILE).write_text(
        refresh.RunRecord(time="t", skipped="nothing").model_dump_json()
    )
    assert main(["refresh", "status"]) == 0
    assert capsys.readouterr().out.startswith("refresh 2026-11-01: skipped, nothing\n")


# ---------------------------------------------------------------------------------------------------------
# Container engines: gates, calibration, the jobs, the walk (no network, Docker or GPU: fakes throughout)

SPEED = engine_refresh.speed_figure_names()
MODEL = engine_refresh.MEASUREMENT_MODEL
DIGEST = "sha256:" + "ab" * 32
ALL_PHASES = tuple(engine_bench.Phase)


def tf(
    *, ppl: float = 2.0, diff: int = 0, chunks: int = 1, positions: int = 1000
) -> engine_bench.TeacherForced:
    """A teacher-forced score: ``diff`` positions of the first chunk have another top token."""
    tops = [[1] * positions for _ in range(chunks)]
    tops[0][:diff] = [2] * diff
    return engine_bench.TeacherForced(
        path=engine_bench.ScoringPath.VLLM_PROMPT_LOGPROBS,
        chunks=chunks,
        positions=positions * chunks,
        mean_logprob=-0.7,
        perplexity=ppl,
        top_tokens=tops,
    )


def erun(
    figures: dict[str, float] | None = None,
    *,
    drop: tuple[str, ...] = (),
    score: engine_bench.TeacherForced | None = None,
    scored: bool = True,
    unmeasured: tuple[engine_bench.Phase, ...] = (),
    phases: tuple[engine_bench.Phase, ...] = ALL_PHASES,
) -> engine_bench.EngineRun:
    """One measurement: every speed figure 100 unless given or dropped."""
    figs: dict[str, float | None] = {name: 100.0 for name in SPEED if name not in drop}
    figs.update(figures or {})
    outcomes = {
        phase: engine_bench.PhaseOutcome(
            status=engine_bench.PhaseStatus.NOT_MEASURED
            if phase in unmeasured
            else engine_bench.PhaseStatus.MEASURED,
            reason="boom" if phase in unmeasured else None,
        )
        for phase in phases
    }
    return engine_bench.EngineRun(
        engine="vllm",
        model="m",
        url="u",
        started="t",
        figures=figs,
        phases=outcomes,
        teacher_forced=(score or tf()) if scored else None,
    )


def limits(engine: Engine = Engine.VLLM, ppl: float = 1.02, top: float = 0.97) -> engine_refresh.EngineLimits:
    return engine_refresh.EngineLimits(
        engine=engine, ppl_ratio_max=ppl, same_top_min=top, calibrated_on="a", adjacent="b", arms={}
    )


def test_container_gate_fails_each_limit_once_and_records_only_without_limits() -> None:
    judge = engine_refresh.judge_engine
    cur = [erun(), erun()]
    ok = judge(cur, [erun(), erun()], limits())
    assert ok.gate is GateVerdict.PASSED
    assert ok.noise == {"ppl_ratio": 1.0, "same_top": 1.0}  # current against itself, for the record

    slow = judge(cur, [erun({"write_tps_16_at_once": 90.0}), erun({"write_tps_16_at_once": 90.0})], limits())
    assert slow.gate is GateVerdict.FAILED
    assert slow.speed_failures[0].startswith("write_tps_16_at_once 10.0 % slower")
    assert slow.slowdown == pytest.approx(0.10)

    gone = judge(cur, [erun(drop=("judge_write_tps",)), erun()], limits())  # a figure missing from one round
    assert gone.gate is GateVerdict.FAILED
    assert gone.speed_failures == ["not measured: judge_write_tps"]

    unscored = judge(cur, [erun(scored=False), erun()], limits())
    assert (unscored.gate, unscored.quality_failures) == (
        GateVerdict.FAILED,
        ["teacher-forced scoring not measured"],
    )
    assert judge([erun(scored=False)] * 2, [erun()] * 2, None).gate is GateVerdict.FAILED  # not RECORD_ONLY

    over = judge(cur, [erun(score=tf(ppl=2.1)), erun()], limits())  # ratio 1.05 against a limit of 1.02
    assert over.gate is GateVerdict.FAILED
    assert over.quality_failures[0].startswith("perplexity ratio 1.0500 over")
    flips = judge(cur, [erun(), erun(score=tf(diff=50))], limits())  # 95 % same top against 97 %
    assert flips.gate is GateVerdict.FAILED
    assert flips.quality["same_top"] == pytest.approx(0.95)
    assert flips.quality_failures[0].startswith("same top token 0.9500 under")

    apart = judge(cur, [erun(score=tf(chunks=2)), erun()], limits())
    assert "not comparable" in apart.quality_failures[0]
    with pytest.raises(engine_refresh.NotComparableError):
        engine_refresh.compare_scores(tf(positions=10), tf(positions=11))
    with pytest.raises(engine_refresh.NotComparableError):  # a NaN would pass both limits silently
        engine_refresh.compare_scores(tf(), tf(ppl=float("nan")))
    compared = engine_refresh.compare_scores(tf(), tf(ppl=2.2, diff=10))
    assert (compared.ppl_ratio, compared.same_top, compared.positions) == (
        pytest.approx(1.1),
        pytest.approx(0.99),
        1000,
    )

    recorded = judge(cur, [erun(), erun()], None)  # no limits yet: reported, not judged
    assert (recorded.gate, recorded.quality) == (GateVerdict.RECORD_ONLY, {"ppl_ratio": 1.0, "same_top": 1.0})
    assert (
        judge(cur, [erun({"write_tps_1_at_once": 50.0})] * 2, None).gate is GateVerdict.FAILED
    )  # speed first


def test_the_speed_figures_follow_the_harness_sizes() -> None:
    assert SPEED == (
        "write_tps_1_decode_phase",
        "write_tps_1_at_once",
        "write_tps_16_at_once",
        "read_tps_prefill_512",
        "read_tps_prefill_2048",
        "read_tps_prefill_8192",
        "judge_write_tps",
    )
    assert "read_tps_prefill_64" in engine_refresh.speed_figure_names(engine_bench.TINY)


def test_bootstrap_passes_only_when_every_phase_was_measured() -> None:
    assert engine_refresh.bootstrap_verdict(erun()).gate is GateVerdict.PASSED
    failed = engine_refresh.bootstrap_verdict(erun(unmeasured=(engine_bench.Phase.TEACHER_FORCED,)))
    assert failed.gate is GateVerdict.FAILED
    assert failed.speed_failures == ["not measured: teacher_forced: boom"]
    short = engine_refresh.bootstrap_verdict(erun(phases=(engine_bench.Phase.WARM,)))
    assert short.gate is GateVerdict.FAILED  # a phase that never ran is not measured either
    assert engine_refresh.bootstrap_verdict(None, "boom").speed_failures == ["not measured: boom"]


def test_calibrated_limits_follow_design_4_3() -> None:
    arm = engine_refresh.ArmScore
    noise = {
        "restart": arm(ppl_ratio=1.001, same_top=0.9995),
        "empty-cache": arm(ppl_ratio=1.004, same_top=0.998),
        "autotune-off": arm(ppl_ratio=1.002, same_top=0.999),
    }
    ppl_max, top_min = engine_refresh.calibrated_limits(
        {**noise, "adjacent": arm(ppl_ratio=1.01, same_top=0.9763)}
    )
    assert ppl_max == pytest.approx(1.014)  # the noise arms' largest + 0.01
    assert top_min == pytest.approx(0.9703)  # the adjacent pair's figure minus 0.6 points
    _, capped = engine_refresh.calibrated_limits({**noise, "adjacent": arm(ppl_ratio=1.0, same_top=1.0)})
    assert capped == pytest.approx(0.994)
    lower = {**noise, "restart": arm(ppl_ratio=1.0, same_top=0.99)}
    _, never_above = engine_refresh.calibrated_limits({**lower, "adjacent": arm(ppl_ratio=1.0, same_top=1.0)})
    assert never_above == pytest.approx(0.99)  # and never above the noise arms' lowest


def image_of(engine: Engine, tag: str) -> str:
    """A digest of its own for each version: prune keeps an image that a kept version shares."""
    return f"repo/{engine.value}@sha256:{hashlib.sha256(tag.encode()).hexdigest()}"


def put(
    engine: Engine,
    tag: str,
    *,
    gate: GateVerdict = GateVerdict.NOT_RUN,
    publisher: Publisher = Publisher.OFFICIAL,
    pulled: bool = True,
    current: bool = False,
    image: bool = True,
) -> VersionRecord:
    record = VersionRecord(
        engine=engine,
        tag=tag,
        image=image_of(engine, tag) if image else "",
        publisher=publisher,
        gate=gate,
        pulled=pulled,
        recipes={MODEL: json.dumps(engine_refresh.BASE_RECIPES[engine])},
    )
    write_record(record)
    if current:
        (versions_dir(engine) / "current").symlink_to(tag)
    return record


def approved_recipe(engine: Engine, model: str = "org/approved") -> None:
    """A recipe file for a second model, as an approval installs one."""
    recipe = {**engine_refresh.BASE_RECIPES[engine], "model": model}
    write_text_atomically(engine_refresh.recipe_file(engine, model), json.dumps(recipe))


class Served:
    """What ``ContainerRun`` offers a job once the server is up."""

    def __init__(self, fleet: Fleet, tag: str) -> None:
        self.fleet, self.tag, self.run_id, self.killed = fleet, tag, f"run{len(fleet.starts)}", None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def stamp(self) -> dict[str, str]:
        return {"tag": self.tag}

    def memory(self) -> MemoryNeed:
        return MemoryNeed(fall_gib=30.0, gpu_peak_gib=20.0, host_peak_gib=1.0)


class Refused:
    """A start the memory check refused before ``sparkrun run`` was called."""

    launched = False

    def __init__(self, number: int) -> None:
        self.number = number

    def __enter__(self) -> Self:
        msg = f"start {self.number}: not enough memory available"
        raise NotEnoughMemoryError(msg)

    def __exit__(self, *_exc: object) -> None:
        return None


class Fleet:
    """Fake container starts and measurements: every start is logged, each can be scripted by its number."""

    def __init__(self) -> None:
        self.starts: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_on: set[int] = set()
        self.refuse_on: set[int] = set()  # refused before sparkrun ran (memory short at the start)
        self.by_start: dict[int, dict[str, Any]] = {}  # start number -> erun() keywords
        self.by_tag: dict[str, dict[str, Any]] = {}
        self.removed: list[str] = []
        self.tokenizers: set[str] = set()
        self.number = -1

    def harness(self) -> engine_refresh.Harness:
        return engine_refresh.Harness(
            start=self.start,
            measure=self.measure,
            sparkrun=lambda _version: Path("sparkrun"),
            docker=self.docker,
        )

    def start(self, record, model, recipe, binary, guards, *, kind):
        number = len(self.starts)
        self.starts.append((record.tag, kind.value, recipe))
        if number in self.fail_on:
            msg = f"start {number} failed"
            raise ContainerStartError(msg)
        return Refused(number) if number in self.refuse_on else Served(self, record.tag)

    def measure(self, url, engine, *, stamp, startup_s, phases, tokenizer):
        self.number += 1
        self.tokenizers.add(tokenizer)
        return erun(
            **{**self.by_tag.get(stamp["tag"], {}), **self.by_start.get(self.number, {}), "phases": phases}
        )

    def docker(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if command[1] == "rmi":
            self.removed.append(command[2])
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def fleet(world: World, monkeypatch: pytest.MonkeyPatch) -> Fleet:
    monkeypatch.setattr(engine_refresh, "DEFAULT_LEASE_DIR", world.root / "lease")
    engine_refresh.install_recipes()  # the pull job does this before it realizes them
    return Fleet()


def measure_job(
    world: World, fleet: Fleet, engine: Engine, tag: str, mode: engine_refresh.MeasureMode, **kw: Any
) -> engine_refresh.EngineMeasureResult:
    job = engine_refresh.MeasureJob(world.run_dir, engine, tag, mode, harness=fleet.harness(), **kw)
    assert job.execute() == 0
    done = read_model(job.folder / "measure.json", engine_refresh.EngineMeasureResult)
    assert done is not None
    return done


def test_a_candidate_is_measured_interleaved_judged_and_switched_only_on_a_pass(
    world: World, fleet: Fleet
) -> None:
    put(Engine.VLLM, "v0.32.0", gate=GateVerdict.PASSED, current=True)
    put(Engine.VLLM, "v0.33.0")
    approved_recipe(Engine.VLLM)  # an approved file: measured in its own job, never part of the gate
    candidate = engine_refresh.MeasureMode.CANDIDATE
    recorded = measure_job(world, fleet, Engine.VLLM, "v0.33.0", candidate)
    # warm starts (current, candidate), then rounds current, candidate, current, candidate; REFRESH runs
    assert [tag for tag, _, _ in fleet.starts] == ["v0.32.0", "v0.33.0"] * 3
    assert {kind for _, kind, _ in fleet.starts} == {"refresh"}
    assert recorded.verdict is not None
    assert (recorded.verdict.gate, recorded.switched) == (
        GateVerdict.RECORD_ONLY,
        False,
    )  # not calibrated yet
    assert read_record(Engine.VLLM, "v0.33.0").gate is GateVerdict.RECORD_ONLY  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.VLLM)) == "v0.32.0"
    results = read_record(Engine.VLLM, "v0.33.0").results  # type: ignore[union-attr]
    assert [r["kind"] for r in results] == ["warm", "candidate", "candidate"]
    assert results[1]["complete"] is True
    assert results[1]["gpu_peak_gib"] == 20.0
    assert set(recorded.steps.values()) == {"ok"}
    assert fleet.tokenizers == {MODEL}  # llama-benchy counts the model's own tokens, never the served name's
    assert fleet.removed == [image_of(Engine.VLLM, "v0.33.0")]  # a candidate that is not current goes at once

    write_model(engine_refresh.limits_path(Engine.VLLM), limits())  # calibrated
    fleet.starts.clear()
    fleet.number = -1
    put(Engine.VLLM, "v0.34.0")
    passed = measure_job(world, fleet, Engine.VLLM, "v0.34.0", candidate)
    assert (passed.verdict.gate, passed.switched) == (GateVerdict.PASSED, True)  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.VLLM)) == "v0.34.0"
    assert link_target(versions_dir(Engine.VLLM), "previous") == "v0.32.0"

    put(Engine.VLLM, "v0.35.0")
    fleet.by_tag["v0.35.0"] = {"figures": {"write_tps_16_at_once": 80.0}}
    kept = measure_job(world, fleet, Engine.VLLM, "v0.35.0", candidate)
    assert (kept.verdict.gate, kept.switched) == (GateVerdict.FAILED, False)  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.VLLM)) == "v0.34.0"
    assert "slower" in read_record(Engine.VLLM, "v0.35.0").reason  # type: ignore[union-attr]

    put(
        Engine.VLLM, "v0.36.0", publisher=Publisher.COMMUNITY
    )  # passes, but a community build is never switched
    community = measure_job(world, fleet, Engine.VLLM, "v0.36.0", candidate)
    assert (community.verdict.gate, community.switched) == (GateVerdict.PASSED, False)  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.VLLM)) == "v0.34.0"


def test_nothing_starts_after_a_failed_start_and_a_candidate_needs_a_current_on_disk(
    world: World, fleet: Fleet
) -> None:
    put(Engine.SGLANG, "v0.5.22-cu130", gate=GateVerdict.PASSED, current=True)
    put(Engine.SGLANG, "v0.5.23-cu130")
    fleet.fail_on = {3}  # the second measured start
    done = measure_job(world, fleet, Engine.SGLANG, "v0.5.23-cu130", engine_refresh.MeasureMode.CANDIDATE)
    assert len(fleet.starts) == 4  # D22: nothing starts after it
    assert done.verdict.gate is GateVerdict.FAILED  # type: ignore[union-attr]
    assert "start 3 failed" in done.verdict.reason  # type: ignore[union-attr]
    assert done.steps["r2-current"].startswith("not measured: not started after")
    assert read_record(Engine.SGLANG, "v0.5.23-cu130").gate is GateVerdict.FAILED  # type: ignore[union-attr]

    for refusal in ({"fail_on": {2}}, {"refuse_on": {3}}):  # current's start failed; the candidate's refused
        fleet.starts.clear()
        fleet.fail_on, fleet.refuse_on = refusal.get("fail_on", set()), refusal.get("refuse_on", set())
        put(Engine.SGLANG, "v0.5.24-cu130")
        unjudged = measure_job(
            world, fleet, Engine.SGLANG, "v0.5.24-cu130", engine_refresh.MeasureMode.CANDIDATE
        )
        assert unjudged.verdict.gate is GateVerdict.NOT_RUN  # type: ignore[union-attr]
        assert read_record(Engine.SGLANG, "v0.5.24-cu130").gate is GateVerdict.NOT_RUN  # type: ignore[union-attr]
        assert "v0.5.24-cu130" not in engine_refresh.judged_tags(Engine.SGLANG)  # offered again next run
    fleet.fail_on, fleet.refuse_on = set(), set()

    def broken(*_args: object) -> None:
        msg = "a bug"
        raise RuntimeError(msg)

    put(Engine.SGLANG, "v0.5.24-cu130")  # the last job's prune took its image
    job = engine_refresh.MeasureJob(
        world.run_dir,
        Engine.SGLANG,
        "v0.5.24-cu130",
        engine_refresh.MeasureMode.CANDIDATE,
        harness=fleet.harness(),
    )
    job._candidate = broken  # type: ignore[method-assign]
    assert job.execute() == 0  # recorded, and the rest of the chain still runs
    assert (
        "RuntimeError: a bug"
        in read_model(job.folder / "measure.json", engine_refresh.EngineMeasureResult).error
    )  # type: ignore[union-attr]

    put(Engine.TRTLLM, "1.3.0rc12", current=True, pulled=False)  # a current whose image is gone
    put(Engine.TRTLLM, "1.3.0rc13")
    refused = measure_job(world, fleet, Engine.TRTLLM, "1.3.0rc13", engine_refresh.MeasureMode.CANDIDATE)
    assert "no `current` with its image on disk" in refused.not_measured
    assert read_record(Engine.TRTLLM, "1.3.0rc13").gate is GateVerdict.NOT_RUN  # type: ignore[union-attr]


def test_a_bootstrap_makes_the_first_current_only_when_official_and_every_phase_measured(
    world: World, fleet: Fleet
) -> None:
    bootstrap = engine_refresh.MeasureMode.BOOTSTRAP
    put(Engine.SGLANG, "v0.5.23-cu130")
    done = measure_job(world, fleet, Engine.SGLANG, "v0.5.23-cu130", bootstrap)
    assert [kind for _, kind, _ in fleet.starts] == ["bootstrap", "bootstrap"]  # a warm start, a measured one
    assert (done.verdict.gate, done.switched) == (GateVerdict.PASSED, True)  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.SGLANG)) == "v0.5.23-cu130"

    put(Engine.TRTLLM, "1.3.0rc13")
    fleet.by_tag["1.3.0rc13"] = {"unmeasured": (engine_bench.Phase.TEACHER_FORCED,)}
    failed = measure_job(world, fleet, Engine.TRTLLM, "1.3.0rc13", bootstrap)
    assert (failed.verdict.gate, failed.switched) == (GateVerdict.FAILED, False)  # type: ignore[union-attr]
    assert failed.verdict.speed_failures == ["not measured: teacher_forced: boom"]  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.TRTLLM)) is None

    put(Engine.VLLM, "v0.33.0-gb10.0", publisher=Publisher.COMMUNITY)
    alone = measure_job(world, fleet, Engine.VLLM, "v0.33.0-gb10.0", engine_refresh.MeasureMode.CANDIDATE)
    assert alone.mode is bootstrap  # planned as a candidate; no `current` when it ran: measured alone
    assert (alone.verdict.gate, alone.switched) == (GateVerdict.PASSED, False)  # type: ignore[union-attr]
    assert link_target(versions_dir(Engine.VLLM)) is None  # measured and recorded; never made current

    fleet.starts.clear()
    put(Engine.SGLANG, "v0.5.24-cu130", publisher=Publisher.OFFICIAL)
    behind = measure_job(world, fleet, Engine.SGLANG, "v0.5.24-cu130", bootstrap)
    assert behind.mode is engine_refresh.MeasureMode.CANDIDATE  # a `current` appeared earlier in the chain
    assert len(fleet.starts) == 6  # compared with it: never a second bootstrap over it


def test_an_approved_file_is_measured_on_current_alone_without_a_gate(world: World, fleet: Fleet) -> None:
    put(Engine.VLLM, "v0.33.0", gate=GateVerdict.PASSED, current=True)
    done = measure_job(world, fleet, Engine.VLLM, "v0.33.0", engine_refresh.MeasureMode.APPROVED, model=MODEL)
    assert done.verdict is None
    folder = engine_refresh.engine_folder(world.run_dir, Engine.VLLM, "v0.33.0")
    assert not (
        folder / "measure.json"
    ).exists()  # its own folder: a candidate's verdict is never overwritten
    assert (folder / "approved" / "nvidia--Qwen3-30B-A3B-NVFP4" / "measure.json").exists()
    assert [r["kind"] for r in read_record(Engine.VLLM, "v0.33.0").results] == ["warm", "approved"]  # type: ignore[union-attr]
    put(Engine.VLLM, "v0.34.0")
    elsewhere = measure_job(
        world, fleet, Engine.VLLM, "v0.34.0", engine_refresh.MeasureMode.APPROVED, model=MODEL
    )
    assert "is not vllm's `current`" in elsewhere.not_measured


def calibrate_job(
    world: World, fleet: Fleet, engine: Engine, tag: str, adjacent: str
) -> engine_refresh.CalibrationResult:
    job = engine_refresh.CalibrateJob(world.run_dir, engine, tag, adjacent, harness=fleet.harness())
    assert job.execute() == 0
    done = read_model(
        engine_refresh.engine_folder(world.run_dir, engine, tag) / "calibration.json",
        engine_refresh.CalibrationResult,
    )
    assert done is not None
    return done


def test_calibration_sets_limits_from_the_arms_only_when_every_arm_measured(
    world: World, fleet: Fleet
) -> None:
    sglang = Engine.SGLANG
    put(sglang, "v0.5.23-cu130", gate=GateVerdict.PASSED, current=True)
    put(sglang, "v0.5.22-cu130")
    # starts: 0 warm, 1 reference, 2 restart, 3 empty cache, 4 autotune off, 5 warm adjacent, 6 adjacent
    fleet.by_start = {
        2: {"score": tf(ppl=2.002, diff=1)},
        3: {"score": tf(ppl=2.004, diff=2)},
        4: {"score": tf(ppl=2.003, diff=1)},
        6: {"score": tf(ppl=2.02, diff=24)},
    }
    done = calibrate_job(world, fleet, sglang, "v0.5.23-cu130", "v0.5.22-cu130")
    assert [tag for tag, _, _ in fleet.starts] == ["v0.5.23-cu130"] * 5 + ["v0.5.22-cu130"] * 2
    assert fleet.starts[3][2]["runtime_cache"] is False
    assert fleet.starts[4][2]["command"].endswith("--disable-flashinfer-autotune")
    assert fleet.starts[2][2]["runtime_cache"] == {"key_by_image": True}  # the restart reuses the cache
    assert done.missing == []
    set_limits = engine_refresh.read_limits(sglang)
    assert set_limits is not None
    assert set_limits.ppl_ratio_max == pytest.approx(1.002 + 0.01)  # the noise arms' largest + 0.01
    assert set_limits.same_top_min == pytest.approx(0.976 - 0.006)  # the adjacent figure minus 0.6 points
    assert set(set_limits.arms) == {"restart", "empty-cache", "autotune-off", "adjacent"}
    assert read_record(sglang, "v0.5.22-cu130").reason == "adjacent version for calibration"  # type: ignore[union-attr]

    before = engine_refresh.read_limits(sglang)
    put(sglang, "v0.5.22-cu130")  # the first calibration pruned the adjacent image
    fleet.starts.clear()
    fleet.number = -1
    fleet.fail_on = {3}  # one arm cannot start: nothing is loosened by half a calibration
    half = calibrate_job(world, fleet, sglang, "v0.5.23-cu130", "v0.5.22-cu130")
    assert half.missing
    assert half.limits is None
    assert engine_refresh.read_limits(sglang) == before


def test_calibration_is_refused_unless_current_with_both_images_and_no_candidate_under_judgment(
    world: World, fleet: Fleet
) -> None:
    put(Engine.VLLM, "v0.33.0", gate=GateVerdict.PASSED, current=True)
    put(Engine.VLLM, "v0.32.0", pulled=False)
    assert "v0.32.0: its image is not on disk" in (
        engine_refresh.calibration_refusal(Engine.VLLM, "v0.33.0", "v0.32.0") or ""
    )
    assert (
        engine_refresh.calibration_refusal(Engine.VLLM, "v0.33.0", "v0.32.0", adjacent_pending=True) is None
    )
    assert "is not vllm's `current`" in (
        engine_refresh.calibration_refusal(Engine.VLLM, "v0.32.0", "v0.31.0") or ""
    )
    put(Engine.VLLM, "v0.32.0")
    put(Engine.VLLM, "v0.34.0")  # newer than current, pulled, no verdict: under judgment
    refused = calibrate_job(world, fleet, Engine.VLLM, "v0.33.0", "v0.32.0")
    assert "v0.34.0 is under judgment" in refused.not_measured
    assert fleet.starts == []
    put(Engine.VLLM, "v0.32.0")  # the refused job's prune took its image
    put(Engine.VLLM, "v0.34.0", gate=GateVerdict.FAILED)  # judged: calibration may run
    assert engine_refresh.calibration_refusal(Engine.VLLM, "v0.33.0", "v0.32.0") is None


# --- the walk -------------------------------------------------------------------------------------


def tag_fetch(tags: dict[str, list[str]]) -> Any:
    """A registry that lists ``tags`` by repository (none for any other)."""

    def fetch(url: str, headers: Mapping[str, str], timeout: float) -> Response:
        repository = url.split("/v2/")[1].split("/tags/list", maxsplit=1)[0]
        return Response(200, {}, json.dumps({"tags": tags.get(repository, [])}).encode())

    return fetch


def walk(world: World, **kw: Any) -> refresh.RunRecord:
    """Run the engine walk (not ``refresh run``) with fakes and submit its chain; the run record."""
    world.run_dir.mkdir(parents=True, exist_ok=True)
    record = refresh.RunRecord(time="t")
    dry = bool(kw.pop("dry_run", False))
    submitter = refresh.JobSubmitter(record, dry_run=dry)
    after = kw.pop("llama_measure_job", None)
    chain = engine_refresh.EngineWalk(
        argparse.Namespace(dry_run=dry, home=str(world.home)),
        world.run_dir,
        record,
        submitter,
        Path(os.environ["LOCALMODELS_SUBMIT"]),
        **kw,
    )
    chain.decide()
    chain.submit(after)
    return record


def submitted(world: World) -> list[str]:
    return world.submitted.read_text().splitlines() if world.submitted.exists() else []


def test_the_first_walk_bootstraps_an_engine_in_one_chain_with_the_adjacent_pull_and_the_calibration(
    world: World,
) -> None:
    fetch = tag_fetch(
        {
            "vllm/vllm-openai": ["v0.32.0", "v0.33.0", "v0.33.0.rc1", "latest"],
            "timothystewart6/vllm-gb10": ["v0.33.0-gb10.0"],
        }
    )
    record = walk(world, fetch=fetch)
    pull, measure, adjacent, calibrate, community_pull, community, same_client = submitted(world)
    assert "--tag v0.33.0-gb10.0 --publisher community" in community_pull
    assert "--tag v0.33.0-gb10.0 --mode candidate" in community  # of the official bootstrapped before it
    assert "--name refresh-pull-vllm" in pull
    assert "--class resume --resume-ok --no-anywhere --cpus 2 --mem 4G" in pull
    assert "--after" not in pull
    assert "engine-pull --run" in pull
    assert "--tag v0.33.0 --publisher official" in pull
    assert "--name refresh-engine-vllm" in measure
    assert "--class restart --gpu --mem 61G --after 101" in measure  # the 57.0 GiB seed + 4, ceil
    assert "timeout --kill-after=180 2400" in measure
    assert "--tag v0.33.0 --mode bootstrap" in measure
    assert "--tag v0.32.0" in adjacent
    assert "--after 102" in adjacent  # one chain: one image pulled at a time
    assert "--name refresh-calibrate-vllm" in calibrate
    assert "--after 103 --after 102" in calibrate  # two parents: the adjacent pull and the bootstrap measure
    assert "--adjacent v0.32.0" in calibrate
    assert "timeout --kill-after=180 3600" in calibrate
    assert "--name refresh-same-client" in same_client
    assert "--after 106" in same_client  # last in the chain: no llama.cpp measure job
    assert "timeout --kill-after=180 600" in same_client
    plans = {(p.engine, p.publisher): p for p in record.engines}
    vllm = plans[("vllm", "official")]
    assert (vllm.tag, vllm.mode, vllm.reason) == ("v0.33.0", "bootstrap", "")
    assert (plans[("vllm", "community")].tag, plans[("vllm", "community")].mode) == (
        "v0.33.0-gb10.0",
        "candidate",
    )
    assert plans[("sglang", "official")].reason == "no new tag"
    assert [job.job for job in record.jobs] == ["101", "102", "103", "104", "105", "106", "107"]


def test_walk_plans_for_candidates_community_builds_tried_tags_and_unreadable_lists(  # noqa: PLR0915 - walk after walk
    world: World,
) -> None:
    put(Engine.VLLM, "v0.32.0", gate=GateVerdict.PASSED, current=True)
    put(Engine.SGLANG, "v0.5.22-cu130", gate=GateVerdict.FAILED)  # tried: its newer-than check stops there
    fetch = tag_fetch(
        {
            "vllm/vllm-openai": ["v0.32.0", "v0.33.0"],
            "timothystewart6/vllm-gb10": ["v0.33.0-gb10.0"],
            "lmsysorg/sglang": ["v0.5.22-cu130", "v0.5.22-runtime"],
        }
    )
    approved_recipe(Engine.VLLM)
    record = walk(world, fetch=fetch, llama_measure_job="900")
    _pull, measure, approved, community_pull, community_measure, same_client = submitted(world)
    assert (
        "--tag v0.33.0 --mode candidate" in measure
    )  # official, with a current: a candidate, no calibration
    assert "--after 101" in measure
    assert "--mode candidate --model" not in measure  # the gate is the measurement model's alone
    # the approved file on what is then current, in its own job (a community build never becomes current)
    assert "--tag v0.33.0 --mode approved --model org/approved" in approved
    assert "--after 102" in approved
    assert "engine-vllm-v0.33.0-approved-org--approved.log" in approved  # its own log
    assert "--tag v0.33.0-gb10.0 --publisher community" in community_pull
    assert "--tag v0.33.0-gb10.0 --mode candidate" in community_measure  # measured, recorded, never switched
    assert "--after 900" in same_client  # after llama.cpp's measure job, not the chain's last
    assert [p.reason for p in record.engines if p.engine == "sglang"] == ["no new tag"]
    assert "calibrate" not in " ".join(submitted(world))

    world.submitted.unlink()
    put(
        Engine.TRTLLM, "1.3.0rc12", publisher=Publisher.COMMUNITY, gate=GateVerdict.NOT_RUN
    )  # irrelevant record
    nothing = walk(world, fetch=tag_fetch({"timothystewart6/vllm-gb10": ["v0.33.0-gb10.0"]}))
    assert (
        "--mode candidate" in submitted(world)[1]
    )  # an engine with a current: the community build is a candidate
    assert nothing.engines[1].tag == "v0.33.0-gb10.0"

    world.submitted.unlink()
    (versions_dir(Engine.VLLM) / "current").unlink()  # no current: a community build is a bootstrap, alone
    alone = walk(world, fetch=tag_fetch({"timothystewart6/vllm-gb10": ["v0.33.0-gb10.0"]}))
    jobs = submitted(world)
    assert ["--mode bootstrap" in jobs[1], len([j for j in jobs if "calibrate" in j])] == [True, 0]
    assert alone.engines[1].mode == "bootstrap"

    world.submitted.unlink()

    def offline(*_args: object) -> Response:
        msg = "no route"
        raise OSError(msg)

    broken = walk(world, fetch=offline)
    assert broken.engines[0].reason.startswith("tag list unreadable: registry-1.docker.io/vllm/vllm-openai")
    assert len(submitted(world)) == 1  # only the same-client job: nothing to pull

    world.submitted.unlink()
    put(
        Engine.VLLM, "v0.32.0", gate=GateVerdict.PASSED, current=True, pulled=False
    )  # rolled back: image gone
    gone = walk(world, fetch=tag_fetch({"vllm/vllm-openai": ["v0.32.0", "v0.33.0"]}))
    current_pull, candidate_pull = submitted(world)[:2]
    assert (
        "--tag v0.32.0 --publisher official" in current_pull
    )  # current first: the candidate is compared with it
    assert "--tag v0.33.0" in candidate_pull
    assert gone.engines[0].tag == "v0.33.0"

    world.submitted.unlink()
    put(Engine.SGLANG, "v0.5.23-cu130")  # pulled, never judged (its measure job never got the card)
    sglang = tag_fetch({"lmsysorg/sglang": ["v0.5.22-cu130", "v0.5.23-cu130"]})
    again = walk(world, fetch=sglang)
    assert next(p.tag for p in again.engines if p.engine == "sglang") == "v0.5.23-cu130"  # offered again
    world.submitted.unlink()
    put(Engine.SGLANG, "v0.5.23-cu130", gate=GateVerdict.FAILED)  # judged: tried (D12)
    judged = walk(world, fetch=sglang)
    assert next(p.reason for p in judged.engines if p.engine == "sglang") == "no new tag"


def test_the_sparkrun_move_needs_identical_dry_runs_or_a_comparison_job_the_candidates_wait_for(
    world: World,
) -> None:
    put(Engine.VLLM, "v0.32.0", gate=GateVerdict.PASSED, current=True)
    newer = lambda: TagListing(tags=("v0.4.1", "v0.4.2", "v0.4.3", "v0.4.4rc1"))  # noqa: E731
    fetch = tag_fetch({"vllm/vllm-openai": ["v0.32.0", "v0.33.0"]})
    same = lambda *_a: "exit 0\nvllm serve ..."  # noqa: E731
    record = walk(world, fetch=fetch, sparkrun_tags=newer, sparkrun=Path, dry_runner=same)
    assert engine_refresh.in_use_version() == "v0.4.3"  # moved with no measurement
    assert not any("refresh-sparkrun" in line for line in submitted(world))
    assert any("sparkrun moved v0.4.2 -> v0.4.3" in r for r in record.engine_reports)

    world.submitted.unlink()
    state = engine_refresh.read_sparkrun_state()
    assert state is not None
    write_model(engine_refresh.sparkrun_state_path(), state.model_copy(update={"version": "v0.4.2"}))  # back
    differs = lambda binary, *_a: f"exit 0\nvllm serve {binary.name}"  # noqa: E731
    record = walk(world, fetch=fetch, sparkrun_tags=newer, sparkrun=Path, dry_runner=differs)
    compare, pull, measure, *_ = submitted(world)
    assert "--name refresh-sparkrun" in compare
    assert "--old v0.4.2 --new v0.4.3" in compare
    assert "--class restart --gpu --mem 61G" in compare
    assert "timeout --kill-after=180 2400" in compare  # 2,400 s for the one engine that has a current
    assert "--after" not in compare
    assert "--after 101" in pull  # the candidates are chained after it
    assert "--after 102" in measure
    assert engine_refresh.in_use_version() == "v0.4.2"

    world.submitted.unlink()
    failing = lambda *_a: "exit 1\nError: something"  # noqa: E731
    stuck = walk(world, fetch=fetch, sparkrun_tags=newer, sparkrun=Path, dry_runner=failing)
    assert engine_refresh.in_use_version() == "v0.4.2"  # two identical failures prove nothing
    assert any("a dry run failed: not checked" in r for r in stuck.engine_reports)
    assert not any("refresh-sparkrun" in line for line in submitted(world))

    world.submitted.unlink()
    keep = engine_refresh.SparkrunState(version="v0.4.2", tried=["v0.4.3"])
    write_model(engine_refresh.sparkrun_state_path(), keep)
    walk(world, fetch=fetch, sparkrun_tags=newer, sparkrun=Path, dry_runner=differs)
    assert not any(
        "refresh-sparkrun" in line for line in submitted(world)
    )  # a tried version waits for a newer


def test_the_sparkrun_comparison_moves_on_a_pass_and_marks_tried_only_a_failed_gate(
    world: World, fleet: Fleet
) -> None:
    put(Engine.VLLM, "v0.33.0", gate=GateVerdict.PASSED, current=True)

    def compare(**script: Any) -> engine_refresh.SparkrunState:
        fleet.starts.clear()
        fleet.number = -1
        fleet.fail_on, fleet.by_start = script.get("fail_on", set()), script.get("by_start", {})
        write_model(engine_refresh.sparkrun_state_path(), engine_refresh.SparkrunState(version="v0.4.2"))
        job = engine_refresh.SparkrunCompareJob(world.run_dir, "v0.4.2", "v0.4.3", harness=fleet.harness())
        assert job.execute() == 0
        state = engine_refresh.read_sparkrun_state()
        assert state is not None
        return state

    unlimited = compare()  # no limits yet: speed passed, quality cannot be judged
    assert (unlimited.version, unlimited.tried) == ("v0.4.2", [])  # stays, and is compared again next run
    write_model(engine_refresh.limits_path(Engine.VLLM), limits())
    old_failed = compare(fail_on={2})  # a start under the old sparkrun failed: not the new one's doing
    assert (old_failed.version, old_failed.tried) == ("v0.4.2", [])
    slower = {n: {"figures": {"write_tps_16_at_once": 80.0}} for n in (1, 3)}  # measured starts 1, 3: new
    failed = compare(by_start=slower)
    assert (failed.version, failed.tried) == ("v0.4.2", ["v0.4.3"])  # its own gate failed: tried (D12)
    moved = compare()
    assert (moved.version, moved.previous) == ("v0.4.3", "v0.4.2")


def test_the_sparkrun_dry_runs_compare_after_the_run_specific_values_are_removed() -> None:
    old = (
        "sparkrun v0.4.2\nCluster:   sparkrun_6eed2e11934cb7a3_49960de5880e\n"
        "recipe /tmp/localmodels-recipe-0123456789abcdef0123456789abcdef.json at 2026-11-08T10:11:12+00:00\n"
        "label localmodels.run=0123456789abcdef0123456789abcdef run 0.0s\n"
        "id 123e4567-e89b-12d3-a456-426614174000\n"
        "Serve command:\n  vllm serve nvidia/m --port 18102\n"
    )
    new = (
        "sparkrun v0.4.3\nCluster:   sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb\n"
        "recipe /tmp/localmodels-recipe-fedcba9876543210fedcba9876543210.json at 2026-12-01T00:00:01+00:00\n"
        "label localmodels.run=fedcba9876543210fedcba9876543210 run 1.5s\n"
        "id 00000000-0000-0000-0000-000000000000\n"
        "Serve command:\n  vllm serve nvidia/m --port 18102\n"
    )
    assert engine_refresh.normalise(old) == engine_refresh.normalise(new)
    assert engine_refresh.normalise(old) != engine_refresh.normalise(new.replace("18102", "18103"))
    # a -vvv dry run: the docker run line and the serve command count; the debug chatter does not
    docker = (
        "12:59:44 [DEBUG] sparkrun.orchestration.executors.docker: docker run command: docker run -d "
        "--name sparkrun_6eed2e11934cb7a3_49960de5880e_solo --label sparkrun.intent_id=6eed2e11934cb7a3 "
        "-v /home/u/.cache/sparkrun/runtime-cache/vllm/m-3c74a3cb:/cache/runtime img bash"
    )
    serve = "12:59:44 [DEBUG] sparkrun.runtimes.base: Serve command: vllm serve m --port 18102"
    chatter = "12:59:44 [DEBUG] probing host (0.12s)"
    one = f"{docker}\n{chatter}\n{serve}\nServe command:\n  vllm serve m --port 18102\n"
    two = one.replace("6eed2e11934cb7a3_49960de5880e", "aaaaaaaaaaaaaaaa_bbbbbbbbbbbb").replace(
        "probing host (0.12s)", "probing the host differently"
    )
    assert engine_refresh.launch_lines(one) == engine_refresh.launch_lines(two)
    assert "docker run -d" in engine_refresh.launch_lines(one)
    mounted = one.replace(":/cache/runtime", ":/cache/runtime -v /etc/x:/x")
    assert engine_refresh.launch_lines(one) != engine_refresh.launch_lines(mounted)  # a new mount differs
    assert engine_refresh.launch_lines("other output") == "other output"  # no launch lines: compared whole
    assert (
        engine_refresh.newest_sparkrun(["v0.4.9", "v0.4.10", "v0.5.0rc1", "x"]) == "v0.4.10"
    )  # numeric order


def test_a_dry_run_prints_the_engine_chain_and_submits_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    walk(world, fetch=tag_fetch({"lmsysorg/sglang": ["v0.5.22-cu130", "v0.5.23-cu130"]}), dry_run=True)
    out = capsys.readouterr().out
    assert "refresh-pull-sglang" in out
    assert "--after '<build job id>'" in out
    assert not world.submitted.exists()


# --- the pull job ----------------------------------------------------------------------------------


class Docker:
    """A fake ``docker``: info gives a root, pull and inspect succeed or fail as told."""

    def __init__(self, *, fail_pull: bool = False, digest: str = f"repo/vllm@{DIGEST}") -> None:
        self.calls: list[list[str]] = []
        self.fail_pull, self.digest = fail_pull, digest

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        cmd = list(command)
        self.calls.append(cmd)
        if cmd[1] == "info":
            return subprocess.CompletedProcess(cmd, 0, "/var/lib/docker\n", "")
        if cmd[1] == "pull":
            return subprocess.CompletedProcess(
                cmd, 1 if self.fail_pull else 0, "", "denied" if self.fail_pull else ""
            )
        return subprocess.CompletedProcess(cmd, 0, self.digest + "\n", "")


def pull_job(world: World, tmp_path: Path, docker: Docker, **kw: Any) -> engine_refresh.PullJob:
    realize = tmp_path / "fake-sparkrun"
    realize.write_text('#!/bin/sh\necho \'{"model": "' + MODEL + '", "container": "x"}\'\n', encoding="utf-8")
    realize.chmod(0o755)
    world.run_dir.mkdir(parents=True, exist_ok=True)
    return engine_refresh.PullJob(
        world.run_dir,
        Engine.VLLM,
        "v0.33.0",
        Publisher.OFFICIAL,
        docker=docker,
        pull=docker,
        sparkrun=lambda _v: realize,
        free_bytes=kw.pop("free_bytes", lambda _path: 500 * 10**9),
        tools=docker,
    )


def test_the_pull_job_records_the_digest_and_realized_recipes_and_a_failed_pull_is_a_tried_tag(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installs: list[str] = []

    def install() -> list[Path]:
        installs.append("install")
        if len(installs) > 1:
            msg = "cannot read the D16 inputs"
            raise engine_bench.PromptFileError(msg)
        return []

    monkeypatch.setattr(engine_bench, "install_prompts", install)
    docker = Docker()
    assert pull_job(world, tmp_path, docker).execute() == 0
    assert installs == ["install"]  # the prompts were missing: installed on the CPU, before any GPU start
    record = read_record(Engine.VLLM, "v0.33.0")
    assert record is not None
    assert (record.pulled, record.image, record.sparkrun_version) == (True, f"repo/vllm@{DIGEST}", "v0.4.2")
    assert json.loads(record.recipes[MODEL])["model"] == MODEL
    assert ["docker", "pull", "vllm/vllm-openai:v0.33.0"] in docker.calls
    assert ["uvx", engine_bench.BENCHY_PACKAGE, "--help"] in docker.calls  # benchy fetched on the CPU

    failing = Docker(fail_pull=True)
    put(Engine.VLLM, "v0.34.0", pulled=False, image=False)
    job = engine_refresh.PullJob(
        world.run_dir,
        Engine.VLLM,
        "v0.35.0",
        Publisher.OFFICIAL,
        docker=failing,
        pull=failing,
        free_bytes=lambda _p: 10**12,
        tools=failing,
    )
    assert job.execute() == 0  # exit 0 whatever the outcome (a refused prompt install too): the chain goes on
    assert len(installs) == 2
    tried = read_record(Engine.VLLM, "v0.35.0")
    assert tried is not None
    assert (tried.pulled, tried.image) == (False, "")
    assert "denied" in tried.reason
    assert "v0.35.0" in tried_tags(Engine.VLLM)  # D12: not tried again until a newer tag appears


def test_the_pull_job_pulls_the_digest_the_record_names_and_refuses_for_want_of_disk(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(Engine.VLLM, "v0.33.0", pulled=False)  # a rollback: record and digest only
    docker = Docker()
    pull_job(world, tmp_path, docker).execute()
    assert [
        "docker",
        "pull",
        image_of(Engine.VLLM, "v0.33.0"),
    ] in docker.calls  # what was measured, even if the tag moved

    put(Engine.VLLM, "v0.33.0", current=True)  # current, on disk
    low = pull_job(world, tmp_path, Docker(), free_bytes=lambda _path: 99 * 10**9)
    assert low.execute() == 0
    assert read_record(Engine.VLLM, "v0.33.0").pulled is True  # type: ignore[union-attr]  # still on disk

    put(Engine.VLLM, "v0.36.0", pulled=False, image=False)
    small = Docker()
    refused = pull_job(world, tmp_path, small, free_bytes=lambda _path: 99 * 10**9)
    refused.tag = "v0.36.0"
    assert refused.execute() == 0
    assert not any(call[1] == "pull" for call in small.calls)
    assert read_record(Engine.VLLM, "v0.36.0").reason.startswith("refused: /var/lib/docker has 99 GB free")  # type: ignore[union-attr]

    def full(_record: VersionRecord) -> Path:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(engine_refresh, "write_record", full)
    assert (
        pull_job(world, tmp_path, Docker()).execute() == 1
    )  # the one error exit: the record cannot be written


# --- the same-client job, the digest lines and the commands ----------------------------------------


def test_the_same_client_job_measures_llama_cpp_once_per_build(world: World, fleet: Fleet) -> None:
    calls: list[Path] = []

    def llama(binary: Path, model: Path, lease: Path, tokenizer: str) -> engine_bench.EngineRun:
        assert tokenizer == MODEL  # the container file's Hub name: llama-benchy cannot read a GGUF's (2233)
        calls.append(model)
        return erun()

    harness = engine_refresh.Harness(llama=llama)
    for _ in range(2):
        assert engine_refresh.SameClientJob(world.run_dir, str(world.home), harness=harness).execute() == 0
    assert len(calls) == 1  # the second run found the record for this build
    gguf = engine_refresh.SAME_CLIENT_FILES[MODEL]
    assert engine_refresh.same_client_record(CURRENT, gguf) is not None
    # keyed by the llama.cpp file (two projects may pair one container file with different llama.cpp files)
    assert engine_refresh.same_client_path(CURRENT, gguf).name == (
        f"Qwen--Qwen3-30B-A3B-GGUF@{gguf.revision}--Qwen3-30B-A3B-Q4_K_M.gguf.json"
    )
    assert engine_refresh.same_client_record("deadbeef0", gguf) is None


def test_the_digest_lines_say_what_each_engine_did(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    def plan(
        engine: str, tag: str, publisher: str = "official", mode: str = "candidate", reason: str = ""
    ) -> refresh.EnginePlan:
        return refresh.EnginePlan(
            engine=engine, repository="r", publisher=publisher, tag=tag, mode=mode, reason=reason
        )

    def result(engine: Engine, tag: str, mode: engine_refresh.MeasureMode, **kw: Any) -> None:
        folder = engine_refresh.engine_folder(world.run_dir, engine, tag)
        write_model(
            folder / "measure.json",
            engine_refresh.EngineMeasureResult(engine=engine.value, tag=tag, mode=mode, **kw),
        )

    def verdict(gate: GateVerdict, *failures: str) -> engine_refresh.EngineVerdict:
        return engine_refresh.EngineVerdict(gate=gate, speed_failures=list(failures))

    cand, boot = engine_refresh.MeasureMode.CANDIDATE, engine_refresh.MeasureMode.BOOTSTRAP
    world.run_dir.mkdir(parents=True)
    put(Engine.VLLM, "v0.40.0", pulled=False, image=False).model_copy(update={"reason": "denied"})
    write_record(read_record(Engine.VLLM, "v0.40.0").model_copy(update={"reason": "denied"}))  # type: ignore[union-attr]
    put(Engine.VLLM, "v0.41.0")
    result(Engine.SGLANG, "s1", cand, switched=True, verdict=verdict(GateVerdict.PASSED))
    result(
        Engine.SGLANG, "s2", cand, verdict=verdict(GateVerdict.FAILED, "a 10 % slower", "b 9 % slower", "c")
    )
    result(Engine.SGLANG, "s3", cand, verdict=verdict(GateVerdict.RECORD_ONLY))
    result(Engine.TRTLLM, "t1", boot, switched=True, verdict=verdict(GateVerdict.PASSED))
    result(
        Engine.TRTLLM, "t2", boot, verdict=verdict(GateVerdict.FAILED, "not measured: teacher_forced: boom")
    )
    plans = [
        plan("vllm", "v0.40.0"),
        plan("vllm", "v0.41.0"),
        plan("sglang", "s1"),
        plan("sglang", "s2"),
        plan("sglang", "s3"),
        plan("trtllm", "t1", mode="bootstrap"),
        plan("trtllm", "t2", mode="bootstrap"),
        plan("vllm", "", reason="no new tag"),
        plan("sglang", "", reason="tag list unreadable: boom"),
    ]
    run = refresh.RunRecord(
        time="t", engines=plans, engine_reports=["sparkrun moved v0.4.2 -> v0.4.3: x", "reaper: y"]
    )
    lines = engine_refresh.engine_lines(refresh.RunView(world.run_dir, run, None, None))
    assert lines == [
        "sparkrun moved v0.4.2 -> v0.4.3: x",
        "vllm v0.40.0 (official): pull failed: denied",
        "vllm v0.41.0 (official): measure job pending",
        "sglang s1 (official): switched",
        "sglang s2 (official): kept back: a 10 % slower; b 9 % slower",
        "sglang s3 (official): recorded for the owner (limits not calibrated)",
        "trtllm t1 (official): bootstrap passed: now current",
        "trtllm t2 (official): bootstrap failed: not measured: teacher_forced: boom",
        "vllm: no new tag",
        "sglang: tag list unreadable: boom",
    ]
    write_model(world.run_dir / refresh.RUN_FILE, run)
    assert main(["refresh", "status"]) == 0
    assert "  sglang s1 (official): switched" in capsys.readouterr().out
    assert [f.text for f in digest.check_refresh(Context(None, ""))][
        -1
    ] == "sglang: tag list unreadable: boom"  # type: ignore[arg-type]


def test_engines_pull_and_calibrate_submit_their_jobs(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["engines", "pull", "vllm", "v0.33.0.rc1"]) == 1
    assert "is not a tag of any vllm source" in capsys.readouterr().err
    assert not world.submitted.exists()
    assert main(["engines", "pull", "vllm", "v0.33.0"]) == 0
    pull = submitted(world)[0]
    assert "--name refresh-pull-vllm" in pull
    assert "--class resume --resume-ok --no-anywhere --cpus 2 --mem 4G" in pull
    assert f"--workdir {world.root / 'state' / 'engines' / 'vllm' / 'pulls'}" in pull

    world.submitted.unlink()
    put(Engine.VLLM, "v0.33.0", gate=GateVerdict.PASSED, current=True)
    monkeypatch.setattr(
        "localmodels.engines.urllib_fetch", tag_fetch({"vllm/vllm-openai": ["v0.32.0", "v0.33.0"]})
    )
    assert main(["engines", "calibrate", "vllm"]) == 0
    adjacent_pull, calibrate = submitted(world)
    assert "--tag v0.32.0" in adjacent_pull
    assert "--name refresh-calibrate-vllm" in calibrate
    assert "--after 101" in calibrate
    assert "--adjacent v0.32.0" in calibrate

    world.submitted.unlink()
    put(
        Engine.VLLM, "v0.34.0"
    )  # a candidate under judgment: the owner's request is refused at submit time too
    assert main(["engines", "calibrate", "vllm"]) == 1
    assert "v0.34.0 is under judgment" in capsys.readouterr().err
    assert not world.submitted.exists()
