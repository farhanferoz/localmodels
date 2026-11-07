"""`localmodels refresh`: the verdict, the newest tag, `run`, `build`, `measure` and the digest line.

No llama.cpp program runs here: a build folder holds wrappers around ``fixtures/refresh/fake_llama.py``, which
prints recorded output. Slurm is a script that records its arguments; the lease folder is empty.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from localmodels import digest, refresh, rules
from localmodels.cli import main
from localmodels.digest import Context
from localmodels.llama_tune import KlStats

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
    noisy = KlStats(ppl_ratio=1.0, mean_kl=0.03, same_top=0.99)
    assert refresh.judge_candidate(base, cand, SAME, noisy, {}).quality_failures  # KL 0.03 is over 0.02
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
    build, measure = world.submitted.read_text().splitlines()
    assert "--class resume --resume-ok --no-anywhere --cpus 8 --mem 8G" in build
    assert "--name refresh-build" in build
    assert f"refresh --home {world.home} build --run {run_dir} --commit b12" in build
    assert "--class restart --gpu --mem 40G --after 101" in measure
    assert f"--workdir {run_dir}" in measure
    assert "timeout --kill-after=30 1200" in measure
    record = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert [job.job for job in record.jobs] == ["101", "102"]
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
    assert (kept.skipped, len(kept.jobs)) == ("", 2)  # a skipped run writes nothing over the run in progress
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
    assert not world.submitted.exists()  # no build job, and the measure job went to urgent.py
    urgent = urgent_log.read_text()
    assert "--name refresh-measure" in urgent
    assert "--time 0:30:00 --gpu --mem 40G --wait" in urgent
    assert "--after" not in urgent
    run_dir = refresh.newest_run()
    assert run_dir is not None
    built = refresh.BuildResult.model_validate_json((run_dir / refresh.BUILD_FILE).read_text())
    assert (built.tag, built.commit, built.ok) == ("b12", CANDIDATE, True)
    record = refresh.RunRecord.model_validate_json((run_dir / refresh.RUN_FILE).read_text())
    assert [(job.name, job.job) for job in record.jobs] == [("refresh-measure", "777")]

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


def test_status_prints_the_newest_run(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["refresh", "status"]) == 0
    assert "no refresh run" in capsys.readouterr().out
    world.run_dir.mkdir(parents=True)
    (world.run_dir / refresh.RUN_FILE).write_text(
        refresh.RunRecord(time="t", skipped="nothing").model_dump_json()
    )
    assert main(["refresh", "status"]) == 0
    assert capsys.readouterr().out.startswith("refresh 2026-11-01: skipped, nothing\n")
