"""The automatic re-tune: one slurm job per model and change, what each state answers, what queues nothing.

Slurm is never reached: ``$LOCALMODELS_SUBMIT`` is a small script the test writes (it records its argument
list), and ``squeue`` on PATH is another.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from localmodels import retune, settings
from localmodels.enums import DeviceTier, DeviceType, WorkloadKind
from localmodels.hardware import HardwareReport
from localmodels.settings import ModelIdentity, TuneRecipe, lookup
from localmodels.stack import Engine, RetunePolicy, TunePlan, TuneState

if TYPE_CHECKING:
    from pathlib import Path

GPU = HardwareReport(
    platform="linux", arch="aarch64", device=DeviceType.CUDA, device_name="NVIDIA GB10",
    total_memory_gb=122.0, tier=DeviceTier.FAST_ACCELERATED,
)  # fmt: skip
CPU = GPU.model_copy(update={"device": DeviceType.CPU, "device_name": "CPU"})
GPU_KEY = "cuda:NVIDIA GB10:122GB:aarch64"
MODEL = ModelIdentity(repository="google/medgemma-1.5-4b-it", revision="a" * 40)
KIND = WorkloadKind.K5
HOUR = retune.RETRY_AFTER_S

# Stands in for slurm's submit.py: records its arguments, prints the job id as its last line, can refuse.
FAKE_SUBMIT = """\
import json, os, sys, time
base = os.environ["FAKE_DIR"]
with open(base + "/argv.jsonl", "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
time.sleep(float(os.environ.get("FAKE_SUBMIT_SLEEP", "0")))
if os.environ.get("FAKE_SUBMIT_REFUSES"):
    print("[submit] REFUSED: the lease keeper is not running", file=sys.stderr)
    sys.exit(1)
job = str(100 + len(open(base + "/argv.jsonl").read().splitlines()))
print("[submit] job " + job + " (never pre-empted); log elsewhere")
print(job)
"""

# Stands in for squeue: a live job prints RUNNING; a finished one prints nothing when asked for unfinished
# states only, and COMPLETED otherwise (slurm keeps finished jobs listable for a day).
FAKE_SQUEUE = """\
#!/bin/sh
[ "$FAKE_SQUEUE_BROKEN" ] && { echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }
if grep -qx "$3" "$FAKE_DIR/live" 2>/dev/null; then echo RUNNING; else
case "$*" in *--states*) ;; *) echo COMPLETED ;; esac; fi
"""


@dataclass
class Slurm:
    folder: Path
    script: Path

    def submissions(self) -> list[list[str]]:
        log = self.folder / "argv.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def live(self, *jobs: str) -> None:
        (self.folder / "live").write_text("\n".join(jobs) + "\n")


@pytest.fixture
def slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Slurm:
    """A GB10 host, an empty store, a fake submit.py and a fake squeue."""
    folder = tmp_path / "fake"
    folder.mkdir()
    script = folder / "submit.py"
    script.write_text(FAKE_SUBMIT)
    squeue = folder / "squeue"
    squeue.write_text(FAKE_SQUEUE)
    squeue.chmod(squeue.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DIR", str(folder))
    monkeypatch.setenv(retune.SUBMIT_ENV, str(script))
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    for name in (settings.RETUNE_ENV, settings.TUNING_ENV, settings.TUNE_CHANGE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "inspect_hardware", lambda: GPU)
    monkeypatch.setattr(settings, "_announced", set())
    return Slurm(folder, script)


def recipe(tmp_path: Path) -> TuneRecipe:
    return TuneRecipe(project="proj", command=("python", "tune.py", "--fast"), workdir=tmp_path)


def ask(tmp_path: Path, model: ModelIdentity = MODEL, **more: Any) -> retune.Outcome:
    """``retune.request`` for a first use of ``model``; ``more`` replaces any keyword."""
    keywords = {
        "hardware": GPU_KEY,
        "build": None,
        "plan": TunePlan.FULL,
        "stale": [],
        "recipe": recipe(tmp_path),
    }
    return retune.request(model, KIND, **{**keywords, **more})


def pending_file(model: ModelIdentity = MODEL) -> Path:
    change = retune.change_id(model, KIND, hardware=GPU_KEY, build=None, plan=TunePlan.FULL, stale=[])
    return retune.pending_dir() / f"{change}.json"


def test_two_lookups_at_once_queue_one_job_with_the_exact_call(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SUBMIT_SLEEP", "0.4")  # the second lookup arrives while the first is submitting
    answers = []
    barrier = threading.Barrier(2)

    def one() -> None:
        barrier.wait()
        answers.append(lookup(MODEL, KIND, recipe=recipe(tmp_path)))

    threads = [threading.Thread(target=one) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(a.tune for a in answers) == [TuneState.ALREADY_QUEUED, TuneState.QUEUED]
    queued = next(a for a in answers if a.tune is TuneState.QUEUED)
    assert queued.tune_detail == "101"
    change = pending_file().stem
    assert slurm.submissions() == [
        [
            "--project", "proj",
            "--what", "localmodels tune: google/medgemma-1.5-4b-it K5 (first use)",
            "--class", "restart",
            "--gpu",
            "--name", f"tune-google--medgemma-1.5-4b-it-{change}",
            "--workdir", str(tmp_path),
            "--log", str(settings.settings_dir() / "tune-logs" / f"{change}.log"),
            "--",
            "timeout", "--kill-after=30", "900",
            "env", "LOCALMODELS_TUNING=1", f"LOCALMODELS_TUNE_CHANGE={change}",
            "python", "tune.py", "--fast",
        ]
    ]  # fmt: skip
    slurm.live("101")
    again = lookup(MODEL, KIND, recipe=recipe(tmp_path))
    assert (again.tune, again.tune_detail) == (TuneState.ALREADY_QUEUED, "101")
    assert len(slurm.submissions()) == 1


def test_a_finished_job_without_a_record_fails_once_and_is_never_queued_again(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ask(tmp_path).state is TuneState.QUEUED
    monkeypatch.setenv("FAKE_SQUEUE_BROKEN", "1")  # slurm cannot be asked: not the same as "finished"
    assert ask(tmp_path).state is TuneState.NO_SCHEDULER
    monkeypatch.delenv("FAKE_SQUEUE_BROKEN")
    ended = ask(tmp_path)  # squeue still knows job 101 (COMPLETED) but lists it as no longer unfinished
    log = settings.settings_dir() / "tune-logs" / f"{pending_file().stem}.log"
    assert (ended.state, ended.detail) == (TuneState.FAILED_BEFORE, str(log))
    assert ask(tmp_path).state is TuneState.FAILED_BEFORE
    assert len(slurm.submissions()) == 1
    assert json.loads(pending_file().read_text())["state"] == "job_ended_without_record"
    pending_file().unlink()  # a person re-arms it
    assert ask(tmp_path).state is TuneState.QUEUED
    # a record adopted after the submission clears the file instead of failing
    slurm.live()
    assert ask(tmp_path, measured_at=2e12).state is TuneState.QUEUED
    assert len(slurm.submissions()) == 3


def test_a_failed_submission_is_tried_again_only_after_an_hour_and_a_dead_submitter_counts_as_failed(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SUBMIT_REFUSES", "1")
    failed = ask(tmp_path, now=1000.0)
    assert (failed.state, failed.detail) == (
        TuneState.FAILED_BEFORE,
        "[submit] REFUSED: the lease keeper is not running",
    )
    monkeypatch.delenv("FAKE_SUBMIT_REFUSES")
    assert ask(tmp_path, now=1000.0 + HOUR - 1).state is TuneState.FAILED_BEFORE
    assert len(slurm.submissions()) == 1
    assert ask(tmp_path, now=1000.0 + HOUR).state is TuneState.QUEUED
    # a process that died while submitting leaves a ``submitting`` file; after the timeout and margin: failed
    pending_file().write_text(json.dumps({"state": "submitting", "at": 5000.0, "pid": 7}))
    assert ask(tmp_path, now=5000.0 + retune.SUBMITTING_STALE_S).state is TuneState.ALREADY_QUEUED
    assert ask(tmp_path, now=5000.0 + retune.SUBMITTING_STALE_S + 1).state is TuneState.FAILED_BEFORE
    assert ask(tmp_path, now=5000.0 + HOUR).state is TuneState.QUEUED


def test_at_most_two_tune_jobs_are_live_and_a_free_slot_goes_to_the_next_lookup(
    slurm: Slurm, tmp_path: Path
) -> None:
    models = [ModelIdentity(repository=f"org/model-{n}", revision=str(n) * 40) for n in range(3)]
    assert [ask(tmp_path, model).state for model in models[:2]] == [TuneState.QUEUED, TuneState.QUEUED]
    slurm.live("101", "102")
    assert ask(tmp_path, models[2]).state is TuneState.WAITING
    assert not pending_file(models[2]).exists()
    slurm.live("102")  # the first job has finished
    assert ask(tmp_path, models[2]).state is TuneState.QUEUED


@pytest.mark.parametrize(
    ("hardware", "call", "env", "expected"),
    [
        pytest.param(GPU, {"retune": RetunePolicy.OFF}, {}, TuneState.DISABLED, id="policy off"),
        pytest.param(GPU, {}, {settings.RETUNE_ENV: "off"}, TuneState.DISABLED, id="environment off"),
        pytest.param(GPU, {}, {settings.TUNING_ENV: "1"}, TuneState.NOT_NEEDED, id="a tune job's own load"),
        pytest.param(GPU, {"recipe": None}, {}, TuneState.NO_RECIPE, id="no recipe"),
        pytest.param(
            GPU,
            {"explicit": {"dtype": "float32", "options": {}}},
            {},
            TuneState.NOTHING_TO_TUNE,
            id="all explicit",
        ),
        pytest.param(CPU, {}, {}, TuneState.NOT_APPLICABLE, id="a CPU lookup"),
        pytest.param(
            GPU, {"engine": Engine.LLAMA_CPP}, {}, TuneState.NOT_AVAILABLE, id="llama.cpp not enabled"
        ),
        pytest.param(
            GPU, {}, {retune.SUBMIT_ENV: "/nonexistent/submit.py"}, TuneState.NO_SCHEDULER, id="no slurm"
        ),
    ],
)
def test_what_queues_nothing_creates_nothing(
    slurm: Slurm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hardware: HardwareReport,
    call: dict[str, Any],
    env: dict[str, str],
    expected: TuneState,
) -> None:
    monkeypatch.setattr(settings, "inspect_hardware", lambda: hardware)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    answer = lookup(MODEL, KIND, **{"recipe": recipe(tmp_path), **call})
    assert answer.tune is expected
    assert answer.tune_plan is TunePlan.FULL
    assert slurm.submissions() == []
    assert not retune.pending_dir().exists()


def test_a_tune_that_measured_but_was_refused_is_not_queued_again(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ask(tmp_path).state is TuneState.QUEUED
    change = pending_file().stem
    retune.job_finished(change, "the chosen candidate's agreement did not pass")
    again = lookup(MODEL, KIND, recipe=recipe(tmp_path))
    assert (again.tune, again.tune_detail) == (
        TuneState.MEASURED_NOT_ADOPTED,
        "the chosen candidate's agreement did not pass",
    )
    assert len(slurm.submissions()) == 1
    retune.job_finished(change, None)  # an adopted record removes the file
    assert not pending_file().exists()
    retune.job_finished("../escape", None)  # an id from the environment is never a path
