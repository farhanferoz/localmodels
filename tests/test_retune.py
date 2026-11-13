"""The automatic re-tune: one slurm job per model and change, what each state answers, what queues nothing.

Slurm is never reached: ``$LOCALMODELS_SUBMIT`` is a small script the test writes (it records its argument
list), and ``squeue`` and ``sacct`` on PATH are others.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from localmodels import retune, settings
from localmodels.enums import DeviceTier, DeviceType, WorkloadKind
from localmodels.hardware import HardwareReport
from localmodels.settings import ModelIdentity, TuneRecipe, lookup
from localmodels.stack import Engine, RetunePolicy, TunePlan, TuneState

GPU = HardwareReport(
    platform="linux", arch="aarch64", device=DeviceType.CUDA, device_name="NVIDIA GB10",
    total_memory_gb=122.0, tier=DeviceTier.FAST_ACCELERATED,
)  # fmt: skip
CPU = GPU.model_copy(update={"device": DeviceType.CPU, "device_name": "CPU"})
GPU_KEY = "cuda:NVIDIA GB10:122GB:aarch64"
MODEL = ModelIdentity(repository="google/medgemma-1.5-4b-it", revision="a" * 40)
KIND = WorkloadKind.K5
HOUR = retune.RETRY_AFTER_S

# Stands in for slurm's submit.py: records its arguments and interpreter, prints the job id as its last line,
# can refuse, hang or print bytes that are not text.
FAKE_SUBMIT = """\
import json, os, sys, time
base = os.environ["FAKE_DIR"]
with open(base + "/argv.jsonl", "a") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
open(base + "/prefix", "w").write(sys.prefix)
time.sleep(float(os.environ.get("FAKE_SUBMIT_SLEEP", "0")))
if os.environ.get("FAKE_SUBMIT_REFUSES"):
    print("[submit] REFUSED: the lease keeper is not running", file=sys.stderr)
    sys.exit(1)
job = str(100 + len(open(base + "/argv.jsonl").read().splitlines()))
if os.environ.get("FAKE_SUBMIT_BAD_BYTES"):
    sys.stdout.flush()
    sys.stdout.buffer.write(b"[submit] caf\\xe9\\n")
print(job)
"""

# Stand in for squeue and sacct. ``live`` and ``finished`` hold "id name" lines. squeue lists the live jobs,
# and the finished ones too unless asked for the unfinished states only (slurm keeps finished jobs for a day);
# every call is counted. sacct lists the finished jobs of a name.
FAKE_SQUEUE = """\
#!/bin/sh
echo call >> "$FAKE_DIR/squeue.calls"
[ "$FAKE_SQUEUE_BROKEN" ] && { echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }
cat "$FAKE_DIR/live" 2>/dev/null
case "$*" in *--states*) ;; *) cat "$FAKE_DIR/finished" 2>/dev/null ;; esac
exit 0
"""
FAKE_SACCT = """\
#!/bin/sh
[ "$FAKE_SACCT_BROKEN" ] && { echo "sacct: error: slurmdbd is down" >&2; exit 1; }
while [ $# -gt 0 ]; do [ "$1" = --name ] && name=$2; shift; done
grep " $name\\$" "$FAKE_DIR/finished" 2>/dev/null | while read id rest; do echo "$id COMPLETED"; done
exit 0
"""


@dataclass
class Slurm:
    folder: Path
    script: Path

    def submissions(self) -> list[list[str]]:
        log = self.folder / "argv.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def name_of(self, job: str) -> str:
        """The ``--name`` the fake gave job 101, 102, ... when it was submitted."""
        argv = self.submissions()[int(job) - 101]
        return f"proj:restart:{argv[argv.index('--name') + 1]}"

    def calls(self) -> int:
        log = self.folder / "squeue.calls"
        return len(log.read_text().splitlines()) if log.exists() else 0

    def put(self, which: str, *jobs: str) -> None:
        """Make squeue list ``jobs`` (ids the fake issued) as live (``live``) or as over (``finished``)."""
        (self.folder / which).write_text("".join(f"{job} {self.name_of(job)}\n" for job in jobs))

    def live(self, *jobs: str) -> None:
        self.put("live", *jobs)

    def finished(self, *jobs: str) -> None:
        self.put("finished", *jobs)


@pytest.fixture
def slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Slurm:
    """A GB10 host, an empty store, fake submit.py, squeue and sacct; the re-tune is on, as pytest needs."""
    folder = tmp_path / "fake"
    folder.mkdir()
    script = folder / "submit.py"
    script.write_text(FAKE_SUBMIT)
    for name, text in (("squeue", FAKE_SQUEUE), ("sacct", FAKE_SACCT)):
        (folder / name).write_text(text)
        (folder / name).chmod(0o755 | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DIR", str(folder))
    monkeypatch.setenv(retune.SUBMIT_ENV, str(script))
    monkeypatch.setenv(settings.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(settings.RETUNE_ENV, settings.RETUNE_ON)  # pytest turns it off unless it is on
    for name in (settings.TUNING_ENV, settings.TUNE_CHANGE_ENV):
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


def state_of(model: ModelIdentity = MODEL) -> str:
    return json.loads(pending_file(model).read_text())["state"]


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
            "--no-check", retune.NO_CHECK,
            "--name", f"tune-google--medgemma-1.5-4b-it-{change}",
            "--workdir", str(tmp_path),
            "--log", str(settings.settings_dir() / "tune-logs" / f"{change}.log"),
            "--",
            "timeout", "--kill-after=30", "2400",
            "env", "LOCALMODELS_TUNING=1", f"LOCALMODELS_TUNE_CHANGE={change}",
            "python", "tune.py", "--fast",
        ]
    ]  # fmt: skip
    if Path("/usr/bin/python3").exists():  # the machine's interpreter, not a project's venv
        assert (slurm.folder / "prefix").read_text() == "/usr"
    assert not list(retune.pending_dir().glob("*.tmp"))
    slurm.live("101")
    again = lookup(MODEL, KIND, recipe=recipe(tmp_path))
    assert (again.tune, again.tune_detail) == (TuneState.ALREADY_QUEUED, "101")
    assert len(slurm.submissions()) == 1


def test_a_finished_or_forgotten_job_without_a_record_fails_once_and_is_never_queued_again(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ask(tmp_path).state is TuneState.QUEUED
    monkeypatch.setenv("FAKE_SQUEUE_BROKEN", "1")  # slurm cannot be asked: not the same as "finished"
    assert ask(tmp_path).state is TuneState.NO_SCHEDULER
    monkeypatch.delenv("FAKE_SQUEUE_BROKEN")
    slurm.finished(
        "101"
    )  # squeue still knows it (COMPLETED) but lists no unfinished job: finished or forgotten
    ended = ask(tmp_path)
    log = settings.settings_dir() / "tune-logs" / f"{pending_file().stem}.log"
    assert (ended.state, ended.detail) == (TuneState.FAILED_BEFORE, str(log))
    assert f"delete {pending_file()}" in ended.note
    assert ask(tmp_path).state is TuneState.FAILED_BEFORE
    assert len(slurm.submissions()) == 1
    assert state_of() == "job_ended_without_record"
    pending_file().unlink()  # a person re-arms it
    assert ask(tmp_path).state is TuneState.QUEUED


def test_a_failed_submission_is_tried_again_after_an_hour_and_a_job_whose_id_was_lost_is_found_by_name(
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
    assert ask(tmp_path, now=1000.0 + HOUR).state is TuneState.QUEUED  # job 102
    # a process that died while submitting leaves a ``submitting`` file; after the timeout and margin: failed
    pending_file().write_text(json.dumps({"state": "submitting", "at": 5000.0, "pid": 7}))
    assert ask(tmp_path, now=5000.0 + retune.SUBMITTING_STALE_S).state is TuneState.ALREADY_QUEUED
    monkeypatch.setenv(
        "FAKE_SQUEUE_BROKEN", "1"
    )  # an earlier attempt may have left a job: not asking is not "none"
    assert ask(tmp_path, now=5000.0 + HOUR).state is TuneState.NO_SCHEDULER
    monkeypatch.delenv("FAKE_SQUEUE_BROKEN")
    monkeypatch.setenv("FAKE_SACCT_BROKEN", "1")
    assert ask(tmp_path, now=5000.0 + HOUR).state is TuneState.NO_SCHEDULER
    assert len(slurm.submissions()) == 2
    # the attempt did queue a job (job 102 is live, named for the change): it is adopted, not submitted again
    slurm.live("102")
    found = ask(tmp_path, now=5000.0 + HOUR)
    assert (found.state, found.detail) == (TuneState.ALREADY_QUEUED, "102")
    assert (state_of(), len(slurm.submissions())) == ("queued", 2)
    # or it has finished since: accounting knows it, so the change failed rather than being retried
    pending_file().write_text(json.dumps({"state": "submit_failed", "at": 5000.0, "reason": "timeout"}))
    slurm.live()
    slurm.finished("102")
    monkeypatch.delenv("FAKE_SACCT_BROKEN")
    over = ask(tmp_path, now=5000.0 + HOUR)
    assert (over.state, state_of(), len(slurm.submissions())) == (
        TuneState.FAILED_BEFORE,
        "job_ended_without_record",
        2,
    )
    # and with no such job anywhere, the retry submits
    pending_file().write_text(json.dumps({"state": "submit_failed", "at": 5000.0, "reason": "timeout"}))
    slurm.finished()
    assert ask(tmp_path, now=5000.0 + HOUR).state is TuneState.QUEUED


def test_at_most_two_tune_jobs_hold_a_slot_and_a_free_slot_goes_to_the_next_lookup(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = [ModelIdentity(repository=f"org/model-{n}", revision=str(n) * 40) for n in range(4)]
    assert ask(tmp_path, models[0]).state is TuneState.QUEUED
    slurm.live("101")
    assert ask(tmp_path, models[1]).state is TuneState.QUEUED
    slurm.live("101", "102")
    before = slurm.calls()
    assert ask(tmp_path, models[2]).state is TuneState.WAITING
    assert slurm.calls() - before == 1  # one squeue for all the queued files
    assert not pending_file(models[2]).exists()
    slurm.live("102")  # the first job has finished: its file moves on during the count
    assert ask(tmp_path, models[2]).state is TuneState.QUEUED
    assert state_of(models[0]) == "job_ended_without_record"
    # a fresh ``submitting`` file holds a slot, and so does a ``queued`` one slurm cannot be asked about
    slurm.live("102", "103")
    assert ask(tmp_path, models[3]).state is TuneState.WAITING
    slurm.live()
    pending_file(models[1]).write_text(json.dumps({"state": "submitting", "at": 9e9, "pid": 7}))
    monkeypatch.setenv("FAKE_SQUEUE_BROKEN", "1")
    assert ask(tmp_path, models[3], now=9e9).state is TuneState.WAITING  # submitting + queued (unknown)
    pending_file(models[1]).write_text(json.dumps({"state": "queued", "at": 1.0, "job": "102"}))
    assert ask(tmp_path, models[3]).state is TuneState.WAITING  # both queued, slurm cannot say


def test_the_claim_and_the_cap_count_happen_under_the_lock(slurm: Slurm, tmp_path: Path) -> None:
    result: list[retune.Outcome] = []
    with settings._locked(retune.pending_dir() / "tune"):
        thread = threading.Thread(target=lambda: result.append(ask(tmp_path)))
        thread.start()
        thread.join(0.3)
        assert thread.is_alive()  # waiting for the lock: nothing claimed, nothing submitted
        assert not pending_file().exists()
        assert slurm.submissions() == []
    thread.join(5)
    assert [r.state for r in result] == [TuneState.QUEUED]


@pytest.mark.parametrize(
    ("hardware", "call", "env", "expected", "reason"),
    [
        pytest.param(GPU, {"retune": RetunePolicy.OFF}, {}, TuneState.DISABLED, "is off", id="policy off"),
        pytest.param(
            GPU, {"retune": "off"}, {}, TuneState.DISABLED, "is off", id="policy off as a plain string"
        ),
        pytest.param(
            GPU, {}, {settings.RETUNE_ENV: "off"}, TuneState.DISABLED, "is off", id="environment off"
        ),
        pytest.param(
            GPU, {}, {settings.RETUNE_ENV: "false"}, TuneState.DISABLED, "not 'on'", id="environment false"
        ),
        pytest.param(
            GPU, {}, {settings.RETUNE_ENV: ""}, TuneState.DISABLED, "not 'on'", id="environment empty"
        ),
        pytest.param(
            GPU, {}, {settings.TUNING_ENV: "1"}, TuneState.NOT_NEEDED, None, id="a tune job's own load"
        ),
        pytest.param(GPU, {"recipe": None}, {}, TuneState.NO_RECIPE, None, id="no recipe"),
        pytest.param(
            GPU,
            {"explicit": {"dtype": "float32", "options": {}}},
            {},
            TuneState.NOTHING_TO_TUNE,
            "explicit",
            id="all explicit",
        ),
        pytest.param(CPU, {}, {}, TuneState.NOT_APPLICABLE, "GPU", id="a CPU lookup"),
        pytest.param(
            GPU, {"engine": Engine.LLAMA_CPP}, {}, TuneState.NOT_AVAILABLE, None, id="llama.cpp not enabled"
        ),
        pytest.param(
            GPU,
            {},
            {retune.SUBMIT_ENV: "/nonexistent/submit.py"},
            TuneState.NO_SCHEDULER,
            None,
            id="no slurm",
        ),
        pytest.param(
            GPU, {}, {retune.SUBMIT_ENV: ""}, TuneState.NO_SCHEDULER, "is empty", id="empty slurm variable"
        ),
        pytest.param(
            GPU, {}, {settings.RETUNE_ENV: None}, TuneState.DISABLED, "under pytest", id="pytest without on"
        ),
        pytest.param(
            GPU,
            {},
            {settings.RETUNE_ENV: None, settings.PYTEST_ENV: None},
            TuneState.DISABLED,
            "under pytest",
            id="while pytest collects",
        ),
    ],
)
def test_what_queues_nothing_creates_nothing(
    slurm: Slurm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hardware: HardwareReport,
    call: dict[str, Any],
    env: dict[str, str | None],
    expected: TuneState,
    reason: str | None,
) -> None:
    probes: list[None] = []
    monkeypatch.setattr(settings, "inspect_hardware", lambda: probes.append(None) or hardware)
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    answer = lookup(MODEL, KIND, **{"recipe": recipe(tmp_path), **call})
    assert answer.tune is expected
    assert answer.tune_plan is TunePlan.FULL
    assert len(probes) == 1  # the hardware is probed once per lookup, tune or not
    assert reason is None or any(reason in note for note in answer.notes)
    assert slurm.submissions() == []
    assert not retune.pending_dir().exists()


def test_a_tune_that_measured_but_was_refused_or_adopted_is_not_queued_again(
    slurm: Slurm, tmp_path: Path
) -> None:
    assert ask(tmp_path).state is TuneState.QUEUED
    change = pending_file().stem
    other = ModelIdentity(repository="org/other", revision="b" * 40)
    retune.job_finished(change, key=other.key, kind=KIND, refusal=None)  # another model's adopt, same job
    retune.job_finished(change, key=MODEL.key, kind=WorkloadKind.K1, refusal="x")  # another kind's
    assert state_of() == "queued"
    retune.job_finished(
        change, key=MODEL.key, kind=KIND, refusal="the chosen candidate's agreement did not pass"
    )
    again = lookup(MODEL, KIND, recipe=recipe(tmp_path))
    assert (again.tune, again.tune_detail) == (
        TuneState.MEASURED_NOT_ADOPTED,
        "the chosen candidate's agreement did not pass",
    )
    assert str(pending_file()) in again.notes[-1]
    retune.job_finished(change, key=MODEL.key, kind=KIND, refusal=None, record="float32 measured today")
    # adopted, yet the lookup still wants a tune for the same change: a failure, not another job
    first = ask(tmp_path)
    assert (first.state, state_of(), len(slurm.submissions())) == (TuneState.FAILED_BEFORE, "adopted", 1)
    assert "adopted (float32 measured today)" in first.note
    assert "no record of this model" in first.note
    assert str(pending_file()) in first.note
    stale = ["torch: 2.9.0 -> 2.10.0"]
    confirm = {"plan": TunePlan.CONFIRM, "stale": stale}
    assert ask(tmp_path, **confirm).state is TuneState.QUEUED
    change = retune.change_id(MODEL, KIND, hardware=GPU_KEY, build=None, **confirm)
    retune.job_finished(change, key=MODEL.key, kind=KIND, refusal=None, record="bfloat16 measured today")
    still = ask(tmp_path, **confirm)
    assert still.state is TuneState.FAILED_BEFORE
    assert "torch: 2.9.0 -> 2.10.0" in still.note
    assert len(slurm.submissions()) == 2
    outside = retune.pending_dir().parent / "escape.json"
    outside.write_text("{}")
    retune.job_finished(
        "../escape", key=MODEL.key, kind=KIND, refusal=None
    )  # an id from the environment is no path
    assert outside.exists()


def test_nothing_that_goes_wrong_while_queuing_raises_out_of_a_lookup(
    slurm: Slurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SUBMIT_BAD_BYTES", "1")  # output that is not text, after the job was queued
    assert ask(tmp_path).state is TuneState.QUEUED
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()  # a service whose working folder was removed
    nowhere = TuneRecipe(project="proj", command=("python", "tune.py"))
    answer = ask(tmp_path, ModelIdentity(repository="org/m", revision="c" * 40), recipe=nowhere)
    assert answer.state is TuneState.FAILED_BEFORE
    assert "FileNotFoundError" in (answer.detail or "")
    pending_file(ModelIdentity(repository="org/m", revision="c" * 40)).write_text("")  # a file never written
    empty = ask(tmp_path, ModelIdentity(repository="org/m", revision="c" * 40))
    assert empty.state is TuneState.FAILED_BEFORE
    assert "cannot be read" in empty.note
    assert str(pending_file(ModelIdentity(repository="org/m", revision="c" * 40))) in empty.note

    def broken(*_args: object, **_kwargs: object) -> retune.Outcome:
        raise RuntimeError("boom")

    monkeypatch.setattr(retune, "request", broken)
    boom = lookup(MODEL, KIND, recipe=recipe(tmp_path))
    assert (boom.tune, boom.tune_detail) == (TuneState.FAILED_BEFORE, "RuntimeError: boom")
