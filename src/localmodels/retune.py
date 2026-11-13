"""The automatic re-tune: queue one tune job per model and change through slurm, and remember it (design 5).

``settings.lookup`` calls ``request`` when a tune is due and nothing rules it out; this module is imported
only then. One tune job is queued per *change* (a model, a kind, the hardware, the build and what went
stale, as an 8-character id). The id's file under the store, ``pending/<change id>.json``, is created with
exclusive create while holding a lock, so two processes or projects asking at once cannot both submit. Its
states:

* ``submitting``: the creator is running ``submit.py``; older than the submit timeout plus a margin counts as
  ``submit_failed``.
* ``queued``: the job id is in the file; live only while slurm lists the job as unfinished.
* ``submit_failed``: no job was queued, or its id was lost; a later lookup tries again after an hour, after
  asking slurm whether a job named for the change exists (live or finished), which it then adopts.
* ``job_ended_without_record``: the job ended and no record was adopted; no automatic retry.
* ``measured_not_adopted``: the tune measured and ``adopt`` refused its result; written by ``adopt`` itself.
* ``adopted``: the tune's record was adopted and the lookup still wants a tune for the same change (a record
  that does not answer it); no automatic retry, or the same change would loop. Written by ``adopt`` itself.

The last three are permanent: a person deleting the file (or a new change id) re-arms the change. Nothing
here raises out of ``request``: a failure is a ``TuneState`` and a note. Torch-free. The scheduler is reached
by running slurm's ``submit.py`` as a subprocess (``$LOCALMODELS_SUBMIT``), never by importing it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel

from localmodels.atomic import write_text_atomically
from localmodels.settings import (
    NO_STACK_STAMP,
    TUNE_CHANGE_ENV,
    TUNING_ENV,
    TUNING_ON,
    _locked,
    _slug,
    settings_dir,
)
from localmodels.stack import TunePlan, TuneState

if TYPE_CHECKING:
    from localmodels.enums import WorkloadKind
    from localmodels.settings import ModelIdentity, TuneRecipe

SUBMIT_ENV: Final = "LOCALMODELS_SUBMIT"
DEFAULT_SUBMIT: Final = Path("~/dev/turing/preflight/slurm/submit.py")
SYSTEM_PYTHON: Final = Path(
    "/usr/bin/python3"
)  # the machine's interpreter, never a project's venv (design 5.2)
MAX_LIVE_TUNES: Final = 2  # simultaneous localmodels tune jobs (design 12, question 2)
SUBMIT_TIMEOUT_S: Final = 30
SUBMITTING_STALE_S: Final = SUBMIT_TIMEOUT_S + 30  # a ``submitting`` file this old was left by a dead process
RETRY_AFTER_S: Final = 3600  # a failed submission is tried again by a later lookup after this
SLURM_TIMEOUT_S: Final = 10  # for each squeue or sacct call
SACCT_MARGIN_S: Final = 60  # sacct is asked from this long before the attempt it looks for
TUNE_LIMIT_S: Final = 2400  # the hard ceiling, by ``timeout``: 40 minutes (D19, was D2's 15)
KILL_AFTER_S: Final = 30
# slurm's ``--no-check`` reason: a tune is a GPU measurement, and a project's tune command has no CPU dry run
NO_CHECK: Final = "the tune is itself a GPU measurement; the project's tune command has no CPU dry run"
_CHANGE_ID_CHARS: Final = 8
_CHANGE_ID = re.compile(r"[0-9a-f]{8}")
_JOB_NAME_FIELDS: Final = 2  # squeue's "%i %j": the job id and the job name

log = logging.getLogger(__name__)


class JobState(StrEnum):
    """The squeue states of a job that has not finished (slurm's ``gpu_jobs.State``)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIGURING = "CONFIGURING"
    COMPLETING = "COMPLETING"
    SUSPENDED = "SUSPENDED"


#: Asked for explicitly: slurm keeps finished jobs for a day, and ``scontrol show job`` still knows them.
_UNFINISHED: Final = ",".join(state.lower() for state in JobState)


class PendingState(StrEnum):
    """Where a change's tune stands (the module docstring)."""

    SUBMITTING = "submitting"
    QUEUED = "queued"
    SUBMIT_FAILED = "submit_failed"
    JOB_ENDED_WITHOUT_RECORD = "job_ended_without_record"
    MEASURED_NOT_ADOPTED = "measured_not_adopted"
    ADOPTED = "adopted"


class Pending(BaseModel):
    """The content of ``pending/<change id>.json``; ``at`` is when it entered ``state`` (epoch seconds).

    ``key`` and ``kind`` name the model the job was queued for: a record of another model adopted by the same
    job says nothing about this change.
    """

    state: PendingState
    at: float
    pid: int | None = None  # of the process running submit.py
    job: str | None = None
    log: str | None = None
    reason: str | None = None  # why a submission failed, or why an adopt was refused
    record: str | None = None  # the adopted record
    trigger: str = ""
    key: str | None = None
    kind: str | None = None


class Outcome(NamedTuple):
    """What ``request`` did: the state, the job id, log or reason behind it, and a sentence for the notes."""

    state: TuneState
    detail: str | None = None
    note: str = ""


class _Existing(NamedTuple):
    """What a change's file says: an answer, or (none) that a job may be submitted, maybe after a retry."""

    outcome: Outcome | None
    since: float | None = None  # when an earlier attempt that may have left a job began


class _Change(NamedTuple):
    """A change as the job is named and described."""

    id: str
    name: str  # what ``--name`` is given
    job: str  # the name slurm gives the job: ``<project>:restart:<name>``
    why: str  # the trigger, in words


class _Queue:
    """What slurm lists as unfinished, asked at most once per request: one ``squeue`` however many files."""

    def __init__(self) -> None:
        self._asked = False
        self._jobs: dict[str, str] | None = None

    def jobs(self) -> dict[str, str] | None:
        """Job id -> job name of every unfinished job; None when slurm could not be asked."""
        if not self._asked:
            self._asked = True
            self._jobs = _squeue()
        return self._jobs


def pending_dir() -> Path:
    """Where the change files are kept: ``pending/`` under the store."""
    return settings_dir() / "pending"


def change_id(
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    hardware: str,
    build: str | None,
    plan: TunePlan,
    stale: list[str],
) -> str:
    """The first 8 hex characters of a hash of what makes a tune a new one (design 5.3).

    ``stale`` is the answer's changed-component lines (each names the old and the new value), empty for a
    first use, which the plan tells apart. No project and no localmodels version is in it.
    """
    text = "\0".join([model.key, kind.value, hardware, build or "", plan.value, *stale])
    return hashlib.sha256(text.encode()).hexdigest()[:_CHANGE_ID_CHARS]


def trigger(plan: TunePlan, stale: list[str]) -> str:
    """Why a tune is due, in the words of the job's description."""
    if plan is TunePlan.FULL:
        return "first use"
    if stale == [NO_STACK_STAMP]:
        return "no stack stamp"
    return "stale: " + ", ".join(line.split(":")[0] for line in stale)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _read(path: Path) -> tuple[Pending | None, str]:
    """The pending file: (it, ""); (None, "") if absent; (None, why) if it cannot be read. Never raises."""
    try:
        return Pending.model_validate_json(path.read_text(encoding="utf-8")), ""
    except FileNotFoundError:
        return None, ""
    except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        return None, f"{path}: {exc!r}"


def _write(path: Path, pending: Pending) -> None:
    write_text_atomically(path, pending.model_dump_json(indent=2, exclude_none=True) + "\n")


def _run(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """A read-only slurm query; None when it could not be run."""
    try:
        return subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, errors="replace", timeout=SLURM_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _squeue() -> dict[str, str] | None:
    """Every unfinished job as id -> name, from one ``squeue``; None when it could not say.

    Asked for the unfinished states only: a finished job prints nothing, whether slurm still remembers it or
    has forgotten it, so absence from this list is what "finished" means.
    """
    done = _run(["squeue", "-h", "-o", "%i %j", "--states", _UNFINISHED])
    if done is None or done.returncode:
        return None
    rows = (line.split(None, 1) for line in done.stdout.splitlines() if line.strip())
    return {row[0]: row[1].strip() for row in rows if len(row) == _JOB_NAME_FIELDS}


def _finished_job(name: str, since: float) -> str | None:
    """The id of a job named ``name`` in slurm's accounting since ``since``, "" if none; None: cannot say."""
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since - SACCT_MARGIN_S))
    done = _run(["sacct", "-n", "-X", "--name", name, "-S", start, "-o", "JobID,State"])
    if done is None or done.returncode:
        return None
    ids = [line.split()[0] for line in done.stdout.splitlines() if line.strip()]
    return ids[0] if ids else ""


def _live_count(folder: Path, now: float, queue: _Queue) -> int:
    """Tune jobs holding a slot: a fresh ``submitting`` file, or a ``queued`` one not known to be over.

    A ``queued`` file whose job slurm no longer lists becomes ``job_ended_without_record`` here, as it would
    for its own lookup. Stops counting at the cap.
    """
    count = 0
    for path in sorted(folder.glob("*.json")):
        pending, _ = _read(path)
        if pending is None:
            continue
        if pending.state is PendingState.SUBMITTING:
            count += now - pending.at <= SUBMITTING_STALE_S
        elif pending.state is PendingState.QUEUED:
            jobs = queue.jobs()
            if jobs is None or pending.job in jobs:  # a job slurm cannot be asked about holds its slot
                count += 1
            else:
                _write(
                    path,
                    pending.model_copy(update={"state": PendingState.JOB_ENDED_WITHOUT_RECORD, "at": now}),
                )
        if count >= MAX_LIVE_TUNES:
            break
    return count


def _recorded(path: Path, pending: Pending, now: float, queue: _Queue) -> Outcome:
    """What a ``queued`` file means now: its job is live, cannot be told, or has ended."""
    job = pending.job or ""
    jobs = queue.jobs()
    if jobs is None:
        return Outcome(
            TuneState.NO_SCHEDULER, job, f"slurm could not say whether tune job {job} has finished"
        )
    if job in jobs:
        return Outcome(TuneState.ALREADY_QUEUED, job, f"a tune is already queued as slurm job {job}")
    ended = pending.model_copy(update={"state": PendingState.JOB_ENDED_WITHOUT_RECORD, "at": now})
    _write(path, ended)
    return _failed(path, ended)


def _failed(path: Path, pending: Pending) -> Outcome:
    """The answer for a change whose tune did not happen or finish; the note names the file to delete."""
    again = f"delete {path} to queue it again"
    if pending.state is PendingState.JOB_ENDED_WITHOUT_RECORD:
        where = f"its log is {pending.log}" if pending.log else "it left no log path"
        note = f"tune job {pending.job} ended and no record was adopted ({where}); {again}"
        return Outcome(TuneState.FAILED_BEFORE, pending.log, note)
    return Outcome(
        TuneState.FAILED_BEFORE,
        pending.reason,
        f"the last attempt to queue a tune failed: {pending.reason}; a lookup an hour later tries again",
    )


def _settled(path: Path, pending: Pending, stale: list[str]) -> Outcome:
    """The answer for a permanent state: nothing is queued until a person deletes the file."""
    again = f"delete {path} to queue it again"
    if pending.state is PendingState.MEASURED_NOT_ADOPTED:
        note = f"a tune measured this model but its result was not adopted: {pending.reason}; {again}"
        return Outcome(TuneState.MEASURED_NOT_ADOPTED, pending.reason, note)
    if pending.state is PendingState.ADOPTED:
        still = ", ".join(stale) or "no record of this model on this hardware answers it"
        adopted = f"the tune queued for this change was adopted ({pending.record})"
        note = f"{adopted} yet this lookup still wants one ({still}); {again}"
        return Outcome(TuneState.FAILED_BEFORE, str(path), note)
    return _failed(path, pending)


def _existing(path: Path, stale: list[str], now: float, queue: _Queue) -> _Existing:
    """What the change's file says to do; no outcome: there is none (or it is due a retry), so submit."""
    pending, problem = _read(path)
    if pending is None:
        note = f"a tune's pending file cannot be read: {problem}; delete {path} to queue it again"
        return _Existing(Outcome(TuneState.FAILED_BEFORE, problem, note)) if problem else _Existing(None)
    if pending.state is PendingState.SUBMITTING:
        if now - pending.at <= SUBMITTING_STALE_S:
            return _Existing(
                Outcome(TuneState.ALREADY_QUEUED, None, f"a tune is being queued by process {pending.pid}")
            )
        reason = f"process {pending.pid} never finished queuing it"
        pending = Pending(state=PendingState.SUBMIT_FAILED, at=pending.at, reason=reason)
    if pending.state is PendingState.SUBMIT_FAILED and now - pending.at >= RETRY_AFTER_S:
        return _Existing(None, since=pending.at)
    if pending.state is PendingState.QUEUED:
        return _Existing(_recorded(path, pending, now, queue))
    return _Existing(_settled(path, pending, stale))


def _earlier_job(path: Path, claim: Pending, change: _Change, since: float, queue: _Queue) -> Outcome | None:
    """A job an earlier attempt left behind (its id was lost) is adopted, never submitted again.

    A job named for the change that slurm lists as unfinished becomes ``queued``; one that accounting shows
    as finished becomes ``job_ended_without_record``. None: there is none, so a new one may be submitted.
    Slurm not answering is not an answer: nothing is submitted.
    """
    log_path = str(settings_dir() / "tune-logs" / f"{change.id}.log")
    jobs = queue.jobs()
    if jobs is None:
        return Outcome(
            TuneState.NO_SCHEDULER, None, "slurm could not say whether the earlier attempt left a job"
        )
    live = next((job for job, listed in jobs.items() if listed == change.job), None)
    if live is not None:
        _write(path, claim.model_copy(update={"state": PendingState.QUEUED, "job": live, "log": log_path}))
        return Outcome(TuneState.ALREADY_QUEUED, live, f"a tune is already queued as slurm job {live}")
    finished = _finished_job(change.job, since)
    if finished is None:
        return Outcome(
            TuneState.NO_SCHEDULER,
            None,
            "slurm's accounting could not say whether the earlier attempt left a job",
        )
    if finished:
        ended = claim.model_copy(
            update={"state": PendingState.JOB_ENDED_WITHOUT_RECORD, "job": finished, "log": log_path}
        )
        _write(path, ended)
        return _failed(path, ended)
    return None


def _create(path: Path, pending: Pending) -> bool:
    """Create ``path`` holding ``pending`` unless it exists; whether this call made it.

    The content is written to a temporary file first and linked to the name, which fails if the name exists:
    an exclusive create, and a file nobody ever sees empty or half written.
    """
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(pending.model_dump_json(indent=2, exclude_none=True) + "\n")
        os.link(temp, path)
    except FileExistsError:
        return False
    finally:
        Path(temp).unlink(missing_ok=True)
    return True


def _python() -> str:
    """The interpreter that runs ``submit.py``: the machine's, so a job never depends on a project's venv."""
    return str(SYSTEM_PYTHON) if SYSTEM_PYTHON.exists() else "python3"


def _submit_argv(
    script: Path, recipe: TuneRecipe, change: str, what: str, log_path: Path, name: str
) -> list[str]:
    """The ``submit.py`` call (design 5.2): the tune runs under ``timeout``, marked as a tune's own load."""
    return [
        _python(), str(script),
        "--project", recipe.project,
        "--what", what,
        "--class", "restart",
        "--gpu",
        "--no-check", NO_CHECK,
        "--name", name,
        "--workdir", str(recipe.workdir or Path.cwd()),
        "--log", str(log_path),
        "--",
        "timeout", f"--kill-after={KILL_AFTER_S}", str(TUNE_LIMIT_S),
        "env", f"{TUNING_ENV}={TUNING_ON}", f"{TUNE_CHANGE_ENV}={change}",
        *recipe.command,
    ]  # fmt: skip


def _run_submit(argv: list[str]) -> tuple[str, str]:
    """Run ``submit.py``: (the job id, ""), or ("", why not). The job id is the last line it prints."""
    try:
        done = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, errors="replace", timeout=SUBMIT_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return "", f"submit.py did not answer in {SUBMIT_TIMEOUT_S} s"
    except OSError as exc:
        return "", f"submit.py could not be run: {exc}"
    if done.returncode:
        return "", _last_line(done.stderr) or f"submit.py exited with status {done.returncode}"
    job = _last_line(done.stdout)
    return (job, "") if job.isdigit() else ("", f"submit.py printed no job id (last line {job!r})")


def _script() -> Path | None:
    """``$LOCALMODELS_SUBMIT``, else the default; an empty variable is no slurm at all."""
    value = os.environ.get(SUBMIT_ENV)
    if value is None:
        return DEFAULT_SUBMIT.expanduser()
    return Path(value).expanduser() if value else None


def request(
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    hardware: str,
    build: str | None,
    plan: TunePlan,
    stale: list[str],
    recipe: TuneRecipe,
    now: float | None = None,
) -> Outcome:
    """Queue the tune this lookup wants unless one is already queued, failed, or the cap is reached.

    ``now`` is for tests. Never raises.
    """
    now = time.time() if now is None else now
    script = _script()
    if script is None or not script.exists():
        where = f"{SUBMIT_ENV} is empty" if script is None else f"{script} does not exist"
        return Outcome(TuneState.NO_SCHEDULER, None, f"{where}, so there is no slurm to queue it on")
    id_ = change_id(model, kind, hardware=hardware, build=build, plan=plan, stale=stale)
    name = f"tune-{_slug(model.repository)}-{id_}"
    change = _Change(id_, name, f"{recipe.project}:restart:{name}", trigger(plan, stale))
    claim = Pending(
        state=PendingState.SUBMITTING,
        at=now,
        pid=os.getpid(),
        trigger=change.why,
        key=model.key,
        kind=kind.value,
    )
    path = pending_dir() / f"{id_}.json"
    try:
        with _locked(path.parent / "tune"):
            if (answer := _claim(path, claim, change, stale)) is not None:
                return answer
    except OSError as exc:
        return Outcome(
            TuneState.FAILED_BEFORE, str(exc), f"a tune cannot be recorded in {path.parent}: {exc!r}"
        )
    return _submit(path, model, kind, recipe, claim, change)


def _claim(path: Path, claim: Pending, change: _Change, stale: list[str]) -> Outcome | None:
    """Under the lock: an answer when no job is to be submitted, else None with the change's file claimed."""
    queue = _Queue()
    found = _existing(path, stale, claim.at, queue)
    if found.outcome is not None:
        return found.outcome
    if found.since is not None:
        if (earlier := _earlier_job(path, claim, change, found.since, queue)) is not None:
            return earlier
        path.unlink(missing_ok=True)
    if _live_count(path.parent, claim.at, queue) >= MAX_LIVE_TUNES:
        note = f"{MAX_LIVE_TUNES} tune jobs are already queued; a later lookup will queue it"
        return Outcome(TuneState.WAITING, None, note)
    if not _create(path, claim):
        return Outcome(TuneState.ALREADY_QUEUED, None, "another process queued it just now")
    return None


def _submit(
    path: Path, model: ModelIdentity, kind: WorkloadKind, recipe: TuneRecipe, claim: Pending, change: _Change
) -> Outcome:
    """Run ``submit.py`` for a claimed change and record the result; any failure is ``submit_failed``."""
    log_path = settings_dir() / "tune-logs" / f"{change.id}.log"
    try:
        what = f"localmodels tune: {model.repository} {kind.value} ({change.why})"
        argv = _submit_argv(_script() or DEFAULT_SUBMIT, recipe, change.id, what, log_path, change.name)
        job, failure = _run_submit(argv)
    except Exception as exc:  # noqa: BLE001 - nothing may fail a model load, and the claim must not be left behind
        job, failure = "", f"{type(exc).__name__}: {exc}"
    if failure:
        done = claim.model_copy(update={"state": PendingState.SUBMIT_FAILED, "reason": failure})
        outcome = _failed(path, done)
    else:
        done = claim.model_copy(update={"state": PendingState.QUEUED, "job": job, "log": str(log_path)})
        outcome = Outcome(TuneState.QUEUED, job, f"queued as slurm job {job}; log {log_path}")
    try:
        _write(path, done)
    except OSError as exc:
        log.warning("tune %s: its pending file could not be updated: %r", change.id, exc)
    return outcome


def job_finished(change: str, *, key: str, kind: WorkloadKind, refusal: str | None, record: str = "") -> None:
    """Called by ``adopt`` in a tune job: tell the change's file how the adopt of ``key``'s ``kind`` went.

    ``refusal`` is the ``AdoptError`` message, or None when the record was adopted (then ``record`` says
    which). Acts only when the file was made for that model and kind: another model's adopt by the same job
    says nothing about this change. Does nothing for a change that has no file, and never raises.
    """
    if not _CHANGE_ID.fullmatch(change):
        return
    path = pending_dir() / f"{change}.json"
    pending, _ = _read(path)
    if pending is None or (pending.key, pending.kind) != (key, kind.value):
        return
    adopted = refusal is None
    update = {
        "state": PendingState.ADOPTED if adopted else PendingState.MEASURED_NOT_ADOPTED,
        "at": time.time(),
        "reason": refusal,
        "record": record or None,
    }
    try:
        _write(path, pending.model_copy(update=update))
    except OSError as exc:
        log.warning("tune %s: its pending file could not be updated: %r", change, exc)
