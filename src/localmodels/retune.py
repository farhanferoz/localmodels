"""The automatic re-tune: queue one tune job per model and change through slurm, and remember it (design 5).

``settings.lookup`` calls ``request`` when a tune is due and nothing rules it out; this module is imported
only then. One tune job is queued per *change* (a model, a kind, the hardware, the build and what went
stale, as an 8-character id). The id's file under the store, ``pending/<change id>.json``, is created with exclusive create
while holding a lock, so two processes or projects asking at once cannot both submit. Its states:

* ``submitting``: the creator is running ``submit.py``; older than the submit timeout plus a margin counts as
  ``submit_failed``.
* ``queued``: the job id is in the file; live only while slurm lists the job as unfinished.
* ``submit_failed``: no job was queued; a later lookup tries again after an hour.
* ``job_ended_without_record``: the job ended and no record was adopted; no automatic retry (a person deleting
  the file, or a new change id, re-arms it).
* ``measured_not_adopted``: the tune measured and ``adopt`` refused its result; written by ``adopt`` itself.

Nothing here raises out of ``request``: a failure is a ``TuneState`` and a note. Torch-free. The scheduler is
reached by running slurm's ``submit.py`` as a subprocess (``$LOCALMODELS_SUBMIT``), never by importing it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
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
MAX_LIVE_TUNES: Final = 2  # simultaneous localmodels tune jobs (design 12, question 2)
SUBMIT_TIMEOUT_S: Final = 30
SUBMITTING_STALE_S: Final = SUBMIT_TIMEOUT_S + 30  # a ``submitting`` file this old was left by a dead process
RETRY_AFTER_S: Final = 3600  # a failed submission is tried again by a later lookup after this
SQUEUE_TIMEOUT_S: Final = 10
TUNE_LIMIT_S: Final = 900  # the hard ceiling of D2, by ``timeout``
KILL_AFTER_S: Final = 30
_CHANGE_ID_CHARS: Final = 8
_CHANGE_ID = re.compile(r"[0-9a-f]{8}")
_INVALID_JOB: Final = "Invalid job id"  # squeue's complaint about a job slurm has forgotten

log = logging.getLogger(__name__)


class JobState(StrEnum):
    """The squeue states of a job that has not finished (slurm's ``gpu_jobs.State``)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIGURING = "CONFIGURING"
    COMPLETING = "COMPLETING"
    SUSPENDED = "SUSPENDED"


_LIVE: Final = frozenset(JobState)

#: Asked for explicitly: slurm keeps finished jobs for a day, and ``scontrol show job`` still knows them.
_UNFINISHED: Final = ",".join(state.lower() for state in JobState)


class PendingState(StrEnum):
    """Where a change's tune stands (the module docstring)."""

    SUBMITTING = "submitting"
    QUEUED = "queued"
    SUBMIT_FAILED = "submit_failed"
    JOB_ENDED_WITHOUT_RECORD = "job_ended_without_record"
    MEASURED_NOT_ADOPTED = "measured_not_adopted"


class Pending(BaseModel):
    """The content of ``pending/<change id>.json``; ``at`` is when it entered ``state`` (epoch seconds)."""

    state: PendingState
    at: float
    pid: int | None = None  # of the process running submit.py
    job: str | None = None
    log: str | None = None
    reason: str | None = None
    trigger: str = ""


class Outcome(NamedTuple):
    """What ``request`` did: the state, the job id, log or reason behind it, and a sentence for the notes."""

    state: TuneState
    detail: str | None = None
    note: str = ""


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


def job_is_live(job: str) -> bool | None:
    """Whether slurm lists ``job`` as unfinished; None when it could not be asked.

    ``squeue --states`` with the unfinished states prints nothing for a job that has finished, whether
    slurm still remembers it or not (it answers "Invalid job id" once it has forgotten it).
    """
    argv = ["squeue", "-h", "-j", job, "-o", "%T", "--states", _UNFINISHED]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=SQUEUE_TIMEOUT_S, check=False)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode == 0:
        return any(line.strip() in _LIVE for line in done.stdout.splitlines())
    return False if _INVALID_JOB in done.stderr else None


def _live_count(folder: Path, now: float) -> int:
    """Tune jobs holding a slot: a fresh ``submitting`` file, or a ``queued`` one whose job is not finished."""
    count = 0
    for path in folder.glob("*.json"):
        pending, _ = _read(path)
        if pending is None:
            continue
        if pending.state is PendingState.SUBMITTING:
            count += now - pending.at <= SUBMITTING_STALE_S
        elif pending.state is PendingState.QUEUED:
            count += (
                job_is_live(pending.job or "") is not False
            )  # a job slurm cannot be asked about holds its slot
    return count


def _recorded(path: Path, pending: Pending, measured_at: float | None, now: float) -> Outcome | None:
    """What a ``queued`` file means now; None: a record was adopted after the submission, so it is removed."""
    job = pending.job or ""
    live = job_is_live(job)
    if live is None:
        return Outcome(
            TuneState.NO_SCHEDULER, job, f"slurm could not say whether tune job {job} has finished"
        )
    if live:
        return Outcome(TuneState.ALREADY_QUEUED, job, f"a tune is already queued as slurm job {job}")
    if measured_at is not None and measured_at > pending.at:
        path.unlink(missing_ok=True)
        return None
    ended = pending.model_copy(update={"state": PendingState.JOB_ENDED_WITHOUT_RECORD, "at": now})
    _write(path, ended)
    return _failed(ended)


def _failed(pending: Pending) -> Outcome:
    if pending.state is PendingState.JOB_ENDED_WITHOUT_RECORD:
        where = f"its log is {pending.log}" if pending.log else "it left no log path"
        note = f"tune job {pending.job} ended and no record was adopted ({where}); deleting its file retries"
        return Outcome(TuneState.FAILED_BEFORE, pending.log, note)
    return Outcome(
        TuneState.FAILED_BEFORE, pending.reason, f"the last attempt to queue a tune failed: {pending.reason}"
    )


def _existing(path: Path, measured_at: float | None, now: float) -> Outcome | None:
    """What the change's pending file says to do; None: there is none (or it was cleared), so submit."""
    pending, problem = _read(path)
    if pending is None:
        note = f"a tune's pending file cannot be read: {problem}"
        return Outcome(TuneState.FAILED_BEFORE, problem, note) if problem else None
    if pending.state is PendingState.SUBMITTING:
        if now - pending.at <= SUBMITTING_STALE_S:
            return Outcome(TuneState.ALREADY_QUEUED, None, f"a tune is being queued by process {pending.pid}")
        reason = f"process {pending.pid} never finished queuing it"
        pending = Pending(state=PendingState.SUBMIT_FAILED, at=pending.at, reason=reason)
    if pending.state is PendingState.SUBMIT_FAILED and now - pending.at >= RETRY_AFTER_S:
        path.unlink(missing_ok=True)
        return None
    if pending.state is PendingState.QUEUED:
        return _recorded(path, pending, measured_at, now)
    if pending.state is PendingState.MEASURED_NOT_ADOPTED:
        note = f"a tune measured this model but its result was not adopted: {pending.reason}"
        return Outcome(TuneState.MEASURED_NOT_ADOPTED, pending.reason, note)
    return _failed(pending)


def _create(path: Path, pending: Pending) -> bool:
    """Create ``path`` holding ``pending`` unless it exists (exclusive create); whether this call made it."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(pending.model_dump_json(indent=2, exclude_none=True) + "\n")
    return True


def _submit_argv(
    script: Path, recipe: TuneRecipe, change: str, what: str, log_path: Path, name: str
) -> list[str]:
    """The ``submit.py`` call (design 5.2): the tune runs under ``timeout``, marked as a tune's own load."""
    return [
        "python3", str(script),
        "--project", recipe.project,
        "--what", what,
        "--class", "restart",
        "--gpu",
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
        done = subprocess.run(argv, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT_S, check=False)  # noqa: S603
    except subprocess.TimeoutExpired:
        return "", f"submit.py did not answer in {SUBMIT_TIMEOUT_S} s"
    except OSError as exc:
        return "", f"submit.py could not be run: {exc}"
    if done.returncode:
        return "", _last_line(done.stderr) or f"submit.py exited with status {done.returncode}"
    job = _last_line(done.stdout)
    return (job, "") if job.isdigit() else ("", f"submit.py printed no job id (last line {job!r})")


def _script() -> Path:
    return Path(os.environ.get(SUBMIT_ENV) or DEFAULT_SUBMIT).expanduser()


def request(  # noqa: PLR0913 - what the lookup knows
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    hardware: str,
    build: str | None,
    plan: TunePlan,
    stale: list[str],
    recipe: TuneRecipe,
    measured_at: float | None = None,
    now: float | None = None,
) -> Outcome:
    """Queue the tune this lookup wants unless one is already queued, failed, or the cap is reached.

    ``measured_at`` is when the record the lookup found was measured: a queued job that has ended after
    which a record was adopted is forgotten, not reported as failed. ``now`` is for tests. Never raises.
    """
    now = time.time() if now is None else now
    script = _script()
    if not script.exists():
        return Outcome(
            TuneState.NO_SCHEDULER, None, f"{script} does not exist, so there is no slurm to queue it on"
        )
    change = change_id(model, kind, hardware=hardware, build=build, plan=plan, stale=stale)
    folder = pending_dir()
    path = folder / f"{change}.json"
    why = trigger(plan, stale)
    try:
        with _locked(folder / "tune"):
            if (found := _existing(path, measured_at, now)) is not None:
                return found
            if _live_count(folder, now) >= MAX_LIVE_TUNES:
                return Outcome(
                    TuneState.WAITING,
                    None,
                    f"{MAX_LIVE_TUNES} tune jobs are already queued; a later lookup will queue it",
                )
            if not _create(
                path, Pending(state=PendingState.SUBMITTING, at=now, pid=os.getpid(), trigger=why)
            ):
                return Outcome(TuneState.ALREADY_QUEUED, None, "another process queued it just now")
    except OSError as exc:
        return Outcome(TuneState.FAILED_BEFORE, str(exc), f"a tune cannot be recorded in {folder}: {exc!r}")
    return _submit(path, model, kind, recipe, change, why, now)


def _submit(
    path: Path,
    model: ModelIdentity,
    kind: WorkloadKind,
    recipe: TuneRecipe,
    change: str,
    why: str,
    now: float,
) -> Outcome:
    log_path = settings_dir() / "tune-logs" / f"{change}.log"
    argv = _submit_argv(
        _script(),
        recipe,
        change,
        f"localmodels tune: {model.repository} {kind.value} ({why})",
        log_path,
        f"tune-{_slug(model.repository)}-{change}",
    )
    job, failure = _run_submit(argv)
    if failure:
        done = Pending(state=PendingState.SUBMIT_FAILED, at=now, reason=failure, trigger=why)
        outcome = _failed(done)
    else:
        done = Pending(state=PendingState.QUEUED, at=now, job=job, log=str(log_path), trigger=why)
        outcome = Outcome(TuneState.QUEUED, job, f"queued as slurm job {job}; log {log_path}")
    try:
        _write(path, done)
    except OSError as exc:
        log.warning("tune %s: its pending file could not be updated: %r", change, exc)
    return outcome


def job_finished(change: str, refusal: str | None) -> None:
    """Called by ``adopt`` in a tune job: success removes the change's file, a refusal records why.

    ``refusal`` is the ``AdoptError`` message, or None when the record was adopted. Does nothing for a
    change that has no file, and never raises.
    """
    if not _CHANGE_ID.fullmatch(change):
        return
    path = pending_dir() / f"{change}.json"
    try:
        if refusal is None:
            path.unlink(missing_ok=True)
            return
        pending, _ = _read(path)
        if pending is not None:
            refused = pending.model_copy(
                update={"state": PendingState.MEASURED_NOT_ADOPTED, "at": time.time(), "reason": refusal}
            )
            _write(path, refused)
    except OSError as exc:
        log.warning("tune %s: its pending file could not be updated: %r", change, exc)
