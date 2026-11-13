"""`localmodels refresh run|build|measure|status`: from "llama.cpp changed" to "re-measured and adopted".

``run`` takes seconds, runs nothing on the GPU and only submits slurm jobs: a CPU job that builds the newest
llama.cpp release (``build``) and a GPU job (``measure``) that compares that build with ``current``, switches
to it when the comparison passes, and re-measures rules K1-K3 on whichever build is ``current`` at its
end. The monthly systemd timer in ``deploy/`` starts ``run``; ``status`` prints the newest run.

Folders: a run's files live in ``$LOCALMODELS_REFRESH_DIR/<YYYY-MM-DDTHHMM>/`` (default
``~/cache/localmodels/refresh``): ``run.json`` (what ``run`` decided and submitted), ``build.json``,
``result.json`` (what ``measure`` ran), ``comparison.json``, and the output of every tool call. The state
(``previous_current`` for a rollback, ``tried`` commits with their verdicts) is
``$LOCALMODELS_STATE_DIR/refresh/state.json``. Design: docs/research/engine-refresh-design-2026-11-07.md.

Nothing here runs a llama.cpp program except ``measure``, which holds the GPU lease quiet while it does.

``run`` also walks the container engines (vLLM, SGLang, TensorRT-LLM) through ``engine_refresh``, which is
loaded only then; the job subcommands ``engine-pull``, ``engine-measure``, ``engine-calibrate``,
``same-client`` and ``sparkrun-compare`` belong to it (docs/research/engine-registry-design-2026-11-08.md).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel, Field

from . import rules
from .atomic import read_model, write_model
from .enums import WorkloadKind
from .lease import DEFAULT_LEASE_DIR, GpuLease, QuietCard, QuietCardTimeoutError
from .llama_builds import (
    COMPARISON_FILE,
    REFERENCE_BUILD,
    REPO_URL,
    builds_dir,
    compile_commit,
    current_commit,
    current_server,
    folder_for,
    llama_home,
    prune_builds,
    server_binary,
    switch,
)
from .pinned import sha256_file, snapshot_dir
from .pull import _default_hub_cache

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping, Sequence

    from .llama_tune import KlStats

log = logging.getLogger(__name__)

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1

REFRESH_DIR_ENV: Final = "LOCALMODELS_REFRESH_DIR"
DEFAULT_REFRESH_DIR: Final = Path("~/cache/localmodels/refresh")
STATE_SUBDIR: Final = "refresh"
STATE_FILE: Final = "state.json"
RUN_FILE: Final = "run.json"
BUILD_FILE: Final = "build.json"
RESULT_FILE: Final = "result.json"
KL_BASE_FILE: Final = "k3.kld"
RUN_ID_FORMAT: Final = "%Y-%m-%dT%H%M"
_RUN_ID = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{4}$")
PROJECT: Final = "localmodels"  # the slurm project of both jobs
LEASE_PROJECT: Final = "localmodels-refresh"  # the name on the quiet request
NOTHING_TO_DO: Final = "nothing to do"  # the front of ``RunRecord.skipped`` when no part has work
JOB_PLACEHOLDER: Final = "<build job id>"  # what a dry run shows for the id of a job it did not submit
LS_REMOTE_TIMEOUT_S: Final = 60

# The slurm jobs (design 2.1).
BUILD_CPUS: Final = 8
BUILD_MEM: Final = "8G"  # measured: b12189 peaked at 5.2 GB on 8 cores (8 min 9 s), slurm advised 6G
MEASURE_MEM: Final = "52G"  # measured: slurm 2157 peaked at 42.4 GB (GPU memory included) of 42.9 GB asked
MEASURE_LIMIT_S: Final = 1200  # the hard stop of the whole measurement, by ``timeout``
KILL_AFTER_S: Final = 30
# `run --urgent`: the measure job goes through slurm's urgent.py (beside submit.py), which pauses what can be
# paused in its way; it has no --after, so the candidate must already be built.
URGENT_SCRIPT: Final = "urgent.py"
URGENT_TIME: Final = "1:30:00"  # slurm's limit: the job wrapper's wait for the card (up to 1 h), then 1200 s
URGENT_TIMEOUT_S: Final = 300  # for urgent.py itself, which looks at what is in the way before submitting
_URGENT_JOB = re.compile(r"\[urgent\] job (\d+) submitted")
#: slurm's pending reason for a job whose ``--after`` job did not end cleanly: it can never start, and this
#: slurm does not cancel such jobs by itself (SchedulerParameters has no kill_invalid_depend).
DEAD_DEPENDENCY: Final = "DependencyNeverSatisfied"
TOOL_TIMEOUT_S: Final = 300.0  # design 2.3: each tool run has its own
SIGTERM_EXIT: Final = 128 + signal.SIGTERM

# The verdict (design 2.4, D13).
MIN_SLOWDOWN: Final = (
    0.02  # a figure may be this much slower than the baseline's mean, or its spread if larger
)
ROUNDS: Final = 2  # the comparison's rounds; each arm is measured once per round
SLOT_COUNTS: Final = (1, 4, 16)  # llama-batched-bench's -npl: requests at once
# K3 on the final current (design 3; October: within +/-5e-5 of the base, 100 % same top token).
K3_MAX_KL: Final = 1e-4
K3_MIN_TOP: Final = 0.999
KL_CHUNKS: Final = 40  # of 512 tokens: about 5 GB for the base file at a 248k vocabulary (design 2.3)
KL_CONTEXT: Final = 512
COMPARISON_STEP: Final = "cmp-"  # the front of every step name that belongs to the comparison
STEP_OK: Final = "ok"
NOT_MEASURED: Final = "not measured: "  # the front of a step's outcome when it did not measure

#: K1's figures: prompt and generation speed of the small model, then of the production model.
K1_FIGURES: Final = ("small_pp512", "small_pp2048", "small_tg128", "prod_pp512", "prod_tg128")
REMEASURED_KINDS: Final = (WorkloadKind.K1, WorkloadKind.K2, WorkloadKind.K3)


class Tool(StrEnum):
    """The llama.cpp programs ``measure`` runs, by file name in a build's ``bin/``."""

    BENCH = "llama-bench"
    BATCHED = "llama-batched-bench"
    PERPLEXITY = "llama-perplexity"


class Arm(StrEnum):
    """The two builds a comparison measures."""

    CURRENT = "current"
    CANDIDATE = "candidate"


class JobName(StrEnum):
    """The names of the two slurm jobs; slurm's own job name is ``<project>:<class>:<name>``."""

    BUILD = "refresh-build"
    MEASURE = "refresh-measure"


# slurm's ``--no-check`` reason per job: each job's first step is the work itself, with no smaller CPU run.
NO_CHECK: Final[dict[JobName, str]] = {
    JobName.BUILD: "the build is itself the check: a CPU job whose first step is the compile",
    JobName.MEASURE: "the job is itself a GPU measurement of a built llama.cpp; it has no CPU dry run",
}


class Model(NamedTuple):
    """A model file in the Hugging Face cache by repository, revision and file name."""

    repository: str
    revision: str
    filename: str


SMALL_MODEL: Final = Model(
    "unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q4_K_M.gguf"
)
PRODUCTION_MODEL: Final = Model(
    "Qwen/Qwen3-30B-A3B-GGUF", "e4d4bafdfb96a411a163846265362aceb0b9c63a", "Qwen3-30B-A3B-Q4_K_M.gguf"
)

#: The nine figures a comparison judges: llama-bench at (512, 0), (2048, 0), (0, 128) and batched-bench's
#: prompt and generation speed at each slot count.
SPEED_FIGURES: Final = (
    "small_pp512",
    "small_pp2048",
    "small_tg128",
    *(f"small_{part}_at_{slots}" for part in ("pp", "tg") for slots in SLOT_COUNTS),
)
_BENCH_NAMES: Final = SPEED_FIGURES[:3]
_BATCHED_NAMES: Final = SPEED_FIGURES[3:]


def model_file(pin: Model) -> Path | None:
    """The model file in the Hub cache; None when it is not there."""
    path = snapshot_dir(pin.repository, pin.revision, _default_hub_cache()) / pin.filename
    return path if path.is_file() else None


# ---------------------------------------------------------------------------------------------------------
# Folders and files


def refresh_dir() -> Path:
    """The folder of the run folders: ``$LOCALMODELS_REFRESH_DIR``, else ``~/cache/localmodels/refresh``."""
    return Path(os.environ.get(REFRESH_DIR_ENV) or DEFAULT_REFRESH_DIR).expanduser()


def state_path() -> Path:
    """The refresh state file, under the same state directory as the rule records."""
    return rules.state_dir() / STATE_SUBDIR / STATE_FILE


class Tried(BaseModel):
    """The verdict on one candidate commit: it stands until a person asks for a retry."""

    passed: bool
    run: str


class RefreshState(BaseModel):
    """What the next run remembers."""

    previous_current: str | None = (
        None  # the build ``current`` was before the last switch, kept for a rollback
    )
    tried: dict[str, Tried] = Field(default_factory=dict)  # short commit -> verdict


def read_state() -> RefreshState:
    """The state file; an empty state when there is none or it cannot be read."""
    return read_model(state_path(), RefreshState) or RefreshState()


def write_state(state: RefreshState) -> None:
    """Store ``state`` atomically."""
    write_model(state_path(), state)


class JobRecord(BaseModel):
    """One submission: its argv and the job id slurm gave it, or why none."""

    name: str
    argv: list[str]
    job: str = ""
    error: str = ""


class EnginePlan(BaseModel):
    """What the engine walk decided for one version source (strings: ``engine_refresh`` owns the enums)."""

    engine: str
    repository: str
    publisher: str
    tag: str = ""  # the candidate; empty when there is none
    mode: str = ""  # bootstrap or candidate, when there is one
    reason: str = ""  # why there is no candidate, or what else the plan notes


class RunRecord(BaseModel):
    """``run.json``: what ``run`` found and submitted."""

    time: str
    skipped: str = ""
    candidate_tag: str = ""
    candidate_commit: str = ""
    no_candidate: str = ""  # why there is no candidate
    stale_rules: list[str] = Field(default_factory=list)
    removed_builds: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)  # refresh jobs that could never start, cancelled
    queued: list[str] = Field(default_factory=list)  # live refresh jobs that made this run skip
    jobs: list[JobRecord] = Field(default_factory=list)
    engines: list[EnginePlan] = Field(default_factory=list)  # the container engines' candidates
    engine_reports: list[str] = Field(default_factory=list)  # reaper, prune and sparkrun-move reports


class BuildResult(BaseModel):
    """``build.json``: the build job's outcome; ``commit`` is the short commit built (the folder name)."""

    tag: str
    commit: str = ""
    ok: bool
    error: str = ""
    seconds: float = 0.0


class Verdict(BaseModel):
    """D13's verdict on a candidate."""

    passed: bool
    speed_failures: list[str]
    quality_failures: list[str]
    slowdown: float  # the largest relative slowdown of the nine figures; 0 or less when none is slower
    figures: dict[str, dict[str, list[float]]]  # name -> {"baseline": [r1, r2], "candidate": [r1, r2]}
    notes: str = ""


class Comparison(Verdict):
    """``comparison.json``: the fields ``llama_builds`` documents, plus what led to them."""

    baseline_commit: str
    measured_at: str
    candidate_tag: str
    slurm_job: str = ""
    kl: dict[str, dict[str, float] | None] = Field(default_factory=dict)  # "control" and "candidate"


class MeasureResult(BaseModel):
    """``result.json``: what ``measure`` ran and decided."""

    measured_at: str = ""
    slurm_job: str = ""
    not_measured: str = ""  # why nothing was measured (the card did not go quiet)
    error: str = ""  # the process was stopped (slurm's timeout) or failed outside any tool run
    steps: dict[str, str] = Field(default_factory=dict)  # step -> "ok" or why it did not measure
    comparison: Comparison | None = None
    switched: bool = False
    switch_problem: str = ""
    final_current: str = ""
    records: list[str] = Field(default_factory=list)  # the kinds whose rule record was written


class RunView(NamedTuple):
    """Everything a run folder holds, each part None when the file is not there yet."""

    folder: Path
    run: RunRecord | None
    build: BuildResult | None
    result: MeasureResult | None


def newest_run(root: Path | None = None) -> Path | None:
    """The newest run folder under ``root`` (default ``refresh_dir()``); None when there is none."""
    root = refresh_dir() if root is None else root
    if not root.is_dir():
        return None
    runs = sorted(p for p in root.iterdir() if p.is_dir() and _RUN_ID.match(p.name))
    return runs[-1] if runs else None


def newest_view(root: Path | None = None) -> RunView | None:
    """The newest run's files; None when there is no run."""
    folder = newest_run(root)
    if folder is None:
        return None
    return RunView(
        folder,
        read_model(folder / RUN_FILE, RunRecord),
        read_model(folder / BUILD_FILE, BuildResult),
        read_model(folder / RESULT_FILE, MeasureResult),
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f} %"


def _comparison_line(day: str, tag: str, view: RunView) -> str | None:
    """The headline when a comparison ran: switched, or kept back and why."""
    result = view.result
    comparison = result.comparison if result is not None else None
    if result is None or comparison is None:
        return None
    what = f"llama.cpp {tag} ({view.build.commit if view.build else '?'})"
    kl = (comparison.kl.get("candidate") or {}).get("mean_kl")
    figures = f"slowdown {_percent(comparison.slowdown)}" + (f", KL {kl:.3f}" if kl is not None else "")
    if result.switched:
        return f"refresh {day}: {what} switched ({figures})"
    failures = "; ".join([*comparison.speed_failures, *comparison.quality_failures][:2])
    return f"refresh {day}: {what} kept back: {failures or result.switch_problem or figures}"


def headline(view: RunView) -> str:
    """One line on what the run did with its candidate (the digest's line and the first line of `status`)."""
    day = view.folder.name[: rules.DATE_LENGTH]
    run, build, result = view.run, view.build, view.result
    if run is not None and run.skipped:
        return f"refresh {day}: skipped, {run.skipped}"
    tag = (build.tag if build else run.candidate_tag if run else "") or ""
    if build is not None and not build.ok:
        return f"refresh {day}: llama.cpp {tag} build failed: {build.error}"
    if line := _comparison_line(day, tag, view):
        return line
    if result is not None and result.not_measured:
        return f"refresh {day}: not measured: {result.not_measured}"
    if not tag:
        return f"refresh {day}: no candidate ({(run.no_candidate if run else '') or 'none'})"
    return f"refresh {day}: llama.cpp {tag}: no comparison yet"


def rules_line(view: RunView) -> str | None:
    """Which rules the run re-measured and which only have a reason; None until ``measure`` wrote records."""
    if view.result is None or not view.result.records:
        return None
    return (
        f"rules re-measured: {' '.join(view.result.records)}; "
        f"with a reason: {' '.join(rules.NOT_MEASURED_HERE)}"
    )


# ---------------------------------------------------------------------------------------------------------
# run


class Candidate(NamedTuple):
    """The newest llama.cpp release tag and the commit it points at."""

    tag: str
    commit: str


_TAG_LINE = re.compile(r"^(?P<sha>[0-9a-f]{40})\trefs/tags/(?P<tag>b(?P<number>\d+))(?P<peeled>\^\{\})?$")


def newest_tag(ls_remote: str, only: str | None = None) -> Candidate | None:
    """The release tag with the largest number after ``b``, from ``git ls-remote --tags`` text.

    The commit is the peeled (``^{}``) line's when the tag is annotated, else the tag line's. Other tags
    (``backup/...``, which the ``b*`` pattern also matches) are ignored. ``only`` picks that one tag instead.
    """
    commits: dict[str, str] = {}
    numbers: dict[str, int] = {}
    for line in ls_remote.splitlines():
        match = _TAG_LINE.match(line)
        if match is None or (only is not None and match.group("tag") != only):
            continue
        tag = match.group("tag")
        numbers[tag] = int(match.group("number"))
        if match.group("peeled") or tag not in commits:
            commits[tag] = match.group("sha")
    if not numbers:
        return None
    tag = max(numbers, key=lambda name: numbers[name])
    return Candidate(tag, commits[tag])


def _ls_remote() -> str:
    """``git ls-remote`` of llama.cpp's release tags."""
    done = subprocess.run(  # noqa: S603
        ["git", "ls-remote", "--tags", REPO_URL, "refs/tags/b*"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=LS_REMOTE_TIMEOUT_S,
    )
    return done.stdout


def candidate_for(
    ls_remote: Callable[[], str],
    current: str | None,
    state: RefreshState,
    *,
    retry: bool,
    tag: str | None = None,
) -> tuple[Candidate | None, str, bool]:
    """The release to compare with ``current`` (the newest, or ``tag``), or None and why not.

    The third value says whether the release list was read at all: when it was not, nothing is known about
    which build is the candidate.
    """
    try:
        found = newest_tag(ls_remote(), only=tag)
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"git ls-remote failed: {exc}", False
    if found is None:
        return (
            None,
            f"git ls-remote listed no tag {tag}" if tag else "git ls-remote listed no b<number> tag",
            True,
        )
    if current and found.commit.startswith(current):
        return None, f"current ({current}) is {found.tag}", True
    if not retry and any(found.commit.startswith(done) for done in state.tried):
        return None, f"{found.tag} was compared already (--retry compares again)", True
    return found, "", True


def _refresh_queue() -> list[tuple[str, str]] | None:
    """Each unfinished refresh job of this project as (id, slurm's reason); None when squeue could not say."""
    from . import retune  # noqa: PLC0415 - the CLI must not import the lookup's modules to start

    done = retune._run(["squeue", "-h", "-o", "%i|%j|%r", "--states", retune._UNFINISHED])  # noqa: SLF001
    if done is None or done.returncode:
        return None
    rows = [line.split("|", 2) for line in done.stdout.splitlines() if line.count("|") >= 2]  # noqa: PLR2004
    return sorted(
        (job, reason.strip())
        for job, name, reason in rows
        if name.startswith(f"{PROJECT}:") and "refresh-" in name
    )


def _scancel(job: str) -> str:
    """Cancel one of the refresh's own jobs; "" when slurm took it, else why not."""
    try:
        done = subprocess.run(  # noqa: S603
            ["scancel", job],  # noqa: S607
            capture_output=True, text=True, errors="replace", timeout=LS_REMOTE_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    return done.stderr.strip() or f"scancel exited {done.returncode}" if done.returncode else ""


def live_refresh_jobs(record: RunRecord, *, dry_run: bool) -> list[str] | None:
    """The refresh jobs still queued or running; None when slurm cannot say.

    A job that waits on a build job that did not end cleanly can never start; it is cancelled here (and
    listed in ``record.cancelled``) so that it does not hold every later run back.
    """
    queue = _refresh_queue()
    if queue is None:
        return None
    for job in (job for job, reason in queue if reason == DEAD_DEPENDENCY):
        why = "" if dry_run else _scancel(job)
        record.cancelled.append(f"{job}{': ' + why if why else ''}")
    return [job for job, reason in queue if reason != DEAD_DEPENDENCY]


def _refresh_command(home: str | None, command: str) -> list[str]:
    """This interpreter running ``localmodels refresh [--home H] <command>``, so a job runs this checkout."""
    return [sys.executable, "-m", "localmodels", "refresh", *(["--home", home] if home else []), command]


def _measure_command(home: str | None, run: Path) -> list[str]:
    """``refresh measure`` under ``timeout``, the hard stop of the whole measurement."""
    measure = [*_refresh_command(home, "measure"), "--run", str(run)]
    return ["timeout", f"--kill-after={KILL_AFTER_S}", str(MEASURE_LIMIT_S), *measure]


def _submit(
    script: Path,
    run: Path,
    name: str,
    what: str,
    flags: Sequence[str],
    command: Sequence[str],
    *,
    no_check: str,
    log_stem: str | None = None,
) -> list[str]:
    """One job's ``submit.py`` (or ``urgent.py``) call; ``no_check``: why it has no check before queuing."""
    from . import retune  # noqa: PLC0415 - the CLI must not import the lookup's modules to start

    return [
        retune._python(),  # noqa: SLF001
        str(script),
        "--project", PROJECT,
        "--name", name,
        "--what", what,
        "--workdir", str(run),  # a systemd user service starts in the home folder
        "--log", str(run / f"{log_stem or name.removeprefix('refresh-')}.log"),
        "--no-check", no_check,
        *flags,
        "--",
        *command,
    ]  # fmt: skip


def build_argv(script: Path, run: Path, tag: str, home: str | None) -> list[str]:
    """The ``submit.py`` call of the CPU build job (resume: cmake continues an interrupted build)."""
    flags = [
        "--class",
        "resume",
        "--resume-ok",
        "--no-anywhere",
        "--cpus",
        str(BUILD_CPUS),
        "--mem",
        BUILD_MEM,
    ]
    what = f"llama.cpp {tag} candidate build (refresh {run.name})"
    return _submit(
        script,
        run,
        JobName.BUILD,
        what,
        flags,
        [*_refresh_command(home, "build"), "--run", str(run), "--commit", tag],
        no_check=NO_CHECK[JobName.BUILD],
    )


def measure_argv(script: Path, run: Path, after: str | None, home: str | None) -> list[str]:
    """The ``submit.py`` call of the GPU measure job, chained after the build job when there is one."""
    flags = ["--class", "restart", "--gpu", "--mem", MEASURE_MEM, *(["--after", after] if after else [])]
    what = f"llama.cpp comparison and K1-K3 re-measure (refresh {run.name})"
    command = _measure_command(home, run)
    return _submit(script, run, JobName.MEASURE, what, flags, command, no_check=NO_CHECK[JobName.MEASURE])


def urgent_argv(script: Path, run: Path, home: str | None) -> list[str]:
    """The ``urgent.py`` call (beside ``script``) of the GPU measure job, for ``run --urgent``."""
    flags = ["--time", URGENT_TIME, "--gpu", "--mem", MEASURE_MEM, "--wait"]
    what = f"llama.cpp comparison and K1-K3 re-measure (refresh {run.name}, urgent)"
    return _submit(
        script.with_name(URGENT_SCRIPT),
        run,
        JobName.MEASURE,
        what,
        flags,
        _measure_command(home, run),
        no_check=NO_CHECK[JobName.MEASURE],
    )


def _run_urgent(argv: list[str]) -> tuple[str, str]:
    """Run ``urgent.py``: (the job id, ""), or ("", why not). It prints ``[urgent] job N submitted; ...``."""
    try:
        done = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, errors="replace", timeout=URGENT_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return "", f"urgent.py did not answer in {URGENT_TIMEOUT_S} s"
    except OSError as exc:
        return "", f"urgent.py could not be run: {exc}"
    match = _URGENT_JOB.search(done.stdout)
    if done.returncode or match is None:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        return "", lines[-1] if lines else f"urgent.py exited {done.returncode} without a job id"
    return match.group(1), ""


def built_folder(home: Path, commit: str) -> Path | None:
    """The finished build of ``commit``; None when it has no folder or no ``llama-server`` in it."""
    folder = folder_for(home, commit)
    return folder if folder is not None and server_binary(folder).is_file() else None


def _decide(args: argparse.Namespace, home: Path, record: RunRecord) -> tuple[Candidate | None, Path | None]:
    """Fill ``record`` with what is due: the candidate, the stale rules, the builds to remove.

    Returns the candidate and the submit script; ``record.skipped`` says why there is nothing to submit.
    The candidate's own build is never removed, so a build made ahead of the run is measured, not redone.
    """
    from . import retune  # noqa: PLC0415 - the CLI must not import the lookup's modules to start

    live = live_refresh_jobs(record, dry_run=args.dry_run)
    if live is None:
        record.skipped = "squeue did not answer, so whether a refresh job is still queued is unknown"
        return None, None
    if live:
        record.skipped, record.queued = f"{', '.join(live)} still queued", live
        return None, None
    if state_path().exists() and read_model(state_path(), RefreshState) is None:
        record.skipped = (
            f"{state_path()} cannot be read: repair or remove it (it names the rollback build, which a run "
            "would otherwise remove, and the candidates already compared)"
        )
        return None, None
    state = read_state()
    candidate, record.no_candidate, listed = candidate_for(
        _ls_remote, current_commit(home), state, retry=args.retry, tag=args.candidate
    )
    keep = [state.previous_current or ""]
    if candidate is not None:
        record.candidate_tag, record.candidate_commit = candidate.tag, candidate.commit
        folder = folder_for(home, candidate.commit)
        keep += [folder.name] if folder is not None else []
    # Without the release list the candidate is unknown, and its build made ahead may be any folder: keep all.
    removed = prune_builds(home, keep=keep, dry_run=args.dry_run) if listed else []
    record.removed_builds = [str(folder) for folder in removed]
    if args.dry_run:
        sys.stdout.writelines(f"would remove {folder}\n" for folder in removed)
    found = rules.assess(rules.rules_text(), rules.host_stamp(current_server(home)))
    record.stale_rules = [a.line for a in found]
    unanswered = {rules.RuleState.NEEDS_REMEASURE, rules.RuleState.DOES_NOT_HOLD}
    due = any(a.kind in REMEASURED_KINDS and a.state in unanswered for a in found)
    script = retune._script()  # noqa: SLF001 - `$LOCALMODELS_SUBMIT`, and the tests' tripwire
    if candidate is None and not due and not args.force_measure:
        record.skipped = f"{NOTHING_TO_DO} ({record.no_candidate}; K1-K3 answered)"
    elif script is None:
        record.skipped = f"no submit script (${retune.SUBMIT_ENV} is empty)"
    elif args.urgent and candidate is not None and built_folder(home, candidate.commit) is None:
        record.skipped = (
            f"--urgent needs {candidate.tag} built first (urgent.py cannot wait for a build job): "
            f"`localmodels llama build --commit {candidate.tag}`, or run without --urgent"
        )
    return candidate, script


class JobSubmitter:
    """Submits the jobs of one run, or prints them in a dry run; ``failed`` says whether one was refused."""

    def __init__(self, record: RunRecord, *, dry_run: bool) -> None:
        self.record = record
        self.dry_run = dry_run
        self.failed = False
        self.count = 0  # jobs submitted or, in a dry run, printed

    def submit(
        self, name: str, argv: list[str], how: Callable[[list[str]], tuple[str, str]] | None = None
    ) -> str | None:
        """Submit ``argv`` (through ``how``, default ``submit.py``); the job id, or None if not queued."""
        self.count += 1
        if self.dry_run:
            sys.stdout.write(f"{shlex.join(argv)}\n")
            return JOB_PLACEHOLDER
        from . import retune  # noqa: PLC0415 - the CLI must not import the lookup's modules to start

        job, why = (how or retune._run_submit)(argv)  # noqa: SLF001
        self.record.jobs.append(JobRecord(name=name, argv=argv, job=job, error=why))
        self.failed |= bool(why)
        sys.stdout.write(f"{name}: {'job ' + job if job else 'not submitted: ' + why}\n")
        return job or None


def _submit_jobs(
    args: argparse.Namespace, script: Path, run: Path, submitter: JobSubmitter, candidate: Candidate | None
) -> str | None:
    """Submit the build job (a candidate not built yet), then the measure job; the measure job's id.

    A candidate whose build is already finished gets its ``build.json`` written here instead of a build job.
    The id is None when the measure job was not queued (``submitter.failed`` then says so).
    """
    built = built_folder(llama_home(args.home), candidate.commit) if candidate is not None else None
    if candidate is not None and built is not None:
        if args.dry_run:
            sys.stdout.write(f"{candidate.tag} is already built in {built}: no build job\n")
        else:
            done = BuildResult(tag=candidate.tag, commit=built.name, ok=True, error="already built")
            write_model(run / BUILD_FILE, done)
    after: str | None = None
    if candidate is not None and built is None:
        after = submitter.submit(JobName.BUILD, build_argv(script, run, candidate.tag, args.home))
    if args.urgent:
        return submitter.submit(JobName.MEASURE, urgent_argv(script, run, args.home), _run_urgent)
    return submitter.submit(JobName.MEASURE, measure_argv(script, run, after, args.home))


def _walk_failed(record: RunRecord, step: str, exc: Exception) -> None:
    """Record and log that the container engines' walk failed; the llama.cpp part goes on without it."""
    why = f"engine walk failed while {step}: {type(exc).__name__}: {exc}"
    record.engine_reports.append(why)
    sys.stderr.write(f"{why}\n")
    log.error("engine walk failed while %s", step, exc_info=exc)


def cmd_run(args: argparse.Namespace) -> int:
    """Decide what is due and submit the jobs; ``--dry-run`` only prints the commands.

    The llama.cpp part decides first; the container engines are walked next (``engine_refresh``), inside
    the same skip rules. A run is skipped as "nothing to do" only when neither part submits a job.
    """
    now = datetime.now(UTC)
    run = refresh_dir() / now.strftime(RUN_ID_FORMAT)
    record = RunRecord(time=now.isoformat(timespec="seconds"))
    candidate, script = _decide(args, llama_home(args.home), record)
    idle = record.skipped.startswith(NOTHING_TO_DO)  # the llama.cpp part has nothing; the engines may
    failed = False
    if (record.skipped and not idle) or script is None:
        sys.stdout.write(f"skipped: {record.skipped}\n")
        failed = record.skipped.startswith("--urgent")  # asked for and not done: say so in the exit code
    else:
        from .engine_refresh import EngineWalk  # noqa: PLC0415 - loaded only when the walk runs

        if not args.dry_run:
            run.mkdir(parents=True, exist_ok=True)  # the jobs' --workdir and --log: there before they queue
        submitter = JobSubmitter(record, dry_run=args.dry_run)
        walk = EngineWalk(args, run, record, submitter, script)
        walked = True
        try:
            walk.decide()
        except Exception as exc:  # noqa: BLE001 - one broken part never stops the other: llama.cpp's jobs go
            _walk_failed(record, "deciding", exc)
            walked = False
        measure = None if idle else _submit_jobs(args, script, run, submitter, candidate)
        if walked:
            try:
                walk.submit(llama_measure_job=measure)
            except Exception as exc:  # noqa: BLE001 - recorded and logged, as above
                _walk_failed(record, "submitting", exc)
        if idle and not submitter.count:
            sys.stdout.write(f"skipped: {record.skipped}\n")
        else:
            record.skipped = ""
        failed = submitter.failed
    if not args.dry_run and not record.queued:  # a run still in progress keeps its folder the newest
        run.mkdir(parents=True, exist_ok=True)
        write_model(run / RUN_FILE, record)
    return EXIT_ERROR if failed else EXIT_OK


# ---------------------------------------------------------------------------------------------------------
# build


def _seconds_since(start: float) -> float:
    return round(time.monotonic() - start, 1)


def cmd_build(args: argparse.Namespace) -> int:
    """Compile the tag; its outcome goes to ``build.json``. Exits 0 when the build failed.

    A failed build must not stop the measure job chained after it: that job still re-measures the rules.
    Only a ``build.json`` that cannot be written is an error.
    """
    run = Path(args.run)
    started = time.monotonic()
    result = BuildResult(tag=args.commit, ok=False)
    try:
        build = compile_commit(llama_home(args.home), args.commit, Path(args.reference).expanduser())
        result = result.model_copy(update={"ok": server_binary(build).is_file(), "commit": build.name})
        if not result.ok:
            result.error = f"{server_binary(build)} was not built"
    except (subprocess.SubprocessError, OSError) as exc:
        result.error = str(exc)
    result.seconds = _seconds_since(started)
    try:
        write_model(run / BUILD_FILE, result)
    except OSError as exc:
        sys.stderr.write(f"cannot write {run / BUILD_FILE}: {exc}\n")
        return EXIT_ERROR
    sys.stdout.write(f"build {'ok' if result.ok else 'failed: ' + result.error}\n")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------------------
# measure: parsing and the verdict (pure)


def parse_bench(text: str, prefix: str) -> dict[str, float]:
    """``{<prefix>pp512: tokens/s, ..., <prefix>tg128: ...}`` from llama-bench's JSON rows.

    Raises:
        ValueError: the text is no JSON list of rows, or holds no throughput row.
    """
    figures: dict[str, float] = {}
    for row in json.loads(text):
        n_prompt, n_gen, speed = row.get("n_prompt", 0), row.get("n_gen", 0), row.get("avg_ts")
        if speed is None:
            continue
        if n_prompt and not n_gen:
            figures[f"{prefix}pp{n_prompt}"] = float(speed)
        elif n_gen and not n_prompt:
            figures[f"{prefix}tg{n_gen}"] = float(speed)
    if not figures:
        msg = "llama-bench printed no throughput row"
        raise ValueError(msg)
    return figures


def parse_batched(text: str, prefix: str) -> dict[str, float]:
    """``{<prefix>pp_at_1: tokens/s, <prefix>tg_at_1: ...}`` from llama-batched-bench's jsonl rows.

    Raises:
        ValueError: a line is not JSON, or no row has the fields.
    """
    figures: dict[str, float] = {}
    for line in filter(None, (raw.strip() for raw in text.splitlines())):
        row = json.loads(line)
        if {"pl", "speed_pp", "speed_tg"} <= row.keys():
            figures[f"{prefix}pp_at_{row['pl']}"] = float(row["speed_pp"])
            figures[f"{prefix}tg_at_{row['pl']}"] = float(row["speed_tg"])
    if not figures:
        msg = "llama-batched-bench printed no speed row"
        raise ValueError(msg)
    return figures


class SpeedJudgement(NamedTuple):
    """The speed gate's answer: what failed, the largest relative slowdown, and every figure's rounds."""

    failures: list[str]
    slowdown: float
    figures: dict[str, dict[str, list[float]]]


def judge_speed(
    names: Sequence[str],
    baseline: Sequence[Mapping[str, float | None]],
    candidate: Sequence[Mapping[str, float | None]],
    *,
    report_missing: bool = True,
) -> SpeedJudgement:
    """D13's speed rule on the means of the rounds, shared by the llama.cpp gate and the container gate.

    A figure fails when ``(mean(base) - mean(cand)) / mean(base) > max(MIN_SLOWDOWN, (max(base) - min(base))
    / min(base))`` (a NaN fails); one missing (or ``None``) in any round of either side fails as "not
    measured: <name>" unless ``report_missing`` is off (the caller already reports the steps that failed).
    """
    failures: list[str] = []
    figures: dict[str, dict[str, list[float]]] = {}
    slowdowns: list[float] = []
    for name in names:
        base = [v for r in baseline if (v := r.get(name)) is not None]
        cand = [v for r in candidate if (v := r.get(name)) is not None]
        if len(base) < ROUNDS or len(cand) < ROUNDS:
            if report_missing:
                failures.append(f"not measured: {name}")
            continue
        figures[name] = {"baseline": base, "candidate": cand}
        mean = statistics.fmean(base)
        slower = (mean - statistics.fmean(cand)) / mean
        slowdowns.append(slower)
        tolerance = max(MIN_SLOWDOWN, (max(base) - min(base)) / min(base))
        if math.isnan(slower) or slower > tolerance:  # a NaN figure is no measurement
            failures.append(f"{name} {_percent(slower)} slower than baseline (limit {_percent(tolerance)})")
    return SpeedJudgement(failures, max(slowdowns, default=0.0), figures)


def judge_candidate(
    baseline: Sequence[Mapping[str, float]],
    candidate: Sequence[Mapping[str, float]],
    control: KlStats | None,
    scored: KlStats | None,
    skipped: Mapping[str, str],
) -> Verdict:
    """D13: pass when no speed figure is slower beyond tolerance, K3 is within limits and all was measured.

    ``baseline`` and ``candidate`` hold one dict of figures per round. A figure is a slowdown failure when
    ``(baseline - candidate) / baseline > max(MIN_SLOWDOWN, baseline spread)`` on the means of the rounds, the
    spread being ``(max - min) / min`` of the baseline's rounds. ``control`` and ``scored`` are the KL
    figures of the current and the candidate build against the current's base. ``skipped`` maps each
    comparison step that did not measure to why; any entry fails the verdict (``cmp-k3`` steps: quality).
    """
    from .llama_tune import UB_LIMITS, judge_ub  # noqa: PLC0415 - heavy; only a measurement needs it

    speed: list[str] = []
    quality: list[str] = []
    for step, why in skipped.items():
        (quality if step.startswith("cmp-k3") else speed).append(f"not measured: {step}: {why}")
    judged = judge_speed(SPEED_FIGURES, baseline, candidate, report_missing=not skipped)
    speed += judged.failures
    figures = judged.figures
    if control is None or scored is None:
        if not any(step.startswith("cmp-k3") for step in skipped):
            quality.append("not measured: K3 comparison")
    else:
        # D18: the limits calibrated on this pinned text. K3's scoring limits (98 % same top) came from
        # October's GPL text; on this one they refused October's accepted switch too (97.63 %, slurm 2159).
        agreement = judge_ub(scored, control, UB_LIMITS)
        if not agreement.passed:
            quality.append(f"K3 outside its limits: {agreement.detail}")
    return Verdict(
        passed=not speed and not quality,
        speed_failures=speed,
        quality_failures=quality,
        slowdown=judged.slowdown,
        figures=figures,
        notes=", ".join(
            f"{name} {statistics.fmean(f['baseline']):.4g} -> {statistics.fmean(f['candidate']):.4g}"
            for name, f in figures.items()
        ),
    )


# ---------------------------------------------------------------------------------------------------------
# measure: running the tools


class SpeedRecipe(NamedTuple):
    """llama-bench and llama-batched-bench on one model: their arguments and the figures each must print."""

    model: Model
    prefix: str  # of every figure's name: "small_" or "prod_"
    bench: tuple[str, ...]
    batched: tuple[str, ...]
    bench_figures: tuple[str, ...]
    batched_figures: tuple[str, ...]


_SLOTS: Final = ",".join(map(str, SLOT_COUNTS))
#: The comparison's figures on the small model (design 2.3 step 4, as cu_compare.sh).
SMALL_SPEED: Final = SpeedRecipe(
    SMALL_MODEL,
    "small_",
    ("-p", "512,2048", "-n", "128", "-fa", "1", "-r", "3", "-o", "json"),
    ("-fa", "on", "-npp", "512", "-ntg", "32", "-npl", _SLOTS, "--output-format", "jsonl"),
    _BENCH_NAMES,
    _BATCHED_NAMES,
)
#: The production model's figures on the final current (design 2.3 step 7, as remeasure_k123.sh).
PROD_SPEED: Final = SpeedRecipe(
    PRODUCTION_MODEL,
    "prod_",
    ("-p", "512", "-n", "128", "-fa", "1", "-r", "3", "-o", "json"),
    ("-fa", "on", "-c", "8192", "-npp", "256", "-ntg", "128", "-npl", _SLOTS, "--output-format", "jsonl"),
    ("prod_pp512", "prod_tg128"),
    tuple(f"prod_tg_at_{slots}" for slots in SLOT_COUNTS),
)


class Measurer:
    """Runs the llama.cpp tools of one measurement and records each step's outcome.

    A step that fails (timeout, non-zero exit, output that does not parse, a missing model) is recorded in
    ``steps`` as ``not measured: <why>``, never raised: the other steps continue.
    """

    def __init__(self, run: Path, home: Path, *, timeout_s: float = TOOL_TIMEOUT_S) -> None:
        self.run = run
        self.home = home
        self.timeout_s = timeout_s
        self.steps: dict[str, str] = {}
        self.base_ready = False

    @property
    def kld(self) -> Path:
        """The teacher-forced base file: about 5 GB, deleted when the measurement ends."""
        return self.run / KL_BASE_FILE

    @property
    def failed(self) -> dict[str, str]:
        """Each step that did not measure, and why."""
        return {
            step: outcome.removeprefix(NOT_MEASURED)
            for step, outcome in self.steps.items()
            if outcome != STEP_OK
        }

    def _fail(self, step: str, why: str) -> None:
        self.steps[step] = f"{NOT_MEASURED}{why}"

    def model(self, pin: Model) -> Path | None:
        """The model file in the Hub cache; None when it is not there."""
        return model_file(pin)

    def tool(self, step: str, argv: Sequence[str | Path]) -> tuple[str, str] | None:
        """Run one tool call: (stdout, stderr), also kept in ``<step>.out`` and ``.err``; None on failure."""
        try:
            done = subprocess.run(  # noqa: S603
                [str(part) for part in argv],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._fail(step, f"timed out after {self.timeout_s:g} s")
            return None
        except OSError as exc:
            self._fail(step, f"could not run {argv[0]}: {exc}")
            return None
        (self.run / f"{step}.out").write_text(done.stdout, encoding="utf-8")
        (self.run / f"{step}.err").write_text(done.stderr, encoding="utf-8")
        if done.returncode:
            self._fail(step, f"exited {done.returncode}")
            return None
        return done.stdout, done.stderr

    def figures(
        self,
        step: str,
        argv: Sequence[str | Path],
        parse: Callable[[str, str], dict[str, float]],
        prefix: str,
        required: Sequence[str],
    ) -> dict[str, float]:
        """The figures one tool call printed; empty when it failed or lacked a required one."""
        output = self.tool(step, argv)
        if output is None:
            return {}
        try:  # a tool's progress lines sit on stderr: only stdout holds its rows
            found = parse(output[0], prefix)
        except ValueError as exc:  # json.JSONDecodeError is one
            self._fail(step, f"output did not parse: {exc}")
            return {}
        if missing := [name for name in required if name not in found]:
            self._fail(step, f"output lacks {', '.join(missing)}")
            return {}
        self.steps[step] = STEP_OK
        return found

    def speed(self, step: str, bin_dir: Path, recipe: SpeedRecipe) -> dict[str, float]:
        """llama-bench, then llama-batched-bench, on the build whose programs are in ``bin_dir``."""
        bench, batched = f"{step}-bench", f"{step}-batched"
        model = self.model(recipe.model)
        if model is None:
            for name in (bench, batched):
                self._fail(name, f"{recipe.model.filename} absent")
            return {}
        found = self.figures(
            bench, [bin_dir / Tool.BENCH, "-m", model, *recipe.bench], parse_bench, recipe.prefix,
            recipe.bench_figures,
        )  # fmt: skip
        found |= self.figures(
            batched, [bin_dir / Tool.BATCHED, "-m", model, *recipe.batched], parse_batched, recipe.prefix,
            recipe.batched_figures,
        )  # fmt: skip
        return found

    def k3_base(self, bin_dir: Path, step: str) -> None:
        """Save the build's token probabilities on the pinned text as the base the next runs compare with."""
        from .llama_tune import PINNED_TEXT, PINNED_TEXT_SHA256  # noqa: PLC0415 - heavy

        self.base_ready = False
        model = self.model(SMALL_MODEL)
        if model is None:
            self._fail(step, f"{SMALL_MODEL.filename} absent")
            return
        if sha256_file(PINNED_TEXT) != PINNED_TEXT_SHA256:
            self._fail(step, f"{PINNED_TEXT.name} does not match its pinned SHA-256")
            return
        argv: list[str | Path] = [
            bin_dir / Tool.PERPLEXITY,
            "-m",
            model,
            "-f",
            PINNED_TEXT,
            "-c",
            str(KL_CONTEXT),
        ]
        argv += ["--chunks", str(KL_CHUNKS), "--kl-divergence-base", self.kld]
        if self.tool(step, argv) is None:
            return
        if not self.kld.is_file():
            self._fail(step, f"no {KL_BASE_FILE} was written")
            return
        self.steps[step] = STEP_OK
        self.base_ready = True

    def k3_against(self, bin_dir: Path, step: str) -> KlStats | None:
        """The build's KL divergence statistics against the saved base; None when not measured."""
        from .llama_tune import parse_kl  # noqa: PLC0415 - heavy

        model = self.model(SMALL_MODEL)
        if model is None:
            self._fail(step, f"{SMALL_MODEL.filename} absent")
            return None
        if not self.base_ready:
            self._fail(step, "no base to compare with")
            return None
        argv: list[str | Path] = [bin_dir / Tool.PERPLEXITY, "-m", model, "-c", str(KL_CONTEXT)]
        argv += ["--kl-divergence-base", self.kld, "--kl-divergence"]
        output = self.tool(step, argv)
        if output is None:
            return None
        try:  # llama-perplexity prints its statistics on stdout (LOG) and its progress on stderr: read both
            stats = parse_kl("\n".join(output))
        except ValueError as exc:
            self._fail(step, str(exc))
            return None
        self.steps[step] = STEP_OK
        return stats


def _means(rounds: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Each figure's mean over the rounds that measured it, for the figures every round measured."""
    return {
        name: statistics.fmean(r[name] for r in rounds)
        for name in (rounds[0] if rounds else {})
        if all(name in r for r in rounds)
    }


def _record(
    kind: WorkloadKind,
    result: MeasureResult,
    run: Path,
    stamp: Mapping[str, str],
    figures: dict[str, float],
    holds: bool,
    detail: str,
) -> None:
    rules.write_record(
        rules.RuleRecord(
            kind=kind.value,
            measured_at=result.measured_at,
            slurm_job=result.slurm_job,
            run_dir=str(run),
            stamp=dict(stamp),
            figures=figures,
            holds=holds,
            detail=detail,
        )
    )
    result.records.append(kind.value)


def write_records(
    result: MeasureResult,
    run: Path,
    stamp: Mapping[str, str],
    small: Mapping[str, float],
    prod: Mapping[str, float],
    repeat: KlStats | None,
) -> None:
    """The rule records of K1, K2 and K3 (design 3), written even when a figure is missing (then it fails)."""
    model = {k: s.get(rules.StampKey.MODEL, "unknown") for k, s in rules.stamps(rules.rules_text())}
    figures = {**small, **prod}

    def stamped(kind: WorkloadKind) -> dict[str, str]:
        return {**stamp, rules.StampKey.MODEL.value: model.get(kind.value, "unknown")}

    def pick(names: Sequence[str]) -> tuple[dict[str, float], str]:
        found = {n: figures[n] for n in names if n in figures}
        missing = [n for n in names if n not in figures]
        return found, f"not measured: {', '.join(missing)}" if missing else "all measured"

    k1, detail = pick(K1_FIGURES)
    _record(WorkloadKind.K1, result, run, stamped(WorkloadKind.K1), k1, len(k1) == len(K1_FIGURES), detail)

    k2, detail = pick([f"{m}_tg_at_{n}" for m in ("small", "prod") for n in SLOT_COUNTS])
    holds = len(k2) == len(SLOT_COUNTS) * 2
    if "prod_tg_at_16" in k2 and "prod_tg_at_4" in k2:
        k2["prod_gain_16_over_4"] = k2["prod_tg_at_16"] / k2["prod_tg_at_4"]
        holds = holds and k2["prod_gain_16_over_4"] > 1
        detail = f"{detail}; 16 slots write {k2['prod_gain_16_over_4']:.2f} times what 4 do"
    _record(WorkloadKind.K2, result, run, stamped(WorkloadKind.K2), k2, holds, detail)

    k3 = repeat.model_dump() if repeat is not None else {}
    holds = repeat is not None and repeat.mean_kl <= K3_MAX_KL and repeat.same_top >= K3_MIN_TOP
    detail = (
        "not measured: the repeat run"
        if repeat is None
        else f"repeat mean KL {repeat.mean_kl:.2e} (at most {K3_MAX_KL:g}), "
        f"same top token {repeat.same_top:.2%} (at least {K3_MIN_TOP:.1%})"
    )
    _record(WorkloadKind.K3, result, run, stamped(WorkloadKind.K3), k3, holds, detail)


def _bin(home: Path, arm: Arm, name: str | None) -> Path:
    """The programs of an arm: ``current``'s (the reference build when there is no link) or a build's."""
    return (
        current_server(home).parent if arm is Arm.CURRENT or name is None else builds_dir(home) / name / "bin"
    )


def _compare(
    run: Path,
    home: Path,
    result: MeasureResult,
    m: Measurer,
    tag: str,
    candidate: str,
    rounds: Mapping[Arm, Sequence[Mapping[str, float]]],
) -> KlStats | None:
    """K3 against current's logits, D13's verdict, the switch on a pass and the state; current's own repeat.

    Fills ``result.comparison``, ``switched`` and ``switch_problem``; writes ``comparison.json`` into the
    candidate's build folder and the run folder.
    """
    current_bin = _bin(home, Arm.CURRENT, None)
    m.k3_base(current_bin, f"{COMPARISON_STEP}k3-base")
    control = m.k3_against(current_bin, f"{COMPARISON_STEP}k3-control")
    scored = m.k3_against(_bin(home, Arm.CANDIDATE, candidate), f"{COMPARISON_STEP}k3-candidate")
    skipped = {step: why for step, why in m.failed.items() if step.startswith(COMPARISON_STEP)}
    verdict = judge_candidate(rounds[Arm.CURRENT], rounds[Arm.CANDIDATE], control, scored, skipped)
    before = current_commit(home)
    comparison = Comparison(
        **verdict.model_dump(),
        baseline_commit=before or "reference",
        measured_at=result.measured_at,
        candidate_tag=tag,
        slurm_job=result.slurm_job,
        kl={
            "control": control.model_dump() if control else None,
            "candidate": scored.model_dump() if scored else None,
        },
    )
    result.comparison = comparison
    write_model(builds_dir(home) / candidate / COMPARISON_FILE, comparison)
    write_model(run / COMPARISON_FILE, comparison)
    state = read_state()
    state.tried[candidate] = Tried(passed=comparison.passed, run=run.name)
    if comparison.passed:
        kept, state.previous_current = state.previous_current, before
        write_state(state)  # before the link moves: a run stopped in between still knows the rollback build
        problem = switch(home, candidate)
        result.switched, result.switch_problem = problem is None, problem or ""
        if not result.switched:
            state.previous_current = kept
    write_state(state)
    return control


def run_measurement(run: Path, home: Path, result: MeasureResult, m: Measurer) -> None:
    """The measurement with the card already quiet (design 2.3, steps 3-9); fills ``result``."""
    build = read_model(run / BUILD_FILE, BuildResult)
    candidate = build.commit if build is not None and build.ok else None
    if candidate is not None and (
        candidate == current_commit(home) or not server_binary(builds_dir(home) / candidate).is_file()
    ):
        candidate = None
    arms = (Arm.CURRENT, Arm.CANDIDATE) if candidate else (Arm.CURRENT,)
    prefix = COMPARISON_STEP if candidate else ""
    rounds: dict[Arm, list[dict[str, float]]] = {Arm.CURRENT: [], Arm.CANDIDATE: []}
    for number in range(1, ROUNDS + 1):  # the arms interleaved, as cu_compare.sh
        for arm in arms:
            rounds[arm].append(m.speed(f"{prefix}r{number}-{arm}", _bin(home, arm, candidate), SMALL_SPEED))

    repeat = None
    if candidate is not None and build is not None:  # build: for the type checker, it is set with candidate
        repeat = _compare(run, home, result, m, build.tag, candidate, rounds)
    final_bin = _bin(home, Arm.CURRENT, None)  # the link, moved or not
    result.final_current = current_commit(home) or "reference"
    if result.switched or candidate is None:  # current's control run is no repeat of the new build
        m.k3_base(final_bin, "final-k3-base")
        repeat = m.k3_against(final_bin, "final-k3-repeat")
    prod = m.speed("prod", final_bin, PROD_SPEED)
    small = _means(rounds[Arm.CANDIDATE if result.switched else Arm.CURRENT])
    write_records(result, run, rules.host_stamp(current_server(home)), small, prod, repeat)


def cmd_measure(args: argparse.Namespace) -> int:
    """Hold the card quiet and run the comparison and the K1-K3 re-measurement; ``result.json`` says what ran.

    Exit 1 only when nothing was measured (the card did not go quiet); a candidate that fails its comparison
    is a result, not an error. SIGTERM (``timeout``) becomes ``SystemExit``, so the 5 GB base file is deleted.
    """
    run, home = Path(args.run), llama_home(args.home)
    run.mkdir(parents=True, exist_ok=True)
    result = MeasureResult(
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        slurm_job=os.environ.get("SLURM_JOB_ID", ""),
    )
    measurer = Measurer(run, home)

    def _terminate(_signum: int, _frame: object) -> None:
        raise SystemExit(SIGTERM_EXIT)

    previous = signal.signal(signal.SIGTERM, _terminate)
    code = EXIT_OK
    try:
        card = QuietCard(GpuLease(DEFAULT_LEASE_DIR, LEASE_PROJECT), reason="llama.cpp refresh measurement")
        with card:
            run_measurement(run, home, result, measurer)
    except QuietCardTimeoutError as exc:
        result.not_measured = str(exc)
        code = EXIT_ERROR
    except BaseException as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        measurer.kld.unlink(missing_ok=True)
        result.steps = measurer.steps
        write_model(run / RESULT_FILE, result)
    sys.stdout.write(headline(RunView(run, None, read_model(run / BUILD_FILE, BuildResult), result)) + "\n")
    return code


# ---------------------------------------------------------------------------------------------------------
# status and registration


def cmd_status(_args: argparse.Namespace) -> int:
    """Print the newest run: what was decided, submitted and measured."""
    view = newest_view()
    if view is None:
        sys.stdout.write(f"no refresh run under {refresh_dir()}\n")
        return EXIT_OK
    out = sys.stdout
    out.write(f"{headline(view)}\n{view.folder}\n")
    if view.run is not None:
        out.writelines(
            f"  {job.name}: {'job ' + job.job if job.job else 'not submitted: ' + job.error}\n"
            for job in view.run.jobs
        )
        out.writelines(f"  removed build {removed}\n" for removed in view.run.removed_builds)
        out.writelines(f"  {line}\n" for line in view.run.stale_rules)
        if view.run.engines or view.run.engine_reports:
            from .engine_refresh import engine_lines  # noqa: PLC0415 - loaded only when there is a walk to show

            out.writelines(f"  {line}\n" for line in engine_lines(view))
    if view.result is not None:
        out.writelines(
            f"  {step}: {outcome}\n" for step, outcome in view.result.steps.items() if outcome != STEP_OK
        )
        if line := rules_line(view):
            out.write(f"  {line}\n")
    return EXIT_OK


#: Each engine job's own flags as (flag, required); every one also takes ``--run``.
JOB_FLAGS: Final = {
    "engine-pull": (("--engine", True), ("--tag", True), ("--publisher", True)),
    "engine-measure": (("--engine", True), ("--tag", True), ("--mode", True), ("--model", False)),
    "engine-calibrate": (("--engine", True), ("--tag", True), ("--adjacent", True)),
    "same-client": (),
    "sparkrun-compare": (("--old", True), ("--new", True)),
}


def _engine_job(function: str) -> Callable[[argparse.Namespace], int]:
    """The command that runs ``engine_refresh.cmd_<function>``, loading that module only when it runs."""

    def run(args: argparse.Namespace) -> int:
        from . import engine_refresh  # noqa: PLC0415 - the container engines are loaded only when used

        return int(getattr(engine_refresh, f"cmd_{function}")(args))

    return run


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``refresh run|build|measure|status`` to the ``localmodels`` command."""
    refresh = subparsers.add_parser("refresh", help="monthly llama.cpp refresh: compare, switch, re-measure")
    refresh.add_argument("--home", help="llama home (see `localmodels llama`)")
    commands = refresh.add_subparsers(dest="refresh_command", required=True)

    run = commands.add_parser("run", help="decide what is due and submit the build and measure jobs")
    run.add_argument("--dry-run", action="store_true", help="print the submit commands and write nothing")
    run.add_argument("--retry", action="store_true", help="compare the newest release again although tried")
    run.add_argument(
        "--force-measure", action="store_true", help="submit the measure job although nothing is due"
    )
    run.add_argument(
        "--candidate", metavar="TAG", help="compare this release tag (b<number>), not the newest"
    )
    run.add_argument(
        "--urgent",
        action="store_true",
        help="submit the measure job through slurm's urgent.py (pauses what can be paused in its way); "
        "the candidate must be built already",
    )
    run.set_defaults(fn=cmd_run)

    build = commands.add_parser("build", help="(a job) compile the release tag and write build.json")
    build.add_argument("--run", required=True, help="the run folder")
    build.add_argument("--commit", required=True, help="the release tag to build")
    build.add_argument(
        "--reference", default=str(REFERENCE_BUILD), help="build whose CMakeCache.txt to reproduce"
    )
    build.set_defaults(fn=cmd_build)

    measure = commands.add_parser(
        "measure", help="(a job) compare the candidate, switch on a pass, re-measure K1-K3"
    )
    measure.add_argument("--run", required=True, help="the run folder")
    measure.set_defaults(fn=cmd_measure)

    commands.add_parser("status", help="print the newest run").set_defaults(fn=cmd_status)

    jobs = {
        "engine-pull": ("(a job) pull one container engine image and realize its recipes", "pull"),
        "engine-measure": ("(a job) measure a container engine version; switch on a pass", "measure"),
        "engine-calibrate": ("(a job) calibrate a container engine's quality limits", "calibrate"),
        "same-client": ("(a job) measure llama.cpp on the container engines' client", "same_client"),
        "sparkrun-compare": ("(a job) compare two sparkrun versions on every engine", "sparkrun_compare"),
    }
    for command, (help_, function) in jobs.items():
        job = commands.add_parser(command, help=help_)
        job.add_argument("--run", required=True, help="the run folder")
        for flag, required in JOB_FLAGS[command]:
            job.add_argument(flag, required=required)
        job.set_defaults(fn=_engine_job(function))
