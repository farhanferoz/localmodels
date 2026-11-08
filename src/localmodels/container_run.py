"""The one way localmodels starts a container engine (vLLM, SGLang, TensorRT-LLM): sparkrun, guarded.

Design: docs/research/engine-registry-design-2026-11-08.md section 3 (D23). ``ContainerRun`` is a context
manager around one served engine and model:

1. Whole memory need (D22 for containers): the larger of the measured fall in available memory and the GPU
   plus host-side peaks a run of that engine and model recorded; a D16 seed only for a bootstrap or refresh
   run. The run waits for memory to settle, then refuses a start whose need plus the host reserve exceeds
   available memory.
2. Start: ``sparkrun run`` offline, telemetry off, labels ``lifecycle=ephemeral``, ``slurm-job=<id>``
   (preflight's job wrapper counts the container's GPU memory) and ``localmodels.run=<uuid>``; the model's
   Hub folder mounted read-only behind a per-run Hub root; the recipe validated (no hooks, no ``include``).
3. Guard: ``launcher.kill_below_floor`` with a ``docker kill`` action; the post-load host-reserve check; the
   container's processes get the guards' OOM score.
4. Stop on every exit it can catch (normal, exception, SIGTERM, chained to the caller's handler); the
   ``Reaper`` (``localmodels engines reap``) removes what a SIGKILL left.

``import localmodels`` does not load this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import logging
import os
import shutil
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

from .engines import VersionRecord, docker_runner, read_record, run_command, write_record
from .launcher import MemoryGuards, NotEnoughMemoryError, ServerStartError, check_fits, kill_below_floor
from .stack import Engine

if TYPE_CHECKING:
    import subprocess
    import types
    from collections.abc import Callable, Mapping, Sequence

    from .engines import Runner

log = logging.getLogger(__name__)

GIB: Final = 1024**3
TELEMETRY_ENV: Final = {"SPARKRUN_NO_TELEMETRY": "1"}
SPARKRUN_REPO: Final = "https://github.com/spark-arena/sparkrun"
SPARKRUN_ROOT: Final = Path.home() / "cache" / "localmodels" / "sparkrun"
HUB_ROOT: Final = Path.home() / ".cache" / "huggingface"
RUN_LABEL: Final = "localmodels.run"
JOB_LABEL: Final = "slurm-job"  # preflight's gpu_jobs.JOB_LABEL
LIFECYCLE_LABEL: Final = "lifecycle=ephemeral"
#: Recipe keys that run commands on this host (sparkrun trusts a local recipe) or pull in another recipe.
RECIPE_HOOKS: Final = frozenset({"pre_exec", "mods", "post_exec", "post_commands", "include"})
#: D16's whole need for Qwen3-30B-A3B NVFP4, GiB (the larger of the fall in free memory and the GPU peak).
SEED_NEED_GIB: Final = {Engine.VLLM: 57.0, Engine.SGLANG: 36.1, Engine.TRTLLM: 37.6}
SETTLE_LIMIT_S: Final = 60.0
SETTLE_SLACK_GIB: Final = 1.0
POLL_S: Final = 1.0
START_TIMEOUT_S: Final = 900.0
STOP_TIMEOUT_S: Final = 120.0
SQUEUE_TIMEOUT_S: Final = 30.0


class RunKind(StrEnum):
    """Who starts the container: only a bootstrap or refresh run starts an engine and model never measured."""

    PROJECT = "project"
    BOOTSTRAP = "bootstrap"
    REFRESH = "refresh"


class ContainerStartError(ServerStartError):
    """A container engine was not started; the message names the remedy."""


def sparkrun_env(hub_root: Path | None = None) -> dict[str, str]:
    """The environment every sparkrun call gets: telemetry off, and the per-run Hub root when there is one."""
    env = {**os.environ, **TELEMETRY_ENV}
    if hub_root is not None:
        env["HF_HOME"] = str(hub_root)
    return env


def run_sparkrun(
    binary: Path, args: Sequence[str], *, timeout: float, hub_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """One sparkrun call with telemetry off (and the per-run Hub root); never raises."""
    return run_command([str(binary), *args], timeout=timeout, env=sparkrun_env(hub_root))


def ensure_sparkrun(tag: str, *, root: Path = SPARKRUN_ROOT, runner: Runner = docker_runner) -> Path:
    """The sparkrun binary of ``tag``, installed from its git tag on first use; telemetry off."""
    home = root / tag
    binary = home / "venv" / "bin" / "sparkrun"
    if binary.is_file():
        return binary
    steps = (
        ["git", "clone", "--depth", "1", "--branch", tag, SPARKRUN_REPO, str(home / "src")],
        ["uv", "venv", str(home / "venv")],
        ["uv", "pip", "install", "--python", str(home / "venv" / "bin" / "python"), str(home / "src")],
    )
    for step in steps:
        done = runner(step)
        if done.returncode != 0:
            msg = (
                f"installing sparkrun {tag}: `{' '.join(step)}` exited "
                f"{done.returncode}: {done.stderr.strip()}"
            )
            raise ContainerStartError(msg)
    run_sparkrun(binary, ["setup", "telemetry", "--disable"], timeout=STOP_TIMEOUT_S)
    return binary


def recipe_problems(recipe: Mapping[str, Any]) -> list[str]:
    """Why ``recipe`` may not run here: each hook or include it names (unreviewed commands)."""
    return [
        f"recipe key {key!r} runs or pulls in commands; recipes here carry none"
        for key in sorted(RECIPE_HOOKS & set(recipe))
    ]


@dataclass(frozen=True)
class MemoryNeed:
    """A run's memory, GiB: the fall in available memory, and the GPU and host-side peaks it measured."""

    fall_gib: float | None = None
    gpu_peak_gib: float | None = None
    host_peak_gib: float | None = None

    @property
    def gib(self) -> float | None:
        """The larger of the fall and GPU + host peaks; None when nothing was measured."""
        parts = [self.fall_gib, (self.gpu_peak_gib or 0.0) + (self.host_peak_gib or 0.0) or None]
        known = [p for p in parts if p is not None]
        return max(known) if known else None

    @classmethod
    def recorded(cls, record: VersionRecord, model: str) -> float | None:
        """The largest whole need any run of ``model`` on this version recorded (results' ``memory``)."""
        needs = [
            cls(**dict(result["memory"])).gib  # type: ignore[call-overload]
            for result in record.results
            if result.get("model") == model and isinstance(result.get("memory"), dict)
        ]
        known = [n for n in needs if n is not None]
        return max(known) if known else None


def _available_gib() -> float | None:
    from .hardware import available_memory  # noqa: PLC0415 - hardware reads /proc only when asked

    reading = available_memory("cuda")
    return None if reading is None else reading.bytes / GIB


@dataclass
class ContainerRun:
    """One container engine serving one model, inside the guards; use as ``with ContainerRun(...) as run``."""

    record: VersionRecord
    model: str  # the Hub repository, e.g. ``nvidia/Qwen3-30B-A3B-NVFP4``
    recipe: dict[str, Any]
    sparkrun: Path
    guards: MemoryGuards
    kind: RunKind = RunKind.PROJECT
    hub_root: Path = HUB_ROOT
    work_dir: Path = field(default_factory=lambda: Path(os.environ.get("TMPDIR", "/tmp")))  # noqa: S108
    docker: Runner = docker_runner
    free_gib: Callable[[], float | None] = _available_gib
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reaper: Reaper | None = None  # default: a ``Reaper`` on ``docker`` and the real ``squeue``
    _recipe_file: Path | None = field(default=None, init=False)
    _run_root: Path | None = field(default=None, init=False)
    _stopped: threading.Event = field(default_factory=threading.Event, init=False)
    _guard: threading.Thread | None = field(default=None, init=False)
    _killed: str | None = field(default=None, init=False)
    _previous_handler: Any = field(default=None, init=False)
    _free_before: float | None = field(default=None, init=False)
    _lowest_free: float | None = field(default=None, init=False)

    # --- before the start -------------------------------------------------------------------------

    def need_gib(self) -> float:
        """The need this start is checked against; a project never starts an unmeasured engine and model."""
        recorded = MemoryNeed.recorded(self.record, self.model)
        if recorded is not None:
            return recorded
        if self.kind is RunKind.PROJECT:
            msg = (
                f"{self.record.engine.value} {self.record.tag} has no measured run of {self.model}: a "
                "project's job does not start it (the chooser answers llama.cpp); a bootstrap or refresh job "
                "measures it first"
            )
            raise ContainerStartError(msg)
        return SEED_NEED_GIB[self.record.engine]

    def _refuse_unready(self) -> None:
        if not self.record.image or not self.record.pulled:
            msg = (
                f"{self.record.engine.value} {self.record.tag}: its image is not on disk; pull it first with "
                f"`localmodels engines pull {self.record.engine.value} {self.record.tag}` (a CPU job), never "
                "inside a "
                "project's job"
            )
            raise ContainerStartError(msg)
        problems = recipe_problems(self.recipe)
        if problems:
            raise ContainerStartError("; ".join(problems))

    def _settle(self) -> float | None:
        """Wait up to ``SETTLE_LIMIT_S`` for available memory to stop rising (it comes back late)."""
        deadline, last = time.monotonic() + SETTLE_LIMIT_S, self.free_gib()
        while time.monotonic() < deadline:
            time.sleep(POLL_S)
            now = self.free_gib()
            if now is None or last is None or now - last < SETTLE_SLACK_GIB / 10:
                return now
            last = now
        return last

    def _prepare_hub(self) -> Path:
        """A per-run Hub root whose one model links to the real ``models--<org>--<name>`` folder."""
        real = self.hub_root / "hub" / f"models--{self.model.replace('/', '--')}"
        if not real.is_dir():
            msg = f"{self.model} is not in the Hub cache ({real}); download it with `localmodels pull` first"
            raise ContainerStartError(msg)
        root = self.work_dir / f"localmodels-hub-{self.run_id}"
        (root / "hub").mkdir(parents=True)
        (root / "hub" / real.name).symlink_to(real)
        self._run_root = root
        return real

    def _write_recipe(self, real_model_dir: Path) -> Path:
        """The recipe sparkrun runs: ours, the read-only model mount and the run's labels (JSON is YAML)."""
        executor = dict(self.recipe.get("executor_config") or {})
        executor["volumes"] = [*executor.get("volumes", []), f"{real_model_dir}:{real_model_dir}:ro"]
        job = os.environ.get("SLURM_JOB_ID", "none")
        executor["labels"] = [LIFECYCLE_LABEL, f"{JOB_LABEL}={job}", f"{RUN_LABEL}={self.run_id}"]
        body = {
            **self.recipe,
            "model": self.model,
            "container": self.record.image,
            "executor_config": executor,
        }
        path = self.work_dir / f"localmodels-recipe-{self.run_id}.json"
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        self._recipe_file = path
        return path

    # --- the run ----------------------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Check, start and guard the server; it is ready when this returns."""
        (self.reaper or Reaper(docker=self.docker)).run()
        self._refuse_unready()
        need = self.need_gib()
        self._free_before = self._settle()
        reserve = self.guards.host_reserve_gib or 0.0
        check_fits(
            f"{self.record.engine.value} {self.record.tag} serving {self.model}",
            need,
            self._free_before,
            reserve,
        )
        recipe = self._write_recipe(self._prepare_hub())
        self._install_sigterm()
        self._start_guard()
        done = run_sparkrun(
            self.sparkrun, self._run_args(recipe), timeout=START_TIMEOUT_S, hub_root=self._run_root
        )
        if done.returncode != 0 or self._killed:
            self.stop()
            why = self._killed or f"sparkrun run exited {done.returncode}: {done.stderr.strip()[-500:]}"
            raise ContainerStartError(why)
        free = self.free_gib()
        if self.guards.host_reserve_gib is not None and (free is None or free < self.guards.host_reserve_gib):
            self.stop()
            reserve = self.guards.host_reserve_gib
            msg = f"the server left {free} GiB available, below the {reserve:g} GiB host reserve"
            raise NotEnoughMemoryError(msg)
        return self

    def __exit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None
    ) -> None:
        """Stop the server, whatever ended the block."""
        self.stop()

    def _run_args(self, recipe: Path) -> list[str]:
        return ["run", str(recipe), "--tp", "1", "--hosts", "localhost", "--no-follow", "--offline"]

    def _install_sigterm(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signal handlers are the main thread's; a worker thread's run relies on the reaper
        previous = signal.getsignal(signal.SIGTERM)
        self._previous_handler = previous

        def on_term(signum: int, frame: types.FrameType | None) -> None:
            self.stop()
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, on_term)

    def _start_guard(self) -> None:
        if self.guards.kill_below_gib is None:
            return
        floor = self.guards.kill_below_gib

        def watch() -> None:
            self._killed = asyncio.run(
                kill_below_floor(
                    0,
                    floor,
                    lambda: POLL_S,
                    self._sample_free,
                    f"{self.record.engine.value} container",
                    alive=lambda: not self._stopped.is_set(),
                    kill=self._docker_kill,
                    on_poll=self._raise_oom_score,
                )
            )

        self._guard = threading.Thread(target=watch, name=f"guard-{self.run_id[:8]}", daemon=True)
        self._guard.start()

    def _sample_free(self) -> float | None:
        free = self.free_gib()
        if free is not None:
            self._lowest_free = free if self._lowest_free is None else min(self._lowest_free, free)
        return free

    # --- the container ----------------------------------------------------------------------------

    def container_ids(self) -> list[str]:
        """This run's containers, by its own label."""
        done = self.docker(
            ["docker", "ps", "-aq", "--no-trunc", "--filter", f"label={RUN_LABEL}={self.run_id}"]
        )
        return done.stdout.split() if done.returncode == 0 else []

    def _docker_kill(self) -> None:
        for cid in self.container_ids():
            self.docker(["docker", "kill", cid])

    def _cgroup(self, cid: str) -> Path:
        return Path("/sys/fs/cgroup/system.slice") / f"docker-{cid}.scope"

    def _raise_oom_score(self) -> None:
        """Give every process of the container the guards' OOM score (raising it needs no privilege)."""
        score = self.guards.server_oom_score_adj
        if score is None:
            return
        for cid in self.container_ids():
            with contextlib.suppress(OSError):
                for pid in (self._cgroup(cid) / "cgroup.procs").read_text(encoding="utf-8").split():
                    with contextlib.suppress(OSError):
                        Path(f"/proc/{pid}/oom_score_adj").write_text(str(score), encoding="utf-8")

    def host_peak_gib(self) -> float | None:
        """The container's host-side memory peak (its cgroup's ``memory.peak``), GiB; None when unreadable."""
        peaks = []
        for cid in self.container_ids():
            with contextlib.suppress(OSError, ValueError):
                peaks.append(int((self._cgroup(cid) / "memory.peak").read_text(encoding="utf-8")) / GIB)
        return max(peaks) if peaks else None

    def memory(self) -> MemoryNeed:
        """What this run used so far: the fall in available memory since its start, and the host-side peak."""
        fall = None
        if self._free_before is not None and self._lowest_free is not None:
            fall = max(0.0, self._free_before - self._lowest_free)
        return MemoryNeed(fall_gib=fall, host_peak_gib=self.host_peak_gib())

    def stamp(self) -> dict[str, str]:
        """What a result served by this run carries (design 3.7; E4 adds the autotune and tuning hashes)."""
        return {
            "engine": self.record.engine.value,
            "image": self.record.image,
            "tag": self.record.tag,
            "publisher": self.record.publisher.value,
            "sparkrun_version": self.record.sparkrun_version,
            "model_file": self.model,
            "run": self.run_id,
        }

    def stop(self) -> None:
        """Stop the server and remove its containers, then log what was left; safe to call more than once."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        if self._recipe_file is not None:
            run_sparkrun(
                self.sparkrun,
                ["stop", str(self._recipe_file), "--tp", "1", "--hosts", "localhost"],
                timeout=STOP_TIMEOUT_S,
                hub_root=self._run_root,
            )
        for cid in self.container_ids():
            self.docker(["docker", "rm", "-f", cid])
        left = self.container_ids()
        log.info("container run %s stopped; containers left: %s", self.run_id, left or "none")
        if self._previous_handler is not None and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._previous_handler)
        if self._run_root is not None:
            shutil.rmtree(self._run_root, ignore_errors=True)
        if self._recipe_file is not None:
            self._recipe_file.unlink(missing_ok=True)


def record_memory(record: VersionRecord, model: str, run: ContainerRun, gpu_peak_gib: float | None) -> None:
    """Add a run's memory to the version's record: the next start's need is then measured, not seeded."""
    latest = read_record(record.engine, record.tag) or record  # a job may have added results meanwhile
    memory = run.memory()
    entry = {
        "model": model,
        "memory": {**dataclasses.asdict(memory), "gpu_peak_gib": gpu_peak_gib},
        "run": run.run_id,
    }
    write_record(latest.model_copy(update={"results": [*latest.results, entry]}))


# --- the reaper -----------------------------------------------------------------------------------


class JobState(StrEnum):
    """The slurm states the reaper reads; every other state counts as not running."""

    RUNNING = "RUNNING"


@dataclass(frozen=True)
class Container:
    """A labelled container: its id, the slurm job it belongs to, and when it was created."""

    cid: str
    job: str
    created: datetime


@dataclass
class ReapReport:
    """What one reap did: containers removed and kept (each with why), or why it removed nothing."""

    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class Reaper:
    """Remove the engine containers whose slurm job is no longer running (``localmodels engines reap``)."""

    docker: Runner = docker_runner
    squeue: Runner = field(default=functools.partial(run_command, timeout=SQUEUE_TIMEOUT_S))

    def jobs(self) -> dict[str, tuple[str, datetime | None]] | None:
        """Every job slurm holds: id -> (state, start); None when the listing failed."""
        done = self.squeue(["squeue", "-h", "-a", "-o", "%i|%T|%S"])
        if done.returncode != 0:
            return None
        jobs: dict[str, tuple[str, datetime | None]] = {}
        for line in done.stdout.splitlines():
            job, _, rest = line.partition("|")
            state, _, start = rest.partition("|")
            try:
                started = datetime.fromisoformat(start).astimezone(UTC)
            except ValueError:
                started = None
            jobs[job.strip()] = (state.strip(), started)
        return jobs

    def containers(self) -> list[Container]:
        """Every container carrying a ``slurm-job`` label."""
        done = self.docker(
            [
                "docker",
                "ps",
                "-a",
                "--no-trunc",
                "--filter",
                f"label={JOB_LABEL}",
                "--format",
                f'{{{{.ID}}}}|{{{{.Label "{JOB_LABEL}"}}}}|{{{{.CreatedAt}}}}',
            ]
        )
        found = []
        for line in done.stdout.splitlines() if done.returncode == 0 else []:
            cid, job, created = [*line.split("|"), "", ""][:3]
            with contextlib.suppress(ValueError):
                found.append(
                    Container(cid, job, datetime.strptime(created.rsplit(" ", 1)[0], "%Y-%m-%d %H:%M:%S %z"))
                )
        return found

    def run(self) -> ReapReport:
        """Remove each container whose job ended, is not running, or restarted after it; log each decision."""
        report = ReapReport()
        jobs = self.jobs()
        if jobs is None:
            report.reason = "squeue did not answer: nothing removed"
            log.warning("reaper: %s", report.reason)
            return report
        for box in self.containers():
            state, started = jobs.get(box.job, (None, None))
            why = None
            if state is None:
                why = f"job {box.job} has ended"
            elif state != JobState.RUNNING:
                why = f"job {box.job} is {state}"
            elif started is not None and started > box.created:
                why = f"job {box.job} started again after the container was created"
            if why is None:
                report.kept.append(f"{box.cid[:12]}: job {box.job} is running")
            else:
                self.docker(["docker", "rm", "-f", box.cid])
                report.removed.append(f"{box.cid[:12]}: {why}")
        for line in report.removed:
            log.info("reaper removed %s", line)
        for line in report.kept:
            log.info("reaper kept %s", line)
        return report


def cmd_reap(_args: object) -> int:
    """``localmodels engines reap``: remove the engine containers of jobs that are no longer running."""
    report = Reaper().run()
    for line in report.removed:
        print(f"removed {line}")  # noqa: T201 - the command's output
    for line in report.kept:
        print(f"kept {line}")  # noqa: T201
    if report.reason:
        print(report.reason)  # noqa: T201
        return 1
    return 0
