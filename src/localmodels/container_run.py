"""The one way localmodels starts a container engine (vLLM, SGLang, TensorRT-LLM): sparkrun, guarded.

Design: docs/research/engine-registry-design-2026-11-08.md section 3 (D23). ``ContainerRun`` is a context
manager around one served engine and model:

1. Whole memory need (D22 for containers): the larger of the measured fall in available memory and the GPU
   plus host-side peaks a run of that engine and model recorded; a D16 seed only for a bootstrap or refresh
   run. The run waits for memory to settle, then refuses a start whose need plus the host reserve exceeds
   available memory.
2. Start: ``sparkrun run`` offline, telemetry off, with the user's own loopback login (``loopback``: its
   steps that log in to this machine over SSH work unattended, whoever is logged in), labels
   ``lifecycle=ephemeral``, ``slurm-job=<id>`` (preflight's job wrapper counts the container's GPU memory) and
   ``localmodels.run=<uuid>``; the model's Hub folder mounted read-only behind a per-run Hub root; the recipe
   validated (no hooks, no ``include``).
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
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self

from . import loopback
from .engines import (
    CONTAINER_ENGINES,
    VersionRecord,
    docker_runner,
    read_record,
    run_command,
    write_record,
)
from .enums import ServingRole
from .hardware import gpu_process_memory_mib, inspect_hardware
from .launcher import (
    LocalServerManager,
    MemoryGuards,
    NotEnoughMemoryError,
    ServerStartError,
    check_fits,
    kill_below_floor,
)
from .params import derive_serving_params
from .process import ServerSpec
from .stack import Engine

if TYPE_CHECKING:
    import types
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    import httpx

    from .engines import Runner
    from .lease import GpuLease
    from .params import ServingParams
    from .settings import SettingsAnswer

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
MIB_PER_GIB: Final = 1024
#: sparkrun's own cache (``core/config.py`` ``DEFAULT_CACHE_DIR``): the runtime cache and the tuning folders.
SPARKRUN_CACHE: Final = Path.home() / ".cache" / "sparkrun"
RUNTIME_CACHE_SUBDIR: Final = (
    "runtime-cache"  # core/runtime_cache.py: ``<cache>/runtime-cache/<family>/<keys>``
)
LAST_USED_MARKER: Final = ".sparkrun-last-used"  # touched in the run's cache leaf at every launch
TUNING_SUBDIR: Final = "tuning"  # tuning/{sglang,vllm}.py: ``<cache>/tuning/<family>``
ABSENT: Final = "absent"
#: The recipe's own names for the stamp's memory fraction and maximum sequences, by sparkrun runtime.
MEMORY_FRACTION_KEYS: Final = ("gpu_memory_utilization", "free_gpu_memory_fraction")
MAX_SEQUENCES_KEYS: Final = ("max_num_seqs", "max_batch_size")


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
    binary: Path,
    args: Sequence[str],
    *,
    timeout: float,
    hub_root: Path | None = None,
    logs_in: bool = False,
) -> subprocess.CompletedProcess[str]:
    """One sparkrun call with telemetry off (and the per-run Hub root); never raises.

    ``logs_in``: a container's start, stop or dry run, whose steps sparkrun may send over ``ssh localhost``;
    it runs with the loopback login's ``ssh`` first on PATH, and a login that does not work is this call's
    failure, its reason in stderr. The recipe realize and the telemetry switch dispatch locally (checked
    2026-11-09: a realize with no agent and no allowed key succeeded), so they leave ``~/.ssh`` alone.
    """
    argv, env = [str(binary), *args], sparkrun_env(hub_root)
    if logs_in:
        try:
            env = loopback.loopback_login().env(env)
        except loopback.LoopbackLoginError as exc:
            return subprocess.CompletedProcess(argv, 1, "", str(exc))
    return run_command(argv, timeout=timeout, env=env)


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
        names = {f.name for f in dataclasses.fields(cls)}
        needs = [
            cls(**{k: v for k, v in result["memory"].items() if k in names}).gib  # type: ignore[attr-defined]
            for result in record.results
            if result.get("model") == model
            and isinstance(result.get("memory"), dict)
            and result.get("complete") is not False  # a guard-killed or partial run is no measurement
        ]
        known = [n for n in needs if n is not None]
        return max(known) if known else None


def available_gib() -> float | None:
    """Memory a new allocation can get now (GiB, the device the engines run on); None when unreadable."""
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
    free_gib: Callable[[], float | None] = available_gib
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reaper: Reaper | None = None  # default: a ``Reaper`` on ``docker`` and the real ``squeue``
    #: The GPU memory each process holds now, MiB by host pid (``nvidia-smi``); None when it cannot say.
    gpu_memory: Callable[[], dict[int, int] | None] = gpu_process_memory_mib
    sparkrun_cache: Path = SPARKRUN_CACHE
    _recipe_file: Path | None = field(default=None, init=False)
    _run_root: Path | None = field(default=None, init=False)
    _stopped: threading.Event = field(default_factory=threading.Event, init=False)
    _guard: threading.Thread | None = field(default=None, init=False)
    _killed: str | None = field(default=None, init=False)
    _previous_handler: Any = field(default=None, init=False)
    _free_before: float | None = field(default=None, init=False)
    _lowest_free: float | None = field(default=None, init=False)
    _host_peak: float | None = field(default=None, init=False)  # read in ``stop`` before the removal
    _sigterm_installed: bool = field(default=False, init=False)
    _oom_failing: bool | None = field(default=None, init=False)  # the last logged outcome
    _gpu_peak: float | None = field(default=None, init=False)  # GiB, the largest reading of the guard's polls
    _launched_at: float | None = field(default=None, init=False)  # wall clock, to find this run's cache leaf

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
        if not os.environ.get("SLURM_JOB_ID"):
            msg = (
                "a container engine starts only inside a slurm job (GPU work goes through slurm, and the "
                "reaper keeps a container only while its job runs): submit it with preflight's submit.py"
            )
            raise ContainerStartError(msg)
        on_disk = (  # the daemon's word, not only the record's: an image removed by hand is gone
            bool(self.record.image)
            and self.docker(["docker", "image", "inspect", self.record.image]).returncode == 0
        )
        if not self.record.pulled or not on_disk:
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
        try:  # checked before memory is weighed, so a start that cannot log in costs nothing
            loopback.loopback_login()
        except loopback.LoopbackLoginError as exc:
            raise ContainerStartError(str(exc)) from exc

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

    @staticmethod
    def model_mounts(real_model_dir: Path) -> list[Path]:
        """The model folder and every file its links reach outside it, each to be mounted read-only.

        A folder's blobs may themselves link into a store shared by several models (on this machine,
        ``hub/blobs/<2 hex>/<sha256>``); the folder alone leaves those links dangling in the container.
        """
        inside = real_model_dir.resolve()
        outside = {
            link.resolve()
            for link in real_model_dir.rglob("*")
            if link.is_symlink() and not link.resolve().is_relative_to(inside)
        }
        return [real_model_dir, *sorted(t for t in outside if t.is_file())]

    def _write_recipe(self, real_model_dir: Path) -> Path:
        """The recipe sparkrun runs: ours, the read-only model mounts and the run's labels (JSON is YAML)."""
        executor = dict(self.recipe.get("executor_config") or {})
        mounts = [f"{m}:{m}:ro" for m in self.model_mounts(real_model_dir)]
        executor["volumes"] = [*executor.get("volumes", []), *mounts]
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
        try:  # from here on, whatever ends the start (an interrupt included) stops what it began
            self._start(need)
        except BaseException:
            self.stop()
            raise
        return self

    def _start(self, need: float) -> None:
        """Write the recipe, install the stop paths and the guard, and start; raises when it did not start."""
        recipe = self._write_recipe(self._prepare_hub())
        self._install_sigterm()
        self._start_guard()
        self._launched_at = time.time()
        done = run_sparkrun(
            self.sparkrun,
            self._run_args(recipe, need),
            timeout=START_TIMEOUT_S,
            hub_root=self._run_root,
            logs_in=True,
        )
        if done.returncode != 0 or self._killed:
            if done.returncode != 0:  # the cause is often well above the last lines: keep it all in the log
                log.error("sparkrun run exited %d:\n%s\n%s", done.returncode, done.stdout, done.stderr)
            why = self._killed or f"sparkrun run exited {done.returncode}: {done.stderr.strip()[-500:]}"
            raise ContainerStartError(why)
        free = self._sample_free()  # the reading after the load counts toward the fall
        floor = self.guards.host_reserve_gib
        if floor is not None and (free is None or free < floor):
            msg = f"the server left {free} GiB available, below the {floor:g} GiB host reserve"
            raise NotEnoughMemoryError(msg)

    def __exit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None
    ) -> None:
        """Stop the server, whatever ended the block."""
        self.stop()

    def _run_args(self, recipe: Path, need_gib: float) -> list[str]:
        return [
            "run", str(recipe), "--tp", "1", "--hosts", "localhost", "--no-follow", "--offline",
            "--memory-limit", f"{math.ceil(need_gib)}G",  # docker --memory: binary units, the whole need
        ]  # fmt: skip

    def _install_sigterm(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signal handlers are the main thread's; a worker thread's run relies on the reaper
        previous = signal.getsignal(signal.SIGTERM)
        self._previous_handler = previous
        self._sigterm_installed = True

        def on_term(signum: int, frame: types.FrameType | None) -> None:
            self.stop()
            if callable(previous):
                previous(signum, frame)
            elif previous != signal.SIG_IGN:  # the default, or a handler not set from Python (None)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, on_term)

    def _start_guard(self) -> None:
        """Watch memory for the whole run: sample it, raise the OOM score, kill below the floor.

        Without a floor it still samples (the recorded fall) and scores: a floor of 0, never reached.
        """
        floor = self.guards.kill_below_gib or 0.0

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
                    on_poll=self._on_poll,
                )
            )
            while self._killed and not self._stopped.wait(POLL_S):
                self._docker_kill()  # sparkrun may still be starting one: kill whatever the run makes

        self._guard = threading.Thread(target=watch, name=f"guard-{self.run_id[:8]}", daemon=True)
        self._guard.start()

    def _on_poll(self) -> None:
        """At each guard poll: the OOM score of the container's processes, and the GPU memory they hold."""
        self._raise_oom_score()
        self._sample_gpu()

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
        if done.returncode != 0:  # unknown, not none: whatever runs is the reaper's to remove
            log.warning("container run %s: docker ps failed (%s)", self.run_id, done.stderr.strip()[:200])
            return []
        return done.stdout.split()

    def _docker_kill(self) -> None:
        for cid in self.container_ids():
            self.docker(["docker", "kill", cid])

    def _cgroup(self, cid: str) -> Path:
        return Path("/sys/fs/cgroup/system.slice") / f"docker-{cid}.scope"

    def _pids(self, cid: str) -> list[str]:
        """The processes of one container, from its cgroup (host pids)."""
        return (self._cgroup(cid) / "cgroup.procs").read_text(encoding="utf-8").split()

    def _sample_gpu(self) -> None:
        """Keep the largest GPU memory the container's processes hold (``nvidia-smi`` lists them by host pid).

        A failed call, or none of the container's processes listed yet, is no reading (not zero).
        """
        held = self.gpu_memory()
        if held is None:
            return
        mine: set[int] = set()
        for cid in self.container_ids():
            with contextlib.suppress(OSError):
                mine.update(int(pid) for pid in self._pids(cid))
        rows = [mib for pid, mib in held.items() if pid in mine]
        if rows:
            total = sum(rows) / MIB_PER_GIB
            self._gpu_peak = total if self._gpu_peak is None else max(self._gpu_peak, total)

    def _raise_oom_score(self) -> None:
        """Give every process of the container the guards' OOM score (raising it needs no privilege)."""
        score = self.guards.server_oom_score_adj
        if score is None:
            return
        done, failed = 0, []
        for cid in self.container_ids():
            try:
                pids = self._pids(cid)
            except OSError as exc:
                failed.append(f"{cid[:12]}: {exc}")
                continue
            for pid in pids:
                try:
                    Path(f"/proc/{pid}/oom_score_adj").write_text(str(score), encoding="utf-8")
                    done += 1
                except FileNotFoundError:
                    pass  # the process ended between the listing and the write
                except OSError as exc:  # another user's process (a container not run as ours)
                    failed.append(f"pid {pid}: {exc}")
        if (done or failed) and bool(failed) != self._oom_failing:  # logged when the outcome changes
            self._oom_failing = bool(failed)  # (a starting container's first processes may still be root's)
            if failed:
                log.warning("OOM score %d not set on %s (set on %d)", score, "; ".join(failed[:3]), done)
            else:
                log.info("OOM score %d set on the container's %d processes", score, done)

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
        peak = self._host_peak if self._stopped.is_set() else self.host_peak_gib()
        return MemoryNeed(fall_gib=fall, gpu_peak_gib=self._gpu_peak, host_peak_gib=peak)

    @property
    def launched(self) -> bool:
        """Whether ``sparkrun run`` was called: False when the start was refused before it (memory, image)."""
        return self._launched_at is not None

    @property
    def killed(self) -> str | None:
        """Why the memory guard killed this run's container; None when it did not."""
        return self._killed

    def _setting(self, keys: Sequence[str]) -> str:
        """The recipe's value for the first of ``keys`` it sets (the runtimes name these differently)."""
        defaults = self.recipe.get("defaults") or {}
        return next((str(defaults[key]) for key in keys if key in defaults), "")

    def _cache_off(self) -> bool:
        setting = self.recipe.get("runtime_cache", True)
        return setting is False or (isinstance(setting, dict) and setting.get("enabled") is False)

    def cache_leaf(self) -> Path | None:
        """The runtime-cache folder this run used: the one whose last-used marker was touched since launch."""
        family = self.sparkrun_cache / RUNTIME_CACHE_SUBDIR / self.record.engine.value
        since = (self._launched_at or 0.0) - 5.0  # the marker is touched by the launch script, not by us
        marks = [(m.stat().st_mtime, m.parent) for m in family.rglob(LAST_USED_MARKER) if m.is_file()]
        fresh = [(when, leaf) for when, leaf in marks if when >= since]
        return max(fresh, default=(0.0, None))[1]

    def stamp(self) -> dict[str, str]:
        """What a result served by this run carries (design 3.7).

        The two cache hashes are of a listing (sorted relative path and size of every file), not of the
        contents: the runtime cache holds gigabytes of compiled kernels, too slow to read at every stamp.
        """
        leaf = None if self._cache_off() else self.cache_leaf()
        family = self.record.engine.value
        return {
            "engine": family,
            "image": self.record.image,
            "tag": self.record.tag,
            "publisher": self.record.publisher.value,
            "sparkrun_version": self.record.sparkrun_version,
            "model_file": self.model,
            "run": self.run_id,
            "recipe_sha256": hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest(),
            "model_revision": str(self.recipe.get("model_revision", "")),
            "memory_fraction": self._setting(MEMORY_FRACTION_KEYS),
            "max_sequences": self._setting(MAX_SEQUENCES_KEYS),
            "autotune_cache_key": "off"
            if self._cache_off()
            else (
                leaf.relative_to(self.sparkrun_cache / RUNTIME_CACHE_SUBDIR).as_posix() if leaf else ABSENT
            ),
            "autotune_cache_hash": listing_hash(leaf) if leaf else ABSENT,
            "tuning_hash": listing_hash(self.sparkrun_cache / TUNING_SUBDIR / family),
        }

    def dry_run(self, *, verbose: bool = False) -> subprocess.CompletedProcess[str]:
        """``sparkrun run --dry-run`` in the per-run Hub layout of a real start; launches nothing.

        ``verbose`` adds sparkrun's ``-vvv``, the only level at which a dry run prints its ``docker run``
        command (checked on v0.4.2).
        """
        try:
            recipe = self._write_recipe(self._prepare_hub())
            args = [*(["-vvv"] if verbose else []), *self._run_args(recipe, self.need_gib()), "--dry-run"]
            return run_sparkrun(
                self.sparkrun, args, timeout=STOP_TIMEOUT_S, hub_root=self._run_root, logs_in=True
            )
        finally:
            self._remove_files()

    def _remove_files(self) -> None:
        if self._run_root is not None:
            shutil.rmtree(self._run_root, ignore_errors=True)
        if self._recipe_file is not None:
            self._recipe_file.unlink(missing_ok=True)

    def stop(self) -> None:
        """Stop the server and remove its containers, then log what was left; safe to call more than once."""
        if self._stopped.is_set():
            return
        self._host_peak = self.host_peak_gib()  # the cgroup goes with the container
        self._stopped.set()
        if self._recipe_file is not None:
            run_sparkrun(
                self.sparkrun,
                ["stop", str(self._recipe_file), "--tp", "1", "--hosts", "localhost"],
                timeout=STOP_TIMEOUT_S,
                hub_root=self._run_root,
                logs_in=True,
            )
        for cid in self.container_ids():
            self.docker(["docker", "rm", "-f", cid])
        left = self.container_ids()
        log.info("container run %s stopped; containers left: %s", self.run_id, left or "none")
        if self._sigterm_installed and threading.current_thread() is threading.main_thread():
            # None: a handler not set from Python, which cannot be put back; the default stands in
            signal.signal(signal.SIGTERM, self._previous_handler or signal.SIG_DFL)
        self._remove_files()


# --- the central server for a bulk job (design 3.8) -----------------------------------------------

LOCALHOST: Final = "127.0.0.1"


@dataclass(frozen=True)
class LlamaFallback:
    """Everything llama.cpp's bulk layout needs, and what the container's start falls back to.

    ``binary``, ``guards`` and ``params`` default to the current build, the refresh's guards and the
    bulk layout derived from the measured device and memory.
    """

    model_path: Path  # the GGUF
    lease: GpuLease
    http: httpx.AsyncClient
    pid_dir: Path
    port: int
    binary: Path | None = None
    guards: MemoryGuards | None = None
    params: ServingParams | None = None


@dataclass(frozen=True)
class BulkServer:
    """A running server: its address, the engine behind it, its stamp, and why it is the fallback if it is."""

    url: str
    engine: Engine
    stamp: dict[str, Any]
    fallback: str | None


def _container_run(answer: SettingsAnswer, fallback: LlamaFallback) -> tuple[ContainerRun, str]:
    """The run for the chosen version and file, and its URL; ``ContainerStartError`` if it cannot be made."""
    from . import engine_refresh  # noqa: PLC0415 - engine_refresh imports this module

    choice = answer.choice
    assert choice is not None  # noqa: S101 - the caller checked
    record = read_record(choice.engine, choice.version) if choice.version else None
    model = (choice.file or "").partition("@")[0]
    if record is None:
        msg = f"{choice.engine.value} {choice.version}: no version record"
        raise ContainerStartError(msg)
    recipe = engine_refresh.realized(record, model)
    if recipe is None:
        msg = f"{choice.engine.value} {record.tag}: no realized recipe for {model}"
        raise ContainerStartError(msg)
    port = (recipe.get("defaults") or {}).get("port")
    if not isinstance(port, int):
        msg = f"{choice.engine.value} {record.tag}: the recipe for {model} names no port"
        raise ContainerStartError(msg)
    run = ContainerRun(
        record,
        model,
        recipe,
        ensure_sparkrun(engine_refresh.in_use_version()),
        fallback.guards or engine_refresh.GUARDS,
        kind=RunKind.PROJECT,
    )
    return run, f"http://{LOCALHOST}:{port}"


@contextlib.asynccontextmanager
async def _llama_server(
    answer: SettingsAnswer, fallback: LlamaFallback, reason: str | None
) -> AsyncIterator[BulkServer]:
    """llama.cpp's bulk layout through the launcher, started and stopped by the context."""
    from . import engine_refresh  # noqa: PLC0415 - engine_refresh imports this module
    from .llama_builds import current_server, llama_home  # noqa: PLC0415

    params = fallback.params
    if params is None:
        report = inspect_hardware()
        params = derive_serving_params(report.device, report.total_memory_gb, role=ServingRole.BULK)
    settings = answer.model_copy(update={"engine": Engine.LLAMA_CPP})
    manager = LocalServerManager(
        ServerSpec(LOCALHOST, fallback.port, fallback.model_path),
        params,
        fallback.lease,
        http=fallback.http,
        pid_dir=fallback.pid_dir,
        binary=fallback.binary or current_server(llama_home()),
        ready_timeout_s=engine_refresh.LLAMA_READY_S,
        guards=fallback.guards or engine_refresh.GUARDS,
        settings=settings,
    )
    await manager.start()
    try:
        stamp = manager.stamp(settings=settings).model_dump(mode="json")
        yield BulkServer(
            f"http://{LOCALHOST}:{fallback.port}",
            Engine.LLAMA_CPP,
            {**stamp, **({"fallback": reason} if reason else {})},
            reason,
        )
    finally:
        await manager.stop()


@contextlib.asynccontextmanager
async def bulk_server(answer: SettingsAnswer, *, fallback: LlamaFallback) -> AsyncIterator[BulkServer]:
    """A running server for a bulk job, as the answer chose: the container engine, else llama.cpp.

    A container answer starts that engine's chosen version on the approved file (the run is entered on the
    calling thread, so its SIGTERM chain is installed when that is the main thread). When the start is
    refused or fails (``ContainerStartError``, ``NotEnoughMemoryError``) llama.cpp's bulk layout is started
    instead and the stamp says why (an ``OSError`` while preparing it too). A llama.cpp answer starts
    llama.cpp. Any other engine, or a container engine named rather than chosen (no version), is no server:
    ``ValueError``. No project writes its own fallback.
    """
    engine = answer.choice.engine if answer.choice else answer.engine
    reason = None
    if engine in CONTAINER_ENGINES and answer.choice is None:
        msg = (
            f"{engine.value} was named, not chosen, so no version is known: "
            "`lookup(..., engine=None, job=...)` chooses the version and file bulk_server starts"
        )
        raise ValueError(msg)
    if engine in CONTAINER_ENGINES:
        stack = contextlib.ExitStack()
        try:
            run, url = _container_run(answer, fallback)
            stack.enter_context(run)
        except (ServerStartError, OSError) as exc:  # ContainerStartError, NotEnoughMemoryError, or the disk
            reason = f"{engine.value} did not start, llama.cpp serves instead: {exc}"
            log.warning("%s", reason)
        else:
            with stack:
                yield BulkServer(
                    url, engine, {**run.stamp(), "settings": answer.stamp().model_dump(mode="json")}, None
                )
            return
    elif engine is not Engine.LLAMA_CPP:
        msg = f"{engine.value} is not a server: bulk_server starts llama.cpp or a container engine"
        raise ValueError(msg)
    async with _llama_server(answer, fallback, reason) as served:
        yield served


def listing_hash(folder: Path) -> str:
    """A hash of every file's relative path and size under ``folder``; "absent" if it is not there."""
    if not folder.is_dir():
        return ABSENT
    rows = sorted(
        (p.relative_to(folder).as_posix(), p.stat().st_size) for p in folder.rglob("*") if p.is_file()
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def record_result(
    record: VersionRecord, model: str, run: ContainerRun, *, extra: Mapping[str, object]
) -> None:
    """Add a run's memory and result to the version's record, so the next start's need is measured.

    ``extra`` is the caller's part of the entry (``kind``, ``job``, ``result_file``, ``complete``,
    ``figures``, ``startup_s``, ``stamp``); the GPU peak the run sampled is added from ``run.memory()``.
    """
    latest = read_record(record.engine, record.tag) or record  # a job may have added results meanwhile
    memory = run.memory()
    entry = {
        "model": model,
        "memory": dataclasses.asdict(memory),
        "run": run.run_id,
        "gpu_peak_gib": memory.gpu_peak_gib,
        **extra,
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
        # %A, not %i: an array task's own id, as its SLURM_JOB_ID (and so its label) carries it
        done = self.squeue(["squeue", "-h", "-a", "-o", "%A|%T|%S"])
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
            if not box.job.isdigit():  # not one of ours (ours start only inside a job): never guessed at
                report.kept.append(f"{box.cid[:12]}: label {JOB_LABEL}={box.job!r} is not a slurm job id")
                continue
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
