"""Local llama-server process lifecycle management and supervision, torch-free.

One ``LocalServerManager`` owns one ``(spec, params, lease)`` server instance. ``Contention``
lets a caller choose service_a's policy (``REFUSE``: never start while another project holds
the GPU) or service_c's (``WAIT``: queue for it, as ``run_gpu_queue.sh`` does) without either
project's launcher code living here — this package exposes the choice, not a default project.
``STAY_UP`` is the opt-in third policy: the server stays loaded beside other work, registered in the lease
directory's ``resident/`` folder instead of holding the lease, and unloads when memory runs short or a quiet
card asks for the machine to itself (``lease.QuietCard``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import socket
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .hardware import available_memory, network_filesystem
from .lease import GpuLease, GpuLeaseClient, LeaseRecord, ResidentRecord
from .process import (
    DRAFT_MODEL_ENV,
    DRAFT_MODEL_FLAGS,
    UBATCH_ENV,
    UBATCH_FLAGS,
    LocalModelsError,
    ServerHandle,
    ServerSpec,
    ServerStatus,
    _is_pid_alive,
    _process_args,
    adds_ubatch,
    auth_headers,
    build_args,
    carries_speculation,
    llama_flags,
    llama_option,
    require_key_off_loopback,
    speculates,
)
from .stack import Engine, LlamaLayout, ServerStamp, Settings, fingerprint, llama_cpp_build

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .params import ServingParams
    from .settings import SettingsAnswer

__all__ = [
    "Contention",
    "LeaseConflictError",
    "LeaseWaitTimeoutError",
    "LocalServerManager",
    "MemoryGuards",
    "NotEnoughMemoryError",
    "PortHeldError",
    "ReservedPortError",
    "ServerStartError",
    "StartupYieldedError",
    "StayUpConfigError",
    "SupervisorExit",
    "stay_up_problem",
    "supervise",
]

_HTTP_OK = 200
_GIB = 1024**3
_MIB_PER_GIB = 1024
# llama-fit-params prints this only at verbosity 4 (measured by service_c, 2026-10-07).
_PROJECTION_RE = re.compile(r"projected to use (\d+) MiB of device memory")
_FIT_PARAMS_TIMEOUT_S = 600.0
# The score is set by a shell that then execs the server (same PID): preexec_fn is not thread-safe.
_OOM_WRAPPER = 'echo {score} > /proc/self/oom_score_adj && exec "$0" "$@"'

_FILE_FLAGS = ("-m", "--api-key-file", *DRAFT_MODEL_FLAGS)

log = logging.getLogger(__name__)


class Contention(StrEnum):
    """What ``LocalServerManager.start()`` does when another live lease holds the GPU."""

    REFUSE = "refuse"
    WAIT = "wait"
    STAY_UP = "stay_up"


class MemoryGuards(BaseModel):
    """Memory protection for a server on a host whose GPU shares its RAM; every guard is off unless set.

    ``kill_below_gib``: SIGKILL the server at once when available memory falls below it, polled every
    ``load_poll_s`` while it loads and every ``watch_poll_s`` while it runs. ``host_reserve_gib``: before the
    load, ``llama-fit-params``' projection plus this must fit in available memory (no projection counts as
    not fitting); after the load, the server is stopped if it left less than this available.
    ``server_oom_score_adj``: the server's OOM score, so the kernel's killer picks it first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kill_below_gib: float | None = Field(default=None, gt=0)
    host_reserve_gib: float | None = Field(default=None, gt=0)
    server_oom_score_adj: int | None = Field(default=None, ge=-1000, le=1000)
    #: Default: ``llama-fit-params`` beside the server binary.
    fit_params_binary: Path | None = None
    load_poll_s: float = Field(default=1.0, gt=0)
    watch_poll_s: float = Field(default=5.0, gt=0)


def stay_up_problem(guards: MemoryGuards, resident_allowance_gib: float | None) -> str | None:
    """Why ``Contention.STAY_UP`` cannot be used with these settings, in plain words; None when it can."""
    wanted = {
        "MemoryGuards.kill_below_gib": guards.kill_below_gib,
        "MemoryGuards.server_oom_score_adj": guards.server_oom_score_adj,
        "resident_allowance_gib": resident_allowance_gib,
    }
    missing = [name for name, value in wanted.items() if value is None]
    if not missing:
        return None
    return (
        f"stay-up mode needs {', '.join(missing)}: it keeps a model loaded beside other work, so the kernel "
        "must pick its server first and the watchdog must free the memory when it runs short"
    )


def _available_gib(on_gpu: bool) -> float | None:
    """Available memory in GiB for a GPU or CPU server (``available_memory``); None when unreadable."""
    reading = available_memory("cuda" if on_gpu else "cpu")
    return None if reading is None else reading.bytes / _GIB


class SupervisorExit(StrEnum):
    """Exit reason of the local model server supervisor."""

    STOPPED = "stopped"
    SERVER_EXITED = "server_exited"
    YIELDED = "yielded"


class StayUpConfigError(LocalModelsError):
    """Raised when ``Contention.STAY_UP`` is asked for without the guards that make it safe."""


class ReservedPortError(LocalModelsError):
    """Raised when ``spec.port`` is in the caller's reserved set."""


class PortHeldError(LocalModelsError):
    """Raised when ``spec.port`` is already bound by a process this manager did not start."""


class LeaseConflictError(LocalModelsError):
    """Raised when ``Contention.REFUSE`` finds another live lease already holding the GPU."""


class LeaseWaitTimeoutError(LocalModelsError):
    """Raised when ``Contention.WAIT`` exceeds ``wait_timeout_s`` without acquiring the lease."""


class ServerStartError(LocalModelsError):
    """Raised when the server process fails to start, become healthy, or match its spec."""


class NotEnoughMemoryError(ServerStartError):
    """Raised when available memory is too low to load the model, or too low once it has loaded.

    A stay-up service waits and tries again (``service.run_serve``); any other caller sees a failed start.
    """


class StartupYieldedError(ServerStartError):
    """Raised when another project requests the GPU while the server is still loading.

    The server has already been stopped and the lease released: this is a yield, not a failure,
    so a long-running caller waits for the other lease to clear and starts again.
    """


def _read_pid_if_exists(pidfile: Path) -> int | None:
    """Read PID from pidfile if it exists, removing corrupted files."""
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except (ValueError, OSError):
        with contextlib.suppress(OSError):
            pidfile.unlink(missing_ok=True)
        return None


def _unlink_pid(pidfile: Path) -> None:
    """Safely unlink a pidfile ignoring errors."""
    with contextlib.suppress(OSError):
        pidfile.unlink(missing_ok=True)


def _write_pid(pidfile: Path, pid: int) -> None:
    """Write PID to pidfile."""
    pidfile.write_text(f"{pid}\n")


def _prepare_pid_dir(pid_dir: Path) -> None:
    """Ensure PID directory exists."""
    pid_dir.mkdir(parents=True, exist_ok=True)


def _open_log(log_path: Path) -> BinaryIO:
    """Open log file for process output."""
    return log_path.open("wb")


def _resolved(args: list[str]) -> list[str]:
    """``args`` with the value of each flag that names a file resolved, so two spellings of one file agree."""
    return [
        str(Path(arg).resolve()) if i and args[i - 1] in _FILE_FLAGS else arg for i, arg in enumerate(args)
    ]


def _is_llama_server_on_port(args: list[str], port: int) -> bool:
    """Check if process arguments belong to a llama-server running on the specified port."""
    if not any("llama-server" in a for a in args):
        return False

    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            try:
                if int(args[i + 1]) == port:
                    return True
            except ValueError:
                pass
        elif a.startswith("--port="):
            try:
                if int(a.split("=", 1)[1]) == port:
                    return True
            except ValueError:
                pass
    return False


class LocalServerManager:
    """Manages lifecycle, health verification, and process isolation of one local llama-server.

    With ``settings`` (a llama.cpp answer) it also passes the answer's ``-ub`` and speculation, each group
    unless the caller's own ``extra_flags`` or ``extra_env`` carry it. The command line a server runs is
    ``build_args(binary, spec, params, manager.effective_flags)``, not ``manager.extra_flags``. The caller
    resolves ``answer.settings.model_file`` into ``spec.model_path``; a file the manager does not serve is
    listed as ``overridden`` in the stamp.
    """

    def __init__(
        self,
        spec: ServerSpec,
        params: ServingParams,
        lease: GpuLease,
        *,
        reserved_ports: frozenset[int] = frozenset(),
        contention: Contention = Contention.REFUSE,
        ready_timeout_s: float = 60.0,
        http: httpx.AsyncClient,
        pid_dir: Path,
        binary: str | Path,
        extra_flags: tuple[str, ...] = (),
        extra_env: Mapping[str, str] | None = None,
        stop_grace_s: float = 20.0,
        wait_poll_s: float = 2.0,
        wait_timeout_s: float | None = None,
        min_slot_tokens: int = 0,
        guards: MemoryGuards | None = None,
        resident_allowance_gib: float | None = None,
        available_gib: Callable[[bool], float | None] = _available_gib,
        settings: SettingsAnswer | None = None,
    ) -> None:
        if settings is not None and settings.engine is not Engine.LLAMA_CPP:
            msg = f"a llama-server takes a llama.cpp settings answer, not one for {settings.engine.value}"
            raise ValueError(msg)
        self.guards = guards or MemoryGuards()
        if contention is Contention.STAY_UP and (
            problem := stay_up_problem(self.guards, resident_allowance_gib)
        ):
            raise StayUpConfigError(problem)
        self.spec = spec
        self.params = params
        self.lease = lease
        self.reserved_ports = reserved_ports
        self.contention = contention
        self.ready_timeout_s = ready_timeout_s
        self._http = http
        self.pid_dir = Path(pid_dir)
        self.binary = binary
        self.extra_flags = extra_flags
        self.extra_env = extra_env
        self.stop_grace_s = stop_grace_s
        self.wait_poll_s = wait_poll_s
        self.wait_timeout_s = wait_timeout_s
        self.min_slot_tokens = min_slot_tokens
        self.resident_allowance_gib = resident_allowance_gib
        self._available_gib = available_gib
        #: The settings answer this server starts with; None starts it exactly as it always was.
        self.settings = settings
        self._watchdog: asyncio.Task[None] | None = None
        self._ready = False
        #: Why the watchdog killed the last server it started, in plain words; None if it did not.
        self.killed_reason: str | None = None
        self._registered_at: int | None = None  # stay-up: when the resident record was first written

    def _set_loaded(self, *, loaded: bool) -> None:
        """Stay-up only: publish whether this manager's server is loaded, in the ``resident/`` record."""
        if self.contention is Contention.STAY_UP:
            assert self.resident_allowance_gib is not None  # noqa: S101 - narrowing for mypy; the constructor checked
            self._registered_at = self._registered_at or int(time.time())
            self.lease.write_resident(
                ResidentRecord(
                    pid=os.getpid(),
                    project=self.lease.project,
                    port=self.spec.port,
                    model_path=str(self.spec.model_path),
                    allowance_gib=self.resident_allowance_gib,
                    loaded=loaded,
                    started_at=self._registered_at,
                )
            )

    def release(self) -> None:
        """Give up what this manager published: its lease, or in stay-up mode its loaded flag."""
        if self.contention is Contention.STAY_UP:
            self._set_loaded(loaded=False)
        else:
            self.lease.remove()

    def yield_requested_by(self) -> str | None:
        """The project this server should step aside for now, if any.

        Stay-up mode yields to a live quiet request only; otherwise a GPU server yields to another project's
        lease request (a server with no GPU layers holds nothing the other project's GPU work needs).
        """
        if self.contention is Contention.STAY_UP:
            quiet = self.lease.quiet_requests()
            return quiet[0].project if quiet else None
        return self.lease.requested_by_other() if self.params.n_gpu_layers != 0 else None

    def _pidfile(self) -> Path:
        return self.pid_dir / f"{self.spec.port}.pid"

    @property
    def effective_flags(self) -> tuple[str, ...]:
        """The caller's ``extra_flags`` plus what the settings answer adds (see ``llama_flags``)."""
        if self.settings is None:
            return self.extra_flags
        return (*self.extra_flags, *llama_flags(self.settings.settings, self.extra_flags, self.extra_env))

    def _applied(self) -> tuple[Settings, list[str]]:
        """The settings this server got from the answer, and those the caller's own flags or env set aside."""
        if self.settings is None:
            return Settings(), []
        chosen = self.settings.settings
        ubatch = adds_ubatch(chosen, self.extra_flags, self.extra_env)
        speculation = chosen.spec is not None and not carries_speculation(self.extra_flags, self.extra_env)
        applied = Settings(
            n_ubatch=chosen.n_ubatch if ubatch else None,
            spec=chosen.spec if speculation else None,
            spec_draft_file=chosen.spec_draft_file if speculation and speculates(chosen) else None,
        )
        served = {Path(file.path).name for file in chosen.model_file or ()}
        wanted = (
            ("n_ubatch", chosen.n_ubatch is not None, ubatch),
            ("spec", chosen.spec is not None, speculation),
            ("model_file", chosen.model_file is not None, self.spec.model_path.name in served),
        )
        return applied, [name for name, asked, passed in wanted if asked and not passed]

    def stamp(self, model_sha256: str | None = None, settings: SettingsAnswer | None = None) -> ServerStamp:
        """How this server is made: llama.cpp's build, the model file, its layout and the host.

        ``model_sha256`` is the file's hash when the caller knows it (a ``PinnedFile``); never computed here.
        ``settings`` is the lookup answer the caller took its settings from; a manager built with an answer
        stamps that one (a different ``settings`` raises ``ValueError``: this server runs what it was built
        with). The stamp's ``applied`` is what the answer gave this server (nothing if the manager has no
        answer), and a group the caller's own flags set aside is added to its ``overridden``.
        """
        if settings is not None and self.settings is not None and settings != self.settings:
            msg = "this server runs the answer its manager was built with; it cannot be stamped with another"
            raise ValueError(msg)
        answer = settings if settings is not None else self.settings
        settings_stamp = None
        if answer is not None:
            applied, kept_out = self._applied()
            stamped = answer.stamp(applied)
            kept_out = [name for name in kept_out if name not in stamped.overridden]
            settings_stamp = stamped.model_copy(update={"overridden": [*stamped.overridden, *kept_out]})
        return ServerStamp(
            engine=Engine.LLAMA_CPP,
            build=llama_cpp_build(Path(self.binary)),
            model_path=str(self.spec.model_path),
            model_sha256=model_sha256,
            layout=LlamaLayout(
                n_ctx=self.params.n_ctx,
                n_parallel=self.params.n_parallel,
                kv_unified=self.params.kv_unified,
                extra_flags=self.effective_flags,
            ),
            host=fingerprint(),
            settings=settings_stamp,
        )

    def free_gib(self) -> float | None:
        """Available memory in GiB for this server's device; None when unreadable."""
        return self._available_gib(self.params.n_gpu_layers != 0)

    async def _projected_gib(self) -> float:
        """llama.cpp's projection of the server's memory (weights, context, compute); raises if none.

        With a settings answer it is projected at the effective ``-ub`` and, when a draft model is among the
        effective flags, the draft file's own projection is added: ``llama-fit-params`` fits one model only.
        """
        ubatch = draft = None
        if self.settings is not None:
            flags = self.effective_flags
            ubatch = llama_option(flags, self.extra_env, UBATCH_FLAGS, UBATCH_ENV)
            draft = llama_option(flags, self.extra_env, DRAFT_MODEL_FLAGS, DRAFT_MODEL_ENV)
        total = await self._fit_gib(self.spec.model_path, ubatch)
        if draft is not None:
            total += await self._fit_gib(Path(draft), ubatch)
        return total

    async def _fit_gib(self, model_path: Path, ubatch: str | None) -> float:
        """One ``llama-fit-params`` projection of ``model_path`` at this server's context, in GiB."""
        fit = self.guards.fit_params_binary or Path(self.binary).with_name("llama-fit-params")
        cmd = [
            str(fit),
            "-m",
            str(model_path),
            "-c",
            str(self.params.n_ctx),
            "-np",
            str(self.params.n_parallel),
            *([UBATCH_FLAGS[0], ubatch] if ubatch is not None else []),
            "-lv",
            "4",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_FIT_PARAMS_TIMEOUT_S)
        except (OSError, TimeoutError) as exc:
            msg = f"llama-fit-params could not run ({exc}); not loading a model it has not projected"
            raise ServerStartError(msg) from exc
        match = _PROJECTION_RE.search(out.decode(errors="replace"))
        if match is None:
            msg = (
                f"llama-fit-params printed no projection for {model_path.name} "
                f"(exit {proc.returncode}); not loading a model it has not projected"
            )
            raise ServerStartError(msg)
        return int(match.group(1)) / _MIB_PER_GIB

    async def _check_fits(self) -> None:
        """Refuse to load when the projection plus the host reserve exceeds available memory."""
        reserve = self.guards.host_reserve_gib
        if reserve is None:
            return
        need = await self._projected_gib() + reserve
        free = self.free_gib()
        if free is None:
            msg = "available memory cannot be read, so the model's fit cannot be checked; not loading it"
            raise ServerStartError(msg)
        if need > free:
            msg = (
                f"{self.spec.model_path.name} needs {need:.1f} GiB (llama.cpp's projection plus a "
                f"{reserve:g} GiB host reserve) but {free:.1f} GiB is available; not loading it"
            )
            raise NotEnoughMemoryError(msg)

    async def _check_left_over(self) -> None:
        """After the load: stop the server if it left less than the host reserve available."""
        reserve = self.guards.host_reserve_gib
        free = self.free_gib()
        if reserve is None or free is None or free >= reserve:
            return
        await self.stop()
        msg = (
            f"the loaded server left {free:.1f} GiB available, under the {reserve:g} GiB reserve; stopped it"
        )
        raise NotEnoughMemoryError(msg)

    async def _watch_memory(self, pid: int, floor: float) -> None:
        """SIGKILL the server's process group the moment available memory falls below ``floor``."""
        while _is_pid_alive(pid):
            await asyncio.sleep(self.guards.watch_poll_s if self._ready else self.guards.load_poll_s)
            free = self.free_gib()
            if free is not None and free < floor:
                self.killed_reason = (
                    f"available memory fell to {free:.1f} GiB, below the {floor:g} GiB kill floor"
                )
                log.error("killing llama-server PID %d: %s", pid, self.killed_reason)
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except OSError:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGKILL)
                self._set_loaded(loaded=False)
                return

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    async def _try_adopt(self, pidfile: Path) -> ServerHandle | None:
        pid = _read_pid_if_exists(pidfile)
        if pid is None:
            return None

        if not _is_pid_alive(pid):
            _unlink_pid(pidfile)
            return None

        args = _process_args(pid)
        if not _is_llama_server_on_port(args, self.spec.port):
            return None

        with contextlib.suppress(httpx.HTTPError, OSError):
            resp = await self._http.get(
                f"http://{self.spec.host}:{self.spec.port}/props",
                timeout=2.0,
                headers=auth_headers(self.spec.api_key_file),
            )
            if resp.status_code == _HTTP_OK:
                data = resp.json()
                if Path(data.get("model_path", "")).name == self.spec.model_path.name:
                    if self.settings is not None and not self._runs_as_started_now(args):
                        # a server with other flags would run the old settings under the new answer's stamp
                        if holders := self._other_holders():
                            msg = (
                                f"port {self.spec.port} runs a llama-server started with other arguments and "
                                f"{', '.join(holders)} holds it; not stopping it"
                            )
                            raise LeaseConflictError(msg)
                        await self.stop()
                        return None
                    return ServerHandle(pid=pid, port=self.spec.port, model_path=self.spec.model_path)
        return None

    def _other_holders(self) -> list[str]:
        """The other projects whose live lease or resident record names this server's port."""
        port, own = self.spec.port, self.lease.project
        held = [lease.project for lease in self.lease.live_others() if lease.port == port]
        return [*held, *(r.project for r in self.lease.residents() if r.port == port and r.project != own)]

    def _runs_as_started_now(self, args: list[str]) -> bool:
        """Whether a running server's arguments are those this manager would start it with.

        The binary is not compared, and a file given under another spelling of its path is the same file.
        """
        wanted = build_args(self.binary, self.spec, self.params, self.effective_flags)
        return _resolved(args[len(args) - len(wanted) + 1 :]) == _resolved(wanted[1:])

    def _check_port_free(self) -> None:
        try:
            with socket.create_connection((self.spec.host, self.spec.port), timeout=0.2):
                msg = (
                    f"port {self.spec.port} is held by a process this manager did not start; free it by hand"
                )
                raise PortHeldError(msg)
        except OSError:
            pass

    async def _acquire_lease(self) -> None:
        """Write this instance's GPU lease, honouring ``Contention`` against other live leases.

        ``WAIT`` publishes the request first, so a server that yields to waiting projects yields to this one,
        and takes its turn in request order (``GpuLeaseClient``): two waiters cannot wait for each other.
        """
        if self.contention is Contention.REFUSE:
            others = self.lease.live_others()
            if others:
                holders = ", ".join(o.project for o in others)
                msg = f"cannot acquire GPU lease; currently held by: {holders}"
                raise LeaseConflictError(msg)
        else:
            client = GpuLeaseClient(self.lease, port=self.spec.port)
            client.begin_wait()
            deadline = time.time() + self.wait_timeout_s if self.wait_timeout_s is not None else None
            while client.step():
                if deadline is not None and time.time() > deadline:
                    client.release()
                    msg = f"timed out waiting {self.wait_timeout_s}s for the GPU lease"
                    raise LeaseWaitTimeoutError(msg)
                await asyncio.sleep(self.wait_poll_s)
        self._write_lease()

    def _write_lease(self) -> None:
        self.lease.write(
            LeaseRecord(
                project=self.lease.project,
                pid=os.getpid(),
                port=self.spec.port,
                model_path=str(self.spec.model_path),
                started_at=int(time.time()),
            )
        )

    async def _poll_ready(self, proc: asyncio.subprocess.Process, log_path: Path) -> None:
        deadline = time.time() + self.ready_timeout_s
        while time.time() < deadline:
            if proc.returncode is not None:
                await self.stop()
                if self.killed_reason is not None:
                    msg = f"llama-server was killed while loading: {self.killed_reason}"
                    raise ServerStartError(msg)
                msg = f"llama-server exited prematurely with code {proc.returncode}; see {log_path}"
                raise ServerStartError(msg)

            if other := self.yield_requested_by():
                await self.stop()
                msg = f"GPU lease requested by {other}; yielding"
                raise StartupYieldedError(msg)

            with contextlib.suppress(httpx.HTTPError, OSError):
                resp = await self._http.get(f"http://{self.spec.host}:{self.spec.port}/health", timeout=1.0)
                if resp.status_code == _HTTP_OK:
                    return
            await asyncio.sleep(0.5)

        await self.stop()
        msg = f"llama-server failed to become ready within {self.ready_timeout_s}s; see {log_path}"
        raise ServerStartError(msg)

    def _check_slot_guarantee(self) -> None:
        """Under ``-kvu`` /props reports the whole pool as every slot's context, so check the sizes here."""
        need = self.params.n_parallel * self.min_slot_tokens
        if self.params.kv_unified and self.min_slot_tokens > 0 and self.params.n_ctx < need:
            msg = (
                f"n_ctx {self.params.n_ctx} cannot give each of {self.params.n_parallel} slots "
                f"min_slot_tokens {self.min_slot_tokens} (needs {need}); not starting the server"
            )
            raise ServerStartError(msg)

    async def _verify_props(self) -> None:
        if self.min_slot_tokens <= 0:
            return
        try:
            resp = await self._http.get(
                f"http://{self.spec.host}:{self.spec.port}/props",
                timeout=5.0,
                headers=auth_headers(self.spec.api_key_file),
            )
            data = resp.json() if resp.status_code == _HTTP_OK else {}
        except Exception as exc:
            await self.stop()
            msg = f"failed to query /props from llama-server: {exc}"
            raise ServerStartError(msg) from exc

        model_name = Path(data.get("model_path", "")).name
        n_ctx = data.get("default_generation_settings", {}).get("n_ctx", 0)
        if model_name != self.spec.model_path.name:
            await self.stop()
            msg = f"server model mismatch: expected {self.spec.model_path.name}, got {model_name}"
            raise ServerStartError(msg)
        if n_ctx < self.min_slot_tokens:
            await self.stop()
            msg = f"server n_ctx {n_ctx} is smaller than required min_slot_tokens {self.min_slot_tokens}"
            raise ServerStartError(msg)

    def _publish_adopted(self, pid: int) -> None:
        """An adopted server is loaded: say so in the stay-up record, or hold the lease if it uses the GPU.

        A stay-up server is watched against its floor as if this manager had started it.
        """
        if self.contention is Contention.STAY_UP:
            self._set_loaded(loaded=True)
            assert self.guards.kill_below_gib is not None  # noqa: S101 - narrowing for mypy; the constructor checked
            self._ready, self.killed_reason = True, None
            self._watchdog = asyncio.create_task(self._watch_memory(pid, self.guards.kill_below_gib))
        elif self.params.n_gpu_layers != 0:
            self._write_lease()

    def _begin_stay_up(self) -> None:
        """Publish ``loaded`` for a stay-up start, then yield at once if a quiet card is already waiting.

        Publishing first and looking second: a card writes its request and then looks at the residents, so one
        of the two sees the other and they cannot both go ahead.
        """
        self._set_loaded(loaded=True)
        if other := self.yield_requested_by():
            self._set_loaded(loaded=False)
            msg = f"GPU lease requested by {other}; yielding"
            raise StartupYieldedError(msg)

    async def start(self) -> ServerHandle:
        """Start this instance's llama-server process, or adopt an identical existing one."""
        if self.spec.port in self.reserved_ports:
            msg = f"port {self.spec.port} is reserved; this manager never starts a server on it"
            raise ReservedPortError(msg)
        self._check_slot_guarantee()

        _prepare_pid_dir(self.pid_dir)
        pidfile = self._pidfile()

        adopted = await self._try_adopt(pidfile)
        if adopted is not None:
            self._publish_adopted(adopted.pid)
            return adopted

        self._check_port_free()
        require_key_off_loopback(self.spec)
        if (kind := network_filesystem(self.spec.model_path)) is not None:
            log.warning(
                "model %s is on a network filesystem (%s): every cold start reads it at network speed; "
                "keep model files on a local disk",
                self.spec.model_path,
                kind,
            )
        args = build_args(self.binary, self.spec, self.params, self.effective_flags)
        if self.guards.server_oom_score_adj is not None:
            args = ["sh", "-c", _OOM_WRAPPER.format(score=self.guards.server_oom_score_adj), *args]
        await self._check_fits()

        publishes = self.contention is Contention.STAY_UP or self.params.n_gpu_layers != 0
        if self.contention is Contention.STAY_UP:
            self._begin_stay_up()
        elif publishes:
            await self._acquire_lease()

        log_path = self.pid_dir / f"{self.spec.port}.log"
        try:
            with _open_log(log_path) as log_file:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=None if self.extra_env is None else {**os.environ, **self.extra_env},
                )

            _write_pid(pidfile, proc.pid)
            self._ready, self.killed_reason = False, None
            if self.guards.kill_below_gib is not None:
                self._watchdog = asyncio.create_task(self._watch_memory(proc.pid, self.guards.kill_below_gib))
            await self._poll_ready(proc, log_path)
            await self._verify_props()
            await self._check_left_over()
            self._ready = True
        except Exception:
            self._cancel_watchdog()
            if publishes:
                self.release()
            raise

        return ServerHandle(pid=proc.pid, port=self.spec.port, model_path=self.spec.model_path)

    async def _terminate_process(self, pid: int) -> bool:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)

        deadline = time.time() + self.stop_grace_s
        while time.time() < deadline:
            if not _is_pid_alive(pid):
                return True
            await asyncio.sleep(0.05)

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

        kill_deadline = time.time() + 1.0
        while time.time() < kill_deadline:
            if not _is_pid_alive(pid):
                return True
            await asyncio.sleep(0.05)

        return not _is_pid_alive(pid)

    async def stop(self) -> bool:
        """Stop this instance's tracked server process cleanly, SIGTERM escalating to SIGKILL."""
        if self.spec.port in self.reserved_ports:
            msg = f"port {self.spec.port} is reserved; this manager never stops a server on it"
            raise ReservedPortError(msg)

        self._cancel_watchdog()
        pidfile = self._pidfile()
        pid = _read_pid_if_exists(pidfile)
        if pid is None:
            return False

        if not _is_pid_alive(pid):
            _unlink_pid(pidfile)
            self.release()
            return False

        args = _process_args(pid)
        if not _is_llama_server_on_port(args, self.spec.port):
            return False

        if not await self._terminate_process(pid):
            return False

        _unlink_pid(pidfile)
        self.release()
        return True

    async def status(self) -> ServerStatus:
        """Inspect running state and health of this instance's server from its pidfile.

        Probes ``spec.host`` — the host the server was configured to listen on — never a
        hardcoded loopback address, so a server bound to a non-loopback host still reports
        healthy.
        """
        pid = _read_pid_if_exists(self._pidfile())
        alive = _is_pid_alive(pid) if pid is not None else False
        healthy = False
        model_path = ""
        if alive:
            with contextlib.suppress(httpx.HTTPError, OSError):
                resp = await self._http.get(
                    f"http://{self.spec.host}:{self.spec.port}/props",
                    timeout=1.0,
                    headers=auth_headers(self.spec.api_key_file),
                )
                if resp.status_code == _HTTP_OK:
                    healthy = True
                    model_path = resp.json().get("model_path", "")
        return ServerStatus(port=self.spec.port, pid=pid, alive=alive, healthy=healthy, model_path=model_path)


async def supervise(
    manager: LocalServerManager,
    handle: ServerHandle,
    *,
    poll_s: float,
    stop: asyncio.Event,
) -> SupervisorExit:
    """Supervise a running server process, yielding on external GPU requests or exit."""
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_s)
                break
            except TimeoutError:
                pass

            if not _is_pid_alive(handle.pid):
                return SupervisorExit.SERVER_EXITED

            if manager.yield_requested_by():
                stopped = await manager.stop()
                if not stopped and _is_pid_alive(handle.pid):
                    msg = f"failed to stop server on port {manager.spec.port}"
                    raise RuntimeError(msg)
                return SupervisorExit.YIELDED

        stopped = await manager.stop()
        if not stopped and _is_pid_alive(handle.pid):
            msg = f"failed to stop server on port {manager.spec.port}"
            raise RuntimeError(msg)
        return SupervisorExit.STOPPED
    finally:
        if not _is_pid_alive(handle.pid):
            manager.release()
