"""Local llama-server process lifecycle management and supervision, torch-free.

One ``LocalServerManager`` owns one ``(spec, params, lease)`` server instance. ``Contention``
lets a caller choose service_a's policy (``REFUSE``: never start while another project holds
the GPU) or service_c's (``WAIT``: queue for it, as ``run_gpu_queue.sh`` does) without either
project's launcher code living here — this package exposes the choice, not a default project.
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
from .lease import GpuLease, GpuLeaseClient, LeaseRecord
from .process import (
    LocalModelsError,
    ServerHandle,
    ServerSpec,
    ServerStatus,
    _is_pid_alive,
    _process_args,
    auth_headers,
    build_args,
    require_key_off_loopback,
)
from .stack import Engine, LlamaLayout, ServerStamp, fingerprint, llama_cpp_build

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .params import ServingParams

__all__ = [
    "Contention",
    "LeaseConflictError",
    "LeaseWaitTimeoutError",
    "LocalServerManager",
    "MemoryGuards",
    "PortHeldError",
    "ReservedPortError",
    "ServerStartError",
    "StartupYieldedError",
    "SupervisorExit",
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

log = logging.getLogger(__name__)


class Contention(StrEnum):
    """What ``LocalServerManager.start()`` does when another live lease holds the GPU."""

    REFUSE = "refuse"
    WAIT = "wait"


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


def _available_gib(on_gpu: bool) -> float | None:
    """Available memory in GiB for a GPU or CPU server (``available_memory``); None when unreadable."""
    reading = available_memory("cuda" if on_gpu else "cpu")
    return None if reading is None else reading.bytes / _GIB


class SupervisorExit(StrEnum):
    """Exit reason of the local model server supervisor."""

    STOPPED = "stopped"
    SERVER_EXITED = "server_exited"
    YIELDED = "yielded"


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
    """Manages lifecycle, health verification, and process isolation of one local llama-server."""

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
        available_gib: Callable[[bool], float | None] = _available_gib,
    ) -> None:
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
        self.guards = guards or MemoryGuards()
        self._available_gib = available_gib
        self._watchdog: asyncio.Task[None] | None = None
        self._ready = False
        #: Why the watchdog killed the last server it started, in plain words; None if it did not.
        self.killed_reason: str | None = None

    def _pidfile(self) -> Path:
        return self.pid_dir / f"{self.spec.port}.pid"

    def stamp(self, model_sha256: str | None = None) -> ServerStamp:
        """How this server is made: llama.cpp's build, the model file, its layout and the host.

        ``model_sha256`` is the file's hash when the caller knows it (a ``PinnedFile``); never computed here.
        """
        return ServerStamp(
            engine=Engine.LLAMA_CPP,
            build=llama_cpp_build(Path(self.binary)),
            model_path=str(self.spec.model_path),
            model_sha256=model_sha256,
            layout=LlamaLayout(
                n_ctx=self.params.n_ctx,
                n_parallel=self.params.n_parallel,
                kv_unified=self.params.kv_unified,
                extra_flags=self.extra_flags,
            ),
            host=fingerprint(),
        )

    def _free_gib(self) -> float | None:
        return self._available_gib(self.params.n_gpu_layers != 0)

    async def _projected_gib(self) -> float:
        """llama.cpp's projection of the server's memory (weights, context, compute); raises if none."""
        fit = self.guards.fit_params_binary or Path(self.binary).with_name("llama-fit-params")
        cmd = [
            str(fit),
            "-m",
            str(self.spec.model_path),
            "-c",
            str(self.params.n_ctx),
            "-np",
            str(self.params.n_parallel),
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
                f"llama-fit-params printed no projection for {self.spec.model_path.name} "
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
        free = self._free_gib()
        if free is None:
            msg = "available memory cannot be read, so the model's fit cannot be checked; not loading it"
            raise ServerStartError(msg)
        if need > free:
            msg = (
                f"{self.spec.model_path.name} needs {need:.1f} GiB (llama.cpp's projection plus a "
                f"{reserve:g} GiB host reserve) but {free:.1f} GiB is available; not loading it"
            )
            raise ServerStartError(msg)

    async def _check_left_over(self) -> None:
        """After the load: stop the server if it left less than the host reserve available."""
        reserve = self.guards.host_reserve_gib
        free = self._free_gib()
        if reserve is None or free is None or free >= reserve:
            return
        await self.stop()
        msg = (
            f"the loaded server left {free:.1f} GiB available, under the {reserve:g} GiB reserve; stopped it"
        )
        raise ServerStartError(msg)

    async def _watch_memory(self, pid: int, floor: float) -> None:
        """SIGKILL the server's process group the moment available memory falls below ``floor``."""
        while _is_pid_alive(pid):
            await asyncio.sleep(self.guards.watch_poll_s if self._ready else self.guards.load_poll_s)
            free = self._free_gib()
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
                    return ServerHandle(pid=pid, port=self.spec.port, model_path=self.spec.model_path)
        return None

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

    async def _poll_ready(
        self, proc: asyncio.subprocess.Process, log_path: Path, lease_written: bool
    ) -> None:
        deadline = time.time() + self.ready_timeout_s
        while time.time() < deadline:
            if proc.returncode is not None:
                await self.stop()
                if self.killed_reason is not None:
                    msg = f"llama-server was killed while loading: {self.killed_reason}"
                    raise ServerStartError(msg)
                msg = f"llama-server exited prematurely with code {proc.returncode}; see {log_path}"
                raise ServerStartError(msg)

            if lease_written:
                other = self.lease.requested_by_other()
                if other:
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
            if self.params.n_gpu_layers != 0:
                self._write_lease()
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
        args = build_args(self.binary, self.spec, self.params, self.extra_flags)
        if self.guards.server_oom_score_adj is not None:
            args = ["sh", "-c", _OOM_WRAPPER.format(score=self.guards.server_oom_score_adj), *args]
        await self._check_fits()

        is_gpu = self.params.n_gpu_layers != 0
        lease_written = False
        if is_gpu:
            await self._acquire_lease()
            lease_written = True

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
            await self._poll_ready(proc, log_path, lease_written)
            await self._verify_props()
            await self._check_left_over()
            self._ready = True
        except Exception:
            self._cancel_watchdog()
            if lease_written:
                self.lease.remove()
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
            self.lease.remove()
            return False

        args = _process_args(pid)
        if not _is_llama_server_on_port(args, self.spec.port):
            return False

        if not await self._terminate_process(pid):
            return False

        _unlink_pid(pidfile)
        self.lease.remove()
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

            # A server with no GPU layers holds nothing another project's GPU work needs.
            requested = manager.params.n_gpu_layers != 0 and manager.lease.requested_by_other()
            if requested:
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
            manager.lease.remove()
