"""Local llama-server process lifecycle management and supervision, torch-free.

One ``LocalServerManager`` owns one ``(spec, params, lease)`` server instance. ``Contention``
lets a caller choose service_a's policy (``REFUSE``: never start while another project holds
the GPU) or service_c's (``WAIT``: queue for it, as ``run_gpu_queue.sh`` does) without either
project's launcher code living here — this package exposes the choice, not a default project.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import httpx

from .lease import GpuLease, LeaseRecord
from .process import (
    LocalModelsError,
    ServerHandle,
    ServerSpec,
    ServerStatus,
    _is_pid_alive,
    _process_args,
    build_args,
)

if TYPE_CHECKING:
    from .params import ServingParams

__all__ = [
    "Contention",
    "LeaseConflictError",
    "LeaseWaitTimeoutError",
    "LocalServerManager",
    "PortHeldError",
    "ReservedPortError",
    "ServerStartError",
    "SupervisorExit",
    "supervise",
]

_HTTP_OK = 200


class Contention(StrEnum):
    """What ``LocalServerManager.start()`` does when another live lease holds the GPU."""

    REFUSE = "refuse"
    WAIT = "wait"


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
        stop_grace_s: float = 20.0,
        wait_poll_s: float = 2.0,
        wait_timeout_s: float | None = None,
        min_slot_tokens: int = 0,
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
        self.stop_grace_s = stop_grace_s
        self.wait_poll_s = wait_poll_s
        self.wait_timeout_s = wait_timeout_s
        self.min_slot_tokens = min_slot_tokens

    def _pidfile(self) -> Path:
        return self.pid_dir / f"{self.spec.port}.pid"

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
            resp = await self._http.get(f"http://{self.spec.host}:{self.spec.port}/props", timeout=2.0)
            if resp.status_code == _HTTP_OK:
                data = resp.json()
                if Path(data.get("model_path", "")).name == self.spec.model_path.name:
                    return ServerHandle(pid=pid, port=self.spec.port, model_path=self.spec.model_path)
        return None

    def _check_port_free(self) -> None:
        try:
            with socket.create_connection((self.spec.host, self.spec.port), timeout=0.2):
                msg = (
                    f"port {self.spec.port} is held by a process this manager did not start; "
                    "free it by hand"
                )
                raise PortHeldError(msg)
        except OSError:
            pass

    async def _acquire_lease(self) -> None:
        """Write this instance's GPU lease, honouring ``Contention`` against other live leases."""
        deadline = time.time() + self.wait_timeout_s if self.wait_timeout_s is not None else None
        while True:
            others = self.lease.live_others()
            if not others:
                break
            if self.contention is Contention.REFUSE:
                holders = ", ".join(o.project for o in others)
                msg = f"cannot acquire GPU lease; currently held by: {holders}"
                raise LeaseConflictError(msg)
            if deadline is not None and time.time() > deadline:
                msg = f"timed out waiting {self.wait_timeout_s}s for the GPU lease"
                raise LeaseWaitTimeoutError(msg)
            await asyncio.sleep(self.wait_poll_s)

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
                msg = f"llama-server exited prematurely with code {proc.returncode}; see {log_path}"
                raise ServerStartError(msg)

            if lease_written:
                other = self.lease.requested_by_other()
                if other:
                    await self.stop()
                    msg = f"GPU lease requested by {other}; yielding"
                    raise ServerStartError(msg)

            with contextlib.suppress(httpx.HTTPError, OSError):
                resp = await self._http.get(f"http://{self.spec.host}:{self.spec.port}/health", timeout=1.0)
                if resp.status_code == _HTTP_OK:
                    return
            await asyncio.sleep(0.5)

        await self.stop()
        msg = f"llama-server failed to become ready within {self.ready_timeout_s}s; see {log_path}"
        raise ServerStartError(msg)

    async def _verify_props(self) -> None:
        if self.min_slot_tokens <= 0:
            return
        try:
            resp = await self._http.get(f"http://{self.spec.host}:{self.spec.port}/props", timeout=5.0)
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

        _prepare_pid_dir(self.pid_dir)
        pidfile = self._pidfile()

        adopted = await self._try_adopt(pidfile)
        if adopted is not None:
            return adopted

        self._check_port_free()

        is_gpu = self.params.n_gpu_layers != 0
        lease_written = False
        if is_gpu:
            await self._acquire_lease()
            lease_written = True

        log_path = self.pid_dir / f"{self.spec.port}.log"
        args = build_args(self.binary, self.spec, self.params, self.extra_flags)

        with _open_log(log_path) as log_file:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

        _write_pid(pidfile, proc.pid)

        try:
            await self._poll_ready(proc, log_path, lease_written)
            await self._verify_props()
        except Exception:
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
                resp = await self._http.get(f"http://{self.spec.host}:{self.spec.port}/props", timeout=1.0)
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

            requested = manager.lease.requested_by_other()
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
