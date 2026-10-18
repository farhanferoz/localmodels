"""`localmodels serve`: run and supervise one configured llama-server instance.

Restarts after a yield once the other project's lease clears. A YIELD only ever comes from
`launcher.supervise` because another project's lease requested the GPU, so the other side's
lease is by definition live when the loop goes round again. `_acquire_lease()` cannot be left to
wait for it: under `Contention.REFUSE` (which governs a start that meets a lease unannounced)
it would raise. So this loop waits itself, polling `GpuLease.live_others()`, and only then
builds a fresh `LocalServerManager` and calls `start()`.

`manager_factory`/`supervise_fn` are injectable (construction by injection, this package's own
rule) so the restart loop is testable without a real llama-server subprocess or GPU.

Under ``Contention.STAY_UP`` the server stays up beside other projects' leases, so a yield (to a quiet card)
or a floor kill is followed by a different wait: until no quiet request remains and available memory is at
least the kill floor plus the service's allowance.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, model_validator

from .hardware import inspect_hardware
from .launcher import (
    Contention,
    LeaseConflictError,
    LocalServerManager,
    MemoryGuards,
    ServerStartError,
    StartupYieldedError,
    SupervisorExit,
    stay_up_problem,
    supervise,
)
from .lease import DEFAULT_LEASE_DIR, GpuLease
from .params import derive_serving_params
from .process import ServerSpec, resolve_llama_server_bin

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .process import ServerHandle

# Matches the shared GPU-lease protocol doc (service_c/docs/2026-09-15-gpu-lease-protocol.md):
# "Polls every 30 s" -- service_a's half of the contract this package now implements.
_DEFAULT_POLL_S = 30.0
_DEFAULT_PID_DIR = Path("~/.local/state/localmodels/pids").expanduser()

EXIT_OK = 0
EXIT_ERROR = 1


class ServeConfig(BaseModel):
    """One configured server: what `LocalServerManager` needs, minus anything measured live.

    Serving parameters (`n_gpu_layers`/`n_ctx`/`n_parallel`) are never part of this config --
    code standard rule 8 ("hardware-aware, never hardware-keyed"): they are derived fresh from
    measured hardware each time `run_serve` (re)starts the server, not pinned to a config value
    that could go stale when the host changes.
    """

    host: str = "127.0.0.1"
    port: int
    model_path: Path
    project: str
    lease_dir: Path = DEFAULT_LEASE_DIR
    reserved_ports: frozenset[int] = Field(default_factory=frozenset)
    contention: Contention = Contention.REFUSE
    binary: Path | None = None
    pid_dir: Path = _DEFAULT_PID_DIR
    ready_timeout_s: float = 60.0
    poll_s: float = _DEFAULT_POLL_S
    device_override: str | None = None
    extra_flags: tuple[str, ...] = ()
    extra_env: dict[str, str] | None = None
    api_key_file: Path | None = None
    guards: MemoryGuards = Field(default_factory=MemoryGuards)
    #: Memory a stay-up service holds all the time; required with ``Contention.STAY_UP``.
    resident_allowance_gib: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _stay_up_needs_its_guards(self) -> ServeConfig:
        if self.contention is Contention.STAY_UP and (
            problem := stay_up_problem(self.guards, self.resident_allowance_gib)
        ):
            raise ValueError(problem)
        return self


def load_serve_config(path: Path) -> ServeConfig:
    """Load a `ServeConfig` from a TOML file's `[server]` table."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return ServeConfig(**data.get("server", data))


def _build_manager(config: ServeConfig, http: httpx.AsyncClient) -> LocalServerManager:
    hardware = inspect_hardware(config.device_override)
    params = derive_serving_params(hardware.device, hardware.total_memory_gb)
    binary = config.binary or resolve_llama_server_bin()
    lease = GpuLease(config.lease_dir, project=config.project)
    return LocalServerManager(
        ServerSpec(
            host=config.host,
            port=config.port,
            model_path=config.model_path,
            api_key_file=config.api_key_file,
        ),
        params,
        lease,
        reserved_ports=config.reserved_ports,
        contention=config.contention,
        ready_timeout_s=config.ready_timeout_s,
        http=http,
        pid_dir=config.pid_dir,
        binary=binary,
        extra_flags=config.extra_flags,
        extra_env=config.extra_env,
        guards=config.guards,
        resident_allowance_gib=config.resident_allowance_gib,
    )


async def _wait_for_other_leases(lease: GpuLease, *, poll_s: float, stop: asyncio.Event) -> bool:
    """Poll until no other live lease remains; False if `stop` was set while waiting."""
    while lease.live_others():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_s)
        if stop.is_set():
            return False
    return True


async def _wait_for_room(
    manager: LocalServerManager, config: ServeConfig, lease: GpuLease, *, stop: asyncio.Event
) -> bool:
    """Poll until no quiet request remains and memory is floor + allowance; False if `stop` is set."""
    assert config.guards.kill_below_gib is not None  # noqa: S101 - narrowing for mypy; the config checked
    assert config.resident_allowance_gib is not None  # noqa: S101 - narrowing for mypy; the config checked
    needed = config.guards.kill_below_gib + config.resident_allowance_gib
    while True:
        free = manager.free_gib()
        if not lease.quiet_requests() and free is not None and free >= needed:
            return True
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=config.poll_s)
        if stop.is_set():
            return False


async def _wait_to_restart(
    manager: LocalServerManager, config: ServeConfig, lease: GpuLease, *, stop: asyncio.Event
) -> bool:
    """Wait for the condition that lets the server start again; False if `stop` was set while waiting."""
    if config.contention is Contention.STAY_UP:
        return await _wait_for_room(manager, config, lease, stop=stop)
    return await _wait_for_other_leases(lease, poll_s=config.poll_s, stop=stop)


def _waits_before_restart(
    failure: ServerStartError | LeaseConflictError, manager: LocalServerManager, *, stay_up: bool
) -> bool:
    """A yielded start, or a stay-up server killed below its floor, waits and restarts; the rest raise."""
    if isinstance(failure, StartupYieldedError | LeaseConflictError):
        return True
    return stay_up and manager.killed_reason is not None


def _restarts_after(exit_reason: SupervisorExit, manager: LocalServerManager, *, stay_up: bool) -> bool:
    """A yield or a stay-up floor kill is followed by a wait and a restart; other exits end the service."""
    floor_kill = stay_up and exit_reason is SupervisorExit.SERVER_EXITED and manager.killed_reason is not None
    return exit_reason is SupervisorExit.YIELDED or floor_kill


async def run_serve(
    config: ServeConfig,
    *,
    stop: asyncio.Event | None = None,
    http: httpx.AsyncClient | None = None,
    manager_factory: Callable[[ServeConfig, httpx.AsyncClient], LocalServerManager] | None = None,
    supervise_fn: Callable[..., Awaitable[SupervisorExit]] = supervise,
) -> SupervisorExit:
    """Serve `config`'s model until stopped, restarting after each yield.

    Returns the final, non-YIELDED `SupervisorExit` (`STOPPED` or `SERVER_EXITED`) -- a YIELD
    is never returned to the caller except when `stop` was set by the time it happened or while
    waiting for the other project's lease to clear, so a caller asking this to stop is honoured
    even mid-yield rather than silently restarting once more. Under `Contention.STAY_UP` a floor kill
    restarts the same way, and the `resident/` record is removed when the service ends.
    """
    stop_event = stop if stop is not None else asyncio.Event()
    lease = GpuLease(config.lease_dir, project=config.project)
    factory = manager_factory or _build_manager
    stay_up = config.contention is Contention.STAY_UP
    owns_http = http is None
    owned_http = http or httpx.AsyncClient()
    try:
        while True:
            manager = factory(config, owned_http)
            try:
                handle: ServerHandle = await manager.start()
            except (StartupYieldedError, LeaseConflictError, ServerStartError) as exc:
                # Another project holds or asked for the GPU as this start ran: a service waits
                # for its lease to clear rather than exiting into a restart loop (G12 C-1, D47).
                # `Contention.REFUSE` still governs a one-shot start, which never reaches here.
                if not _waits_before_restart(exc, manager, stay_up=stay_up):
                    raise
                if stop_event.is_set():
                    return SupervisorExit.YIELDED
                if not await _wait_to_restart(manager, config, lease, stop=stop_event):
                    return SupervisorExit.YIELDED
                continue
            exit_reason = await supervise_fn(manager, handle, poll_s=config.poll_s, stop=stop_event)
            if not _restarts_after(exit_reason, manager, stay_up=stay_up) or stop_event.is_set():
                return exit_reason
            if not await _wait_to_restart(manager, config, lease, stop=stop_event):
                return exit_reason
    finally:
        if stay_up:
            lease.remove_resident()
        if owns_http:
            await owned_http.aclose()


def _cmd_serve(config_path: Path) -> int:
    """Run `localmodels serve` until a SIGTERM/SIGINT asks it to stop cleanly."""
    config = load_serve_config(config_path)

    async def _main() -> SupervisorExit:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop_event.set)
        return await run_serve(config, stop=stop_event)

    result = asyncio.run(_main())
    return EXIT_OK if result is SupervisorExit.STOPPED else EXIT_ERROR
