"""`localmodels serve`: run and supervise one configured llama-server instance.

Restarts after a yield once the other project's lease clears, without duplicating any waiting
logic: a YIELD only ever comes from `launcher.supervise` because another project's lease
requested the GPU, and building a fresh `LocalServerManager` and calling `start()` again
re-enters `_acquire_lease()`, whose `Contention.WAIT` already blocks until that lease clears.
This loop's only job is to notice the YIELD and go again.

`manager_factory`/`supervise_fn` are injectable (construction by injection, this package's own
rule) so the restart loop is testable without a real llama-server subprocess or GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from .hardware import inspect_hardware
from .launcher import Contention, LocalServerManager, SupervisorExit, supervise
from .lease import GpuLease
from .params import derive_serving_params
from .process import ServerSpec, resolve_llama_server_bin

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .process import ServerHandle

# Matches the shared GPU-lease protocol doc (service_c/docs/2026-09-15-gpu-lease-protocol.md):
# "Polls every 30 s" -- service_a's half of the contract this package now implements.
_DEFAULT_POLL_S = 30.0
_DEFAULT_LEASE_DIR = Path("~/.local/state/gpu-lease").expanduser()
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
    lease_dir: Path = _DEFAULT_LEASE_DIR
    reserved_ports: frozenset[int] = Field(default_factory=frozenset)
    contention: Contention = Contention.REFUSE
    binary: Path | None = None
    pid_dir: Path = _DEFAULT_PID_DIR
    ready_timeout_s: float = 60.0
    poll_s: float = _DEFAULT_POLL_S
    device_override: str | None = None
    extra_flags: tuple[str, ...] = ()
    api_key_file: Path | None = None


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
    )


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
    is never returned to the caller except when `stop` was already set by the time it happened,
    so a caller asking this to stop is honoured even mid-yield rather than silently restarting
    once more.
    """
    stop_event = stop if stop is not None else asyncio.Event()
    factory = manager_factory or _build_manager
    owns_http = http is None
    owned_http = http or httpx.AsyncClient()
    try:
        while True:
            manager = factory(config, owned_http)
            handle: ServerHandle = await manager.start()
            exit_reason = await supervise_fn(manager, handle, poll_s=config.poll_s, stop=stop_event)
            if exit_reason is not SupervisorExit.YIELDED or stop_event.is_set():
                return exit_reason
    finally:
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
