"""Unit tests for `localmodels serve`'s restart-after-yield loop and config loading.

`run_serve` takes its `LocalServerManager` through an injectable `manager_factory` (dependency
injection, this package's own construction rule) so these tests exercise the restart loop
without a real llama-server subprocess or GPU -- `launcher.py`'s own test suite (B12.3) already
covers `LocalServerManager.start()`/`supervise()` against a real fake server process; this
module only owns the "build a fresh manager and go again after a YIELD" loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from localmodels.launcher import Contention, SupervisorExit
from localmodels.lease import GpuLease
from localmodels.process import ServerHandle
from localmodels.service import ServeConfig, load_serve_config, run_serve
from tests.test_launcher import _gpu_params, _make_fake_server, _manager

_OTHER_LEASE_HELD_S = 0.3


class _FakeManager:
    def __init__(self, handle: ServerHandle) -> None:
        self._handle = handle
        self.started = False

    async def start(self) -> ServerHandle:
        self.started = True
        return self._handle


def _fake_handle() -> ServerHandle:
    return ServerHandle(pid=1, port=8091, model_path=Path("/m.gguf"))


def test_run_serve_returns_stopped_without_a_yield() -> None:
    build_calls = []

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return SupervisorExit.STOPPED

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test")

    result = asyncio.run(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event())
    )
    assert result is SupervisorExit.STOPPED
    assert len(build_calls) == 1


def test_run_serve_restarts_after_a_yield_and_returns_final_exit(tmp_path: Path) -> None:
    build_calls = []
    outcomes = iter([SupervisorExit.YIELDED, SupervisorExit.STOPPED])

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return next(outcomes)

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test", lease_dir=tmp_path)

    result = asyncio.run(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event())
    )
    assert result is SupervisorExit.STOPPED
    assert len(build_calls) == 2  # a fresh manager was built for the restart


def test_run_serve_stops_immediately_if_stop_is_already_set_after_a_yield() -> None:
    build_calls = []
    outcomes = iter([SupervisorExit.YIELDED])

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    stop = asyncio.Event()

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        stop.set()
        return next(outcomes)

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test")

    result = asyncio.run(run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=stop))
    assert result is SupervisorExit.YIELDED
    assert len(build_calls) == 1  # never restarted -- the caller asked to stop


async def test_run_serve_under_refuse_waits_for_the_other_lease_after_a_yield(tmp_path: Path) -> None:
    """A yield is by definition followed by the other side's lease, so REFUSE must not turn the
    restart into a `LeaseConflictError` crash loop: the restart waits for that lease to clear."""
    bin_path = _make_fake_server(tmp_path)
    model_file = tmp_path / "test.gguf"
    model_file.touch()
    lease_dir = tmp_path / "lease"
    other_path = lease_dir / "service_c.json"
    lease = GpuLease(lease_dir, project="localmodels-test")
    yielded_at = []
    starts = []
    managers = []

    async with httpx.AsyncClient() as http:

        def factory(_config, _http):
            manager = _manager(
                tmp_path=tmp_path, http=http, bin_path=bin_path, port=8198, model_file=model_file,
                params=_gpu_params(), contention=Contention.REFUSE, lease=lease,
            )
            real_start = manager.start

            async def counting_start():
                starts.append(time.monotonic())
                return await real_start()

            manager.start = counting_start
            managers.append(manager)
            return manager

        outcomes = iter([SupervisorExit.YIELDED, SupervisorExit.STOPPED])

        async def fake_supervise(manager, _handle, *, poll_s, stop):
            outcome = next(outcomes)
            if outcome is SupervisorExit.YIELDED:
                await manager.stop()  # what `supervise` does before returning YIELDED
                yielded_at.append(time.monotonic())
                lease_dir.mkdir(parents=True, exist_ok=True)
                other_path.write_text(
                    json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": 1})
                )
                asyncio.get_running_loop().call_later(_OTHER_LEASE_HELD_S, other_path.unlink)
            return outcome

        config = ServeConfig(
            port=8198, model_path=model_file, project="localmodels-test", lease_dir=lease_dir,
            contention=Contention.REFUSE, poll_s=0.05,
        )
        try:
            result = await run_serve(
                config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event(), http=http
            )
        finally:
            await managers[-1].stop()

    assert result is SupervisorExit.STOPPED
    assert len(starts) == 2
    assert starts[1] - yielded_at[0] >= _OTHER_LEASE_HELD_S  # the restart waited for the lease to clear


async def test_run_serve_honours_stop_while_waiting_for_the_other_lease(tmp_path: Path) -> None:
    (tmp_path / "service_c.json").write_text(
        json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": 1})
    )
    build_calls = []
    stop = asyncio.Event()

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        asyncio.get_running_loop().call_later(0.1, stop.set)
        return SupervisorExit.YIELDED

    config = ServeConfig(
        port=8091, model_path=Path("/m.gguf"), project="test", lease_dir=tmp_path, poll_s=0.05
    )

    result = await asyncio.wait_for(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=stop), timeout=5
    )
    assert result is SupervisorExit.YIELDED
    assert len(build_calls) == 1  # asked to stop while waiting: never restarted


def test_run_serve_propagates_server_exited() -> None:
    def factory(_config, _http):
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return SupervisorExit.SERVER_EXITED

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test")

    result = asyncio.run(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event())
    )
    assert result is SupervisorExit.SERVER_EXITED


def test_serve_config_defaults() -> None:
    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="service_a")
    assert config.host == "127.0.0.1"
    assert config.contention is Contention.REFUSE
    assert config.reserved_ports == frozenset()
    assert config.poll_s == pytest.approx(30.0)


def test_load_serve_config_reads_server_table(tmp_path: Path) -> None:
    toml_path = tmp_path / "serve.toml"
    toml_path.write_text(
        """
        [server]
        host = "172.17.0.1"
        port = 8091
        model_path = "/models/qwen3.gguf"
        project = "service_a"
        reserved_ports = [8080]
        contention = "refuse"
        """
    )
    config = load_serve_config(toml_path)
    assert config.host == "172.17.0.1"
    assert config.port == 8091
    assert config.project == "service_a"
    assert config.reserved_ports == frozenset({8080})
    assert config.contention is Contention.REFUSE


def test_load_serve_config_wait_contention(tmp_path: Path) -> None:
    toml_path = tmp_path / "serve.toml"
    toml_path.write_text(
        """
        [server]
        port = 8080
        model_path = "/models/service_c.gguf"
        project = "service_c"
        contention = "wait"
        """
    )
    config = load_serve_config(toml_path)
    assert config.contention is Contention.WAIT
