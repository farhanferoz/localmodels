"""Unit tests for `localmodels serve`'s restart-after-yield loop and config loading.

`run_serve` takes its `LocalServerManager` through an injectable `manager_factory` (dependency
injection, this package's own construction rule) so these tests exercise the restart loop
without a real llama-server subprocess or GPU -- `launcher.py`'s own test suite (B12.3) already
covers `LocalServerManager.start()`/`supervise()` against a real fake server process; this
module only owns the "build a fresh manager and go again after a YIELD" loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localmodels.launcher import Contention, SupervisorExit
from localmodels.process import ServerHandle
from localmodels.service import ServeConfig, load_serve_config, run_serve


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


def test_run_serve_restarts_after_a_yield_and_returns_final_exit() -> None:
    build_calls = []
    outcomes = iter([SupervisorExit.YIELDED, SupervisorExit.STOPPED])

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return next(outcomes)

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test")

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
