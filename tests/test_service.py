"""Unit tests for `localmodels serve`'s restart-after-yield loop and config loading.

`run_serve` takes its `LocalServerManager` through an injectable `manager_factory` (dependency
injection, this package's own construction rule) so these tests exercise the restart loop
without a real llama-server subprocess or GPU -- `launcher.py`'s own test suite (B12.3) already
covers `LocalServerManager.start()`/`supervise()` against a real fake server process; this
module only owns the "build a fresh manager and go again after a YIELD" loop.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from localmodels.launcher import (
    Contention,
    LeaseConflictError,
    MemoryGuards,
    NotEnoughMemoryError,
    StartupYieldedError,
    SupervisorExit,
)
from localmodels.lease import GpuLease, QuietRequest
from localmodels.process import ServerHandle
from localmodels.service import ServeConfig, load_serve_config, run_serve

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


@pytest.mark.parametrize("exit_reason", [SupervisorExit.STOPPED, SupervisorExit.SERVER_EXITED])
def test_run_serve_returns_the_exit_of_a_run_that_did_not_yield(exit_reason: SupervisorExit) -> None:
    build_calls = []

    def factory(_config, _http):
        build_calls.append(1)
        return _FakeManager(_fake_handle())

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return exit_reason

    config = ServeConfig(port=8091, model_path=Path("/m.gguf"), project="test")

    result = asyncio.run(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event())
    )
    assert result is exit_reason
    assert len(build_calls) == 1


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


async def test_run_serve_under_refuse_restarts_after_a_yield_once_the_other_lease_clears(
    tmp_path: Path,
) -> None:
    """A yield is by definition followed by the other side's lease, so REFUSE must not turn the
    restart into a `LeaseConflictError` crash loop: the restart waits for that lease to clear."""
    other_path = tmp_path / "service_c.json"
    starts: list[float] = []
    yielded_at: list[float] = []
    outcomes = iter([SupervisorExit.YIELDED, SupervisorExit.STOPPED])

    def factory(_config, _http):
        return _ScriptedStartManager([], starts)

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        outcome = next(outcomes)
        if outcome is SupervisorExit.YIELDED:
            yielded_at.append(time.monotonic())
            other_path.write_text(
                json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": 1})
            )
            asyncio.get_running_loop().call_later(_OTHER_LEASE_HELD_S, other_path.unlink)
        return outcome

    config = ServeConfig(
        port=8091,
        model_path=Path("/m.gguf"),
        project="test",
        lease_dir=tmp_path,
        contention=Contention.REFUSE,
        poll_s=0.05,
    )
    result = await run_serve(
        config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event()
    )

    assert result is SupervisorExit.STOPPED
    assert len(starts) == 2  # a fresh manager was built and started for the restart
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


def test_load_serve_config_reads_the_server_table_and_defaults_the_rest(tmp_path: Path) -> None:
    toml_path = tmp_path / "serve.toml"
    toml_path.write_text(
        """
        [server]
        host = "172.17.0.1"
        port = 8091
        model_path = "/models/qwen3.gguf"
        project = "service_a"
        reserved_ports = [8080]
        contention = "wait"
        """
    )
    config = load_serve_config(toml_path)
    assert config.host == "172.17.0.1"
    assert config.port == 8091
    assert config.project == "service_a"
    assert config.reserved_ports == frozenset({8080})
    assert config.contention is Contention.WAIT

    toml_path.write_text('[server]\nport = 8080\nmodel_path = "/m.gguf"\nproject = "service_c"\n')
    config = load_serve_config(toml_path)
    assert config.host == "127.0.0.1"  # loopback unless asked
    assert config.contention is Contention.REFUSE
    assert config.reserved_ports == frozenset()


class _ScriptedStartManager:
    """A manager whose `start()` raises each scripted error in turn, then succeeds."""

    def __init__(self, errors: list[Exception], starts: list[float]) -> None:
        self._errors = errors
        self._starts = starts

    async def start(self) -> ServerHandle:
        self._starts.append(time.monotonic())
        if self._errors:
            raise self._errors.pop(0)
        return _fake_handle()


@pytest.mark.parametrize(
    "error",
    [
        StartupYieldedError("GPU lease requested by service_c; yielding"),
        LeaseConflictError("held by service_c"),
    ],
    ids=["yield-while-loading", "refused-at-start"],
)
async def test_run_serve_waits_instead_of_exiting_when_a_start_meets_the_other_lease(
    tmp_path: Path, error: Exception
) -> None:
    """G12 C-1 (D47): a lease request while the model loads, or a REFUSE at a start, used to exit
    the process -- and under systemd's `Restart=always` that is a 5-second restart loop. The
    service now waits for the other lease to clear, then starts again, in the same process."""
    other_path = tmp_path / "service_c.json"
    other_path.write_text(json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": 1}))
    asyncio.get_running_loop().call_later(_OTHER_LEASE_HELD_S, other_path.unlink)
    starts: list[float] = []
    errors = [error]

    def factory(_config, _http):
        return _ScriptedStartManager(errors, starts)

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return SupervisorExit.STOPPED

    config = ServeConfig(
        port=8091,
        model_path=Path("/m.gguf"),
        project="test",
        lease_dir=tmp_path,
        contention=Contention.REFUSE,
        poll_s=0.05,
    )
    result = await run_serve(
        config, manager_factory=factory, supervise_fn=fake_supervise, stop=asyncio.Event()
    )

    assert result is SupervisorExit.STOPPED
    assert len(starts) == 2
    assert starts[1] - starts[0] >= _OTHER_LEASE_HELD_S  # the second start waited for the lease


async def test_run_serve_returns_yielded_when_stopped_while_a_start_waits(tmp_path: Path) -> None:
    (tmp_path / "service_c.json").write_text(
        json.dumps({"pid": os.getpid(), "project": "service_c", "requested_at": 1})
    )
    starts: list[float] = []
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, stop.set)

    def factory(_config, _http):
        return _ScriptedStartManager([StartupYieldedError("yielding")], starts)

    async def fake_supervise(_manager, _handle, *, poll_s, stop):  # pragma: no cover - never reached
        return SupervisorExit.STOPPED

    config = ServeConfig(
        port=8091, model_path=Path("/m.gguf"), project="test", lease_dir=tmp_path, poll_s=0.05
    )
    result = await asyncio.wait_for(
        run_serve(config, manager_factory=factory, supervise_fn=fake_supervise, stop=stop), timeout=5
    )

    assert result is SupervisorExit.YIELDED
    assert len(starts) == 1


def test_stay_up_restarts_only_when_memory_and_the_quiet_cards_allow(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, server_oom_score_adj=500)
    stay_up = {"contention": Contention.STAY_UP, "guards": guards}
    with pytest.raises(ValidationError, match="resident_allowance_gib"):
        ServeConfig(port=8091, model_path=Path("/m.gguf"), project="svc", **stay_up)
    config = ServeConfig(
        port=8091, model_path=Path("/m.gguf"), project="svc", lease_dir=tmp_path, poll_s=0.02,
        resident_allowance_gib=20, **stay_up,
    )  # fmt: skip
    state = {"free": 25.0}  # below the floor plus the allowance (30 GiB)
    quiet = GpuLease(tmp_path, project="bench").write_quiet(
        QuietRequest(pid=os.getpid(), project="bench", requested_at=1, reason="benchmark")
    )
    starts: list[tuple[float, bool]] = []
    outcomes = iter([SupervisorExit.SERVER_EXITED, SupervisorExit.STOPPED])

    class StayUpManager(_FakeManager):
        killed_reason: str | None = "available memory fell below the floor"

        async def start(self) -> ServerHandle:
            starts.append((state["free"], quiet.exists()))
            return await super().start()

        def free_gib(self) -> float:
            return state["free"]

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return next(outcomes)

    async def relax() -> None:
        await asyncio.sleep(0.1)
        state["free"] = 40.0  # enough memory, but the quiet card is still there
        await asyncio.sleep(0.1)
        assert len(starts) == 1
        state["free"] = 25.0
        quiet.unlink()  # the card is gone, but there is not enough memory yet
        await asyncio.sleep(0.1)
        assert len(starts) == 1
        state["free"] = 40.0

    async def main() -> SupervisorExit:
        relaxing = asyncio.create_task(relax())
        result = await run_serve(
            config,
            manager_factory=lambda _c, _h: StayUpManager(_fake_handle()),
            supervise_fn=fake_supervise,
            stop=asyncio.Event(),
        )
        await relaxing
        return result

    assert asyncio.run(main()) is SupervisorExit.STOPPED
    assert starts == [(25.0, True), (40.0, False)]


def test_stay_up_refused_for_memory_waits_and_retries_instead_of_exiting(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, server_oom_score_adj=500, host_reserve_gib=8)
    config = ServeConfig(
        port=8091, model_path=Path("/m.gguf"), project="svc", lease_dir=tmp_path, poll_s=0.05,
        contention=Contention.STAY_UP, guards=guards, resident_allowance_gib=20,
    )  # fmt: skip
    starts: list[float] = []

    class RefusedTwiceThenFits(_FakeManager):
        killed_reason: str | None = None

        async def start(self) -> ServerHandle:
            starts.append(time.monotonic())
            if len(starts) < 3:  # floor + allowance (30 GiB) is there, the fit check's 25 + 8 GiB is not
                raise NotEnoughMemoryError("m.gguf needs 33.0 GiB but 31.0 GiB is available; not loading it")
            return await super().start()

        def free_gib(self) -> float:
            return 31.0

    async def fake_supervise(_manager, _handle, *, poll_s, stop):
        return SupervisorExit.STOPPED

    serve = run_serve(
        config,
        manager_factory=lambda _c, _h: RefusedTwiceThenFits(_fake_handle()),
        supervise_fn=fake_supervise,
        stop=asyncio.Event(),
    )
    assert asyncio.run(asyncio.wait_for(serve, timeout=5)) is SupervisorExit.STOPPED
    assert len(starts) == 3
    assert min(b - a for a, b in itertools.pairwise(starts)) >= config.poll_s  # paced, never a tight loop
