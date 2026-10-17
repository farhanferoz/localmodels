"""The launcher's memory guards, with fake memory readings and a fake server; no GPU, no model."""

from __future__ import annotations

import asyncio
import os
import stat
from typing import TYPE_CHECKING

import httpx
import pytest

from localmodels.launcher import LocalServerManager, MemoryGuards, ServerStartError
from localmodels.lease import GpuLease
from localmodels.process import ServerSpec
from tests.test_launcher import _gpu_params, _make_fake_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PORT = 8187
PROJECTED_GIB = 20  # what the fake llama-fit-params prints (20480 MiB)


def _fit_params(bin_path: Path, output: str = "projected to use 20480 MiB of device memory") -> None:
    script = bin_path.with_name("llama-fit-params")
    script.write_text(f"#!/bin/sh\necho '{output}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _manager(
    tmp_path: Path,
    http: httpx.AsyncClient,
    guards: MemoryGuards,
    readings: Iterator[float],
    *,
    health: int = 3,
) -> LocalServerManager:
    model = tmp_path / "test.gguf"
    model.touch()
    bin_path = _make_fake_server(tmp_path, min_health_attempts=health)
    _fit_params(bin_path)
    last = [0.0]

    def available(_on_gpu: bool) -> float:
        last[0] = next(readings, last[0])
        return last[0]

    return LocalServerManager(
        ServerSpec(host="127.0.0.1", port=PORT, model_path=model),
        _gpu_params(),
        GpuLease(tmp_path / "lease", project="localmodels-test"),
        http=http,
        pid_dir=tmp_path / "pids",
        binary=bin_path,
        stop_grace_s=1.0,
        ready_timeout_s=10.0,
        guards=guards,
        available_gib=available,
    )


def _steady(gib: float) -> Iterator[float]:
    while True:
        yield gib


async def test_load_refused_when_projection_plus_reserve_exceeds_available(tmp_path: Path) -> None:
    guards = MemoryGuards(host_reserve_gib=12)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(PROJECTED_GIB + 11))
        with pytest.raises(ServerStartError, match=r"needs 32\.0 GiB .* 31\.0 GiB is available"):
            await manager.start()
        assert not (tmp_path / "pids" / f"{PORT}.pid").exists()


async def test_load_refused_when_there_is_no_projection(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, MemoryGuards(host_reserve_gib=12), _steady(500))
        _fit_params(tmp_path / "bin" / "llama-server", "nothing useful")
        with pytest.raises(ServerStartError, match="printed no projection"):
            await manager.start()


async def test_server_that_leaves_less_than_the_reserve_is_stopped(tmp_path: Path) -> None:
    readings = iter([100.0, 5.0])  # before the load it fits; after, 5 GiB is left
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, MemoryGuards(host_reserve_gib=12), readings)
        with pytest.raises(ServerStartError, match=r"left 5\.0 GiB .* under the 12 GiB reserve"):
            await manager.start()
        assert not (tmp_path / "pids" / f"{PORT}.pid").exists()


async def test_kill_floor_during_the_load(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, load_poll_s=0.05)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(4), health=10_000)
        with pytest.raises(
            ServerStartError, match=r"killed while loading: available memory fell to 4\.0 GiB"
        ):
            await manager.start()


async def test_kill_floor_watchdog_while_running(tmp_path: Path) -> None:
    guards = MemoryGuards(kill_below_gib=10, load_poll_s=0.05, watch_poll_s=0.05)
    level = [50.0]

    def readings() -> Iterator[float]:
        while True:
            yield level[0]

    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, readings())
        await manager.start()
        assert (await manager.status()).alive
        level[0] = 3.0
        for _ in range(100):
            await asyncio.sleep(0.05)
            if not (await manager.status()).alive:
                break
        assert not (await manager.status()).alive
        assert manager.killed_reason is not None


async def test_server_oom_score_adj_is_set_on_the_server_alone(tmp_path: Path) -> None:
    guards = MemoryGuards(server_oom_score_adj=700)
    async with httpx.AsyncClient() as http:
        manager = _manager(tmp_path, http, guards, _steady(500))
        handle = await manager.start()
        try:
            with open(f"/proc/{handle.pid}/oom_score_adj") as fh:  # noqa: ASYNC230,PTH123
                assert fh.read().strip() == "700"
            with open("/proc/self/oom_score_adj") as fh:  # noqa: ASYNC230,PTH123
                assert fh.read().strip() != "700"
            assert os.getpgid(handle.pid) != os.getpgid(0)
        finally:
            await manager.stop()
