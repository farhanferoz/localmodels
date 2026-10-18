"""``ModelHost`` in stay-up mode: loaded beside another lease, unloaded for a quiet card or the floor."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from localmodels.lease import GpuLease, GpuLeaseClient, QuietRequest
from localmodels.serving import LoadMessage, ModelHost
from localmodels.serving.host import LOW_MEMORY, StayUp
from tests.test_serving import ECHO, ECHO_ARGV, ECHO_ENV

if TYPE_CHECKING:
    from collections.abc import Callable

STAY_UP = StayUp(floor_gib=10, allowance_gib=20, oom_score_adj=500)
LOAD = LoadMessage(model_dir="/m", revision="r1")


def _host(
    tmp_path: Path, available_gib: Callable[[], float | None], *, stay_up: StayUp | None = STAY_UP
) -> ModelHost:
    return ModelHost(
        GpuLeaseClient(GpuLease(tmp_path, project="service_b")),
        idle_unload_s=300,
        poll_s=0.01,
        stay_up=stay_up,
        available_gib=available_gib,
    )


def _acquire(host: ModelHost, *, give_up_after_waits: int = 0) -> bool:
    waits: list[object] = []
    return host.acquire(
        job_id="j",
        model_path="/m",
        should_stop=lambda: len(waits) > give_up_after_waits,
        on_wait=waits.append,
    )


def _resident_loaded(tmp_path: Path) -> bool | None:
    path = tmp_path / "resident" / "service_b-0.json"
    return json.loads(path.read_text())["loaded"] if path.exists() else None


def _quiet_card(tmp_path: Path) -> Path:
    return GpuLease(tmp_path, project="bench").write_quiet(
        QuietRequest(pid=os.getpid(), project="bench", requested_at=int(time.time()), reason="t")
    )


def test_without_stay_up_the_host_takes_the_lease_and_yields_to_an_ordinary_request(tmp_path: Path) -> None:
    host = _host(tmp_path, lambda: None, stay_up=None)
    assert _acquire(host)
    assert (tmp_path / "service_b.json").exists()
    assert not (tmp_path / "resident").exists()
    (tmp_path / "waiter.json").write_text(
        json.dumps({"pid": os.getpid(), "project": "waiter", "requested_at": 1})
    )
    assert host.yield_if_requested() == "waiter"
    assert not (tmp_path / "service_b.json").exists()


def test_a_stay_up_host_loads_while_another_project_holds_the_lease_and_ignores_requests(
    tmp_path: Path,
) -> None:
    (tmp_path / "other.json").write_text(json.dumps({"pid": os.getpid(), "project": "other"}))
    host = _host(tmp_path, lambda: 40.0)
    assert _acquire(host)
    assert not (tmp_path / "service_b.json").exists()
    assert _resident_loaded(tmp_path) is True
    server = host.server(ECHO, ECHO_ARGV, ECHO_ENV, LOAD, lambda: False)
    try:
        assert (Path("/proc") / str(server.pid) / "oom_score_adj").read_text().strip() == "500"
        (tmp_path / "waiter.json").write_text(
            json.dumps({"pid": os.getpid(), "project": "waiter", "requested_at": 1})
        )
        assert host.yield_if_requested() is None
        assert server.alive
        assert host.holding
    finally:
        host.unload()


def test_a_quiet_card_and_a_low_memory_floor_each_unload_the_host(tmp_path: Path) -> None:
    memory: list[float | None] = [40.0]
    host = _host(tmp_path, lambda: memory[0])
    card = _quiet_card(tmp_path)
    assert _acquire(host, give_up_after_waits=1) is False  # the card is up: not loaded, and not loading
    GpuLease.remove_quiet(card)
    assert _acquire(host)
    server = host.server(ECHO, ECHO_ARGV, ECHO_ENV, LOAD, lambda: False)
    card = _quiet_card(tmp_path)
    assert host.yield_if_requested() == "bench"
    assert (server.alive, host.holding, _resident_loaded(tmp_path)) == (False, False, False)

    GpuLease.remove_quiet(card)
    assert _acquire(host)
    server = host.server(ECHO, ECHO_ARGV, ECHO_ENV, LOAD, lambda: False)
    memory[0] = 9.0
    assert host.yield_if_requested() == LOW_MEMORY
    assert (server.alive, host.holding, _resident_loaded(tmp_path)) == (False, False, False)


def test_a_stay_up_host_loads_again_only_when_no_quiet_card_remains_and_memory_has_recovered(
    tmp_path: Path,
) -> None:
    memory: list[float | None] = [29.0]  # floor 10 + allowance 20 = 30 needed
    host = _host(tmp_path, lambda: memory[0])
    assert _acquire(host, give_up_after_waits=2) is False
    assert (host.holding, _resident_loaded(tmp_path)) == (False, None)

    memory[0] = 30.0
    card = _quiet_card(tmp_path)
    assert _acquire(host, give_up_after_waits=2) is False
    assert (host.holding, _resident_loaded(tmp_path)) == (False, None)

    # A card that arrives after the host published is seen by the look that follows, and the host backs off.
    GpuLease.remove_quiet(card)
    late_cards: list[Path] = []

    def _free_then_a_card_arrives() -> float:
        if not late_cards:
            late_cards.append(_quiet_card(tmp_path))
        return 30.0

    host = _host(tmp_path, _free_then_a_card_arrives)
    assert _acquire(host) is False
    assert (host.holding, _resident_loaded(tmp_path)) == (False, False)

    GpuLease.remove_quiet(late_cards[0])
    assert _acquire(host)
    assert (host.holding, _resident_loaded(tmp_path)) == (True, True)
    host.unload()
