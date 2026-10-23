"""Unit tests for the GPU lease protocol implementation.

The shared contract is documented in service_c's `docs/2026-09-15-gpu-lease-protocol.md` and
implemented there by `scripts/run_gpu_queue.sh`'s `gpu_lease_write` (bash, via inline
`python3 -c`), not by importing this package. `test_bash_written_lease_parses` below is a real
fixture of exactly what that shell function writes, built from its own source
(`gpu_lease_write`, `run_gpu_queue.sh:608-637`), not a paraphrase.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from localmodels import lease as lease_module
from localmodels.hardware import AvailableMemory, DeviceType
from localmodels.lease import (
    MAX_FROZEN_S,
    FreezableRecord,
    FreezeCard,
    FreezeError,
    FreezeRecord,
    GpuLease,
    GpuLeaseClient,
    LeaseRecord,
    QuietCard,
    QuietCardTimeoutError,
    QuietRequest,
    ResidentRecord,
)


def test_lease_write_atomic_and_remove(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_a")
    rec = LeaseRecord(pid=os.getpid(), project="service_a", port=8091, model_path="/models/test.gguf")
    path = lease.write(rec)

    assert path.exists()
    assert path.name == "service_a.json"
    data = json.loads(path.read_text())
    assert data["pid"] == os.getpid()
    assert data["port"] == 8091
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())  # no temp file left behind

    lease.remove()
    assert not path.exists()


def test_live_others_purges_dead_pid_and_ignores_garbage_and_its_own_lease(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_a")
    lease.write(LeaseRecord(pid=os.getpid(), project="service_a"))

    dead_path = tmp_path / "other_dead.json"
    dead_path.write_text(json.dumps({"pid": 99999999, "project": "dead_proj"}))

    garbage_path = tmp_path / "corrupt.json"
    garbage_path.write_text("not json at all")

    no_pid_path = tmp_path / "no_pid.json"
    no_pid_path.write_text(json.dumps({"project": "no_pid_proj"}))

    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        live_path = tmp_path / "service_c.json"
        live_path.write_text(
            json.dumps({"pid": sub.pid, "project": "service_c", "requested_at": 1700000000})
        )

        others = lease.live_others()
        assert len(others) == 1
        assert others[0].project == "service_c"
        assert others[0].pid == sub.pid

        assert not dead_path.exists()
        assert garbage_path.exists()
        assert no_pid_path.exists()

        assert lease.requested_by_other() == "service_c"
    finally:
        sub.terminate()
        sub.wait()


@pytest.mark.parametrize(
    "bad_field",
    [
        {"port": "abc"},
        {"model_path": 5},
        {"started_at": 1.5},
        {"pid": 2**70},
        {"pid": 0},
        {"pid": -1},
    ],
    ids=["port-str", "model_path-int", "started_at-float", "pid-huge", "pid-zero", "pid-negative"],
)
def test_type_malformed_lease_is_skipped_and_live_ones_still_reported(
    tmp_path: Path, bad_field: dict[str, object]
) -> None:
    """Contract: readers skip any file they cannot parse -- and leave it alone, it is not ours to judge."""
    lease = GpuLease(tmp_path, project="service_a")
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"project": "bad", "pid": sub.pid, **bad_field}))
        (tmp_path / "service_c.json").write_text(json.dumps({"pid": sub.pid, "project": "service_c"}))

        others = lease.live_others()

        assert [o.project for o in others] == ["service_c"]
        assert bad_path.exists()
    finally:
        sub.terminate()
        sub.wait()


def _service_c_bash_lease_json(
    *,
    pid: int,
    port: str = "",
    model_path: str = "",
    started_at: str = "",
    job_id: str = "",
    lane: str = "",
    requested_at: str = "",
) -> str:
    """Reproduce `gpu_lease_write`'s own inline Python body (run_gpu_queue.sh:612-630) verbatim.

    Any arg passed as "" is OMITTED from the JSON — the same rule the bash function's Python body
    applies — so this is the real shape service_c's queue writes, not an approximation of it.
    """
    doc: dict[str, object] = {"project": "service_c", "pid": pid}
    if port:
        doc["port"] = int(port)
    if model_path:
        doc["model_path"] = model_path
    if started_at:
        doc["started_at"] = int(started_at)
    if job_id:
        doc["job_id"] = job_id
    if lane:
        doc["lane"] = lane
    if requested_at:
        doc["requested_at"] = int(requested_at)
    return json.dumps(doc)


def test_lease_written_by_package_and_by_service_c_bash_both_parse(tmp_path: Path) -> None:
    """A lease this package wrote and one shaped exactly like service_c's bash queue both parse."""
    ours = GpuLease(tmp_path, project="service_a")
    ours.write(LeaseRecord(pid=os.getpid(), project="service_a", port=8091, model_path="/models/m.gguf"))

    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        # admission shape: gpu_lease_mark_admitted (run_gpu_queue.sh:650-654) — port 8080, no
        # requested_at, plus service_c's own job_id/lane fields this package never writes.
        bash_path = tmp_path / "service_c.json"
        bash_path.write_text(
            _service_c_bash_lease_json(
                pid=sub.pid,
                port="8080",
                model_path="/models/Qwen3-30B-A3B-Q4_K_M.gguf",
                started_at="1700000000",
                job_id="job-42",
                lane="accurate",
            )
        )

        others = ours.live_others()
        assert {o.project for o in others} == {"service_c"}
        record = next(o for o in others if o.project == "service_c")
        assert record.pid == sub.pid
        assert record.port == 8080
        assert record.job_id == "job-42"
        assert record.lane == "accurate"
    finally:
        sub.terminate()
        sub.wait()


def test_lease_written_by_service_c_bash_waiting_shape_parses(tmp_path: Path) -> None:
    """gpu_lease_begin_wait's shape (run_gpu_queue.sh:641-644): only pid, job_id, lane, requested_at."""
    lease = GpuLease(tmp_path, project="service_a")
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "service_c.json").write_text(
            _service_c_bash_lease_json(pid=sub.pid, job_id="job-7", lane="fast", requested_at="1700000123")
        )
        assert lease.requested_by_other() == "service_c"
    finally:
        sub.terminate()
        sub.wait()


def test_blockers_orders_waiters_by_request_time_then_name(tmp_path: Path) -> None:
    """Holders always block; an earlier waiter blocks; a later waiter (or a tie lost on name) does not."""
    lease = GpuLease(tmp_path, project="slurm")
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "service_a.json").write_text(json.dumps({"pid": sub.pid, "project": "service_a"}))
        (tmp_path / "early.json").write_text(
            json.dumps({"pid": sub.pid, "project": "early", "requested_at": 100})
        )
        (tmp_path / "late.json").write_text(
            json.dumps({"pid": sub.pid, "project": "late", "requested_at": 300})
        )
        (tmp_path / "abc.json").write_text(
            json.dumps({"pid": sub.pid, "project": "abc", "requested_at": 200})
        )
        (tmp_path / "zzz.json").write_text(
            json.dumps({"pid": sub.pid, "project": "zzz", "requested_at": 200})
        )

        blocking, earlier_waiter = lease.blockers(200)

        assert sorted(r.project for r in blocking) == ["abc", "early", "service_a"]
        assert earlier_waiter
    finally:
        sub.terminate()
        sub.wait()


def test_blockers_holder_alone_blocks_without_an_earlier_waiter(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="slurm")
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "service_c.json").write_text(json.dumps({"pid": sub.pid, "project": "service_c"}))
        (tmp_path / "late.json").write_text(
            json.dumps({"pid": sub.pid, "project": "late", "requested_at": 900})
        )

        blocking, earlier_waiter = lease.blockers(500)

        assert [r.project for r in blocking] == ["service_c"]
        assert not earlier_waiter
    finally:
        sub.terminate()
        sub.wait()


def test_blockers_when_not_waiting_counts_every_live_lease(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="slurm")
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "late.json").write_text(
            json.dumps({"pid": sub.pid, "project": "late", "requested_at": 900})
        )
        (tmp_path / "dead.json").write_text(json.dumps({"pid": 99999999, "project": "dead"}))

        blocking, earlier_waiter = lease.blockers(None)

        assert [r.project for r in blocking] == ["late"]
        assert earlier_waiter
    finally:
        sub.terminate()
        sub.wait()


@pytest.mark.parametrize(("offset_s", "blocks"), [(-100, True), (100, False)], ids=["earlier", "later"])
def test_the_client_waits_only_for_an_earlier_waiter_and_steps_aside_for_it(
    tmp_path: Path, offset_s: int, blocks: bool
) -> None:
    """Moved from service_b's GPU worker tests with the client."""
    client = GpuLeaseClient(GpuLease(tmp_path, project="service_b"))
    client.begin_wait("job-a")
    asked = client.lease.read_own()
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        (tmp_path / "other.json").write_text(
            json.dumps({"pid": sub.pid, "project": "other", "requested_at": int(time.time()) + offset_s})
        )
        assert [r.project for r in client.step()] == (["other"] if blocks else [])
        # while an earlier waiter is queued our request is withdrawn, so a reader that treats every lease as
        # busy is not blocked by us
        assert client.lease.lease_path().exists() is not blocks
        assert client.wait_turn(0.01, should_stop=lambda: True) is not blocks
        assert client.lease.read_own() == asked  # giving up restores the request with its original time
    finally:
        sub.terminate()
        sub.wait()


def test_an_admitted_client_holds_the_gpu_until_it_releases(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_b")
    client = GpuLeaseClient(lease, lane="interactive")
    client.begin_wait("job-a")
    client.admit(model_path="/models/m", job_id="job-b")
    own = lease.read_own()
    assert own is not None
    assert (own.requested_at, own.model_path, own.job_id, own.lane) == (
        None,
        "/models/m",
        "job-b",
        "interactive",
    )
    assert own.started_at is not None
    assert [r.project for r in client.holding()] == ["service_b"]
    client.release()
    assert (lease.read_own(), client.holding()) == (None, [])


def test_gpu_lease_dir_overrides_the_default_lease_directory(tmp_path: Path) -> None:
    probe = "from localmodels.lease import DEFAULT_LEASE_DIR; print(DEFAULT_LEASE_DIR)"
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "GPU_LEASE_DIR": str(tmp_path), "PYTHONPATH": src}
    out = subprocess.run(  # noqa: S603 - this interpreter and a fixed probe
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == str(tmp_path)


def _resident(project: str, *, pid: int | None = None, loaded: bool = True) -> ResidentRecord:
    return ResidentRecord(
        pid=pid or os.getpid(), project=project, port=8091, model_path="/m.gguf", allowance_gib=20,
        loaded=loaded, started_at=1700000000,
    )  # fmt: skip


def test_stay_up_records_are_invisible_to_a_reader_of_the_top_level_files(tmp_path: Path) -> None:
    service = GpuLease(tmp_path, project="service_a")
    service.write_resident(_resident("service_a"))
    service.write_quiet(QuietRequest(pid=os.getpid(), project="service_a", requested_at=1, reason="test"))

    reader = GpuLease(tmp_path, project="service_c")
    assert reader.live_others() == []
    assert reader.requested_by_other() is None
    assert list(tmp_path.glob("*.json")) == []  # what a top-level reader such as service_a's globs
    assert [r.project for r in reader.residents()] == ["service_a"]
    assert [q.project for q in reader.quiet_requests()] == ["service_a"]


def test_quiet_card_waits_for_loaded_residents_and_gives_up_on_a_stuck_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = GpuLease(tmp_path, project="bench")
    GpuLease(tmp_path, project="crashed").write_resident(_resident("crashed", pid=99999999))  # dead, loaded
    GpuLease(tmp_path, project="gone").write_quiet(
        QuietRequest(pid=99999999, project="gone", requested_at=1, reason="holder died")
    )
    before_boot = GpuLease(tmp_path, project="rebooted").write_quiet(
        QuietRequest(pid=os.getpid(), project="rebooted", requested_at=1, reason="host crashed")
    )
    os.utime(before_boot, (1, 1))
    monkeypatch.setattr("localmodels.lease._boot_time", lambda: 1000.0)
    # A dead holder's request is purged, and so is one written before this boot (its pid may now name another
    # process), so neither can block residents.
    assert lease.quiet_requests() == []
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        stuck = GpuLease(tmp_path, project="stuck")
        stuck.write_resident(_resident("stuck", pid=sub.pid))
        with (
            pytest.raises(QuietCardTimeoutError, match="stuck") as caught,
            QuietCard(lease, reason="t", timeout_s=0.2, poll_s=0.02),
        ):
            pass
        assert "crashed" not in str(caught.value)  # a resident whose process died counts as gone
        assert list((tmp_path / "quiet").glob("bench-*.json")) == []

        threading.Timer(
            0.2, lambda: stuck.write_resident(_resident("stuck", pid=sub.pid, loaded=False))
        ).start()
        started = time.monotonic()
        with (
            QuietCard(lease, reason="t", timeout_s=5, poll_s=0.02),
            QuietCard(GpuLease(tmp_path, project="tune"), reason="t", timeout_s=5, poll_s=0.02),
        ):
            assert 0.15 < time.monotonic() - started < 3  # waited for the unload
            assert {q.project for q in lease.quiet_requests()} == {"bench", "tune"}  # two cards at once
    finally:
        sub.kill()
        sub.wait()
    assert lease.quiet_requests() == []


def test_two_services_or_cards_of_one_project_keep_their_own_records(tmp_path: Path) -> None:
    service_a = GpuLease(tmp_path, project="service_a")
    service_a.write_resident(_resident("service_a").model_copy(update={"port": 8091}))
    service_a.write_resident(_resident("service_a", loaded=False).model_copy(update={"port": 8092}))
    service_a.remove_resident(8092)  # the embed service ends; chat's record stays
    assert [(r.port, r.loaded) for r in service_a.residents()] == [(8091, True)]

    service_a.remove_resident(8091)
    bench = GpuLease(tmp_path, project="bench")
    with QuietCard(bench, reason="outer", timeout_s=0.1, poll_s=0.01):
        with QuietCard(bench, reason="inner", timeout_s=0.1, poll_s=0.01):
            assert sorted(q.reason for q in bench.quiet_requests()) == ["inner", "outer"]
        assert [q.reason for q in bench.quiet_requests()] == ["outer"]  # the inner card withdrew only its own
    assert bench.quiet_requests() == []


class _FakeSystemd:
    """systemctl --user freeze/thaw/show over a table of unit states; a unit not in it has stopped."""

    def __init__(self, *units: str) -> None:
        self.frozen = dict.fromkeys(units, False)
        self.stuck: set[str] = set()  # thaw fails and the unit stays frozen
        self.no_bus = False  # every call fails with nothing on stdout, as with no user bus

    def __call__(self, verb: str, unit: str) -> subprocess.CompletedProcess[str]:
        if self.no_bus:
            return subprocess.CompletedProcess([], 1, "", "Failed to connect to bus")
        if unit not in self.frozen:  # stopped: show still answers, freeze and thaw refuse
            return subprocess.CompletedProcess(
                [], int(verb != "show"), "FreezerState=running\n", "Unit is inactive."
            )
        if verb == "show":
            return subprocess.CompletedProcess(
                [], 0, f"FreezerState={'frozen' if self.frozen[unit] else 'running'}\n", ""
            )
        if verb == "thaw" and unit in self.stuck:
            return subprocess.CompletedProcess([], 1, "", "dbus timeout")
        self.frozen[unit] = verb == "freeze"
        return subprocess.CompletedProcess([], 0, "", "")


def _memory(monkeypatch: pytest.MonkeyPatch, gib: float | None) -> None:
    free = (
        None
        if gib is None
        else AvailableMemory(bytes=int(gib * 2**30), device=DeviceType.CUDA, unified=True, source="t")
    )
    monkeypatch.setattr("localmodels.hardware.available_memory", lambda _device: free)


def test_freeze_card_fits_first_freezes_and_wakes_and_its_holder_stops_blocking_only_its_requester(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    systemd = _FakeSystemd("nq.service")
    monkeypatch.setattr("localmodels.lease._systemctl", systemd)
    sub = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        service_b = GpuLease(tmp_path, project="service_b")
        service_b.write(LeaseRecord(pid=sub.pid, project="service_b", started_at=1))  # holds the GPU
        service_b.write_freezable(FreezableRecord(pid=sub.pid, project="service_b", unit="nq.service"))
        (service,) = GpuLease(tmp_path, project="service_c").freezables()
        service_c = GpuLease(tmp_path, project="service_c")

        _memory(
            monkeypatch, 8.0
        )  # freezing frees nothing, so a requester that does not fit beside it is refused
        with (
            pytest.raises(FreezeError, match="9 GiB needed"),
            FreezeCard(service_c, service, need_gib=9, reason="t"),
        ):
            pass
        _memory(monkeypatch, None)  # unknown memory is refused too
        with (
            pytest.raises(FreezeError, match="unknown"),
            FreezeCard(service_c, service, need_gib=9, reason="t"),
        ):
            pass
        assert service_c.freezes() == []
        assert systemd.frozen["nq.service"] is False

        _memory(monkeypatch, 40.0)

        def check_then_fail() -> None:
            with FreezeCard(service_c, service, need_gib=9, reason="check"):
                assert systemd.frozen["nq.service"] is True
                assert service_c.blockers(None)[0] == []  # the requester's own wait goes through
                others_see = GpuLease(tmp_path, project="slurm").blockers(None)[0]
                assert [r.project for r in others_see] == ["service_b"]
                with FreezeCard(GpuLease(tmp_path, project="other"), service, need_gib=1, reason="second"):
                    pass
                assert systemd.frozen["nq.service"] is True  # the second card left; the first still holds it
                raise RuntimeError  # the requester's work failed: the service is woken all the same

        with pytest.raises(RuntimeError):
            check_then_fail()
        lapsed = service_c.write_freeze(
            FreezeRecord(
                pid=os.getpid(),
                project="service_c",
                target="service_b",
                unit="nq.service",
                frozen_at=int(time.time() - MAX_FROZEN_S) - 1,
                reason="t",
            )
        )
        assert [r.project for r in service_c.blockers(None)[0]] == [
            "service_b"
        ]  # a lapsed freeze exempts nothing
        lapsed.unlink()
        assert systemd.frozen["nq.service"] is False
        assert service_c.freezes() == []
    finally:
        sub.kill()
        sub.wait()


def test_safety_net_wakes_a_dead_requesters_or_overdue_freeze_and_keeps_one_that_would_not_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    systemd = _FakeSystemd("a.service", "b.service", "c.service", "stuck.service")
    monkeypatch.setattr("localmodels.lease._systemctl", systemd)
    lease = GpuLease(tmp_path, project="slurm")
    now = int(time.time())

    def freeze(unit: str, pid: int, frozen_at: int) -> None:
        if unit != "gone.service":
            systemd.frozen[unit] = True
        lease.write_freeze(
            FreezeRecord(pid=pid, project="tc", target="nq", unit=unit, frozen_at=frozen_at, reason="t")
        )

    freeze("a.service", 99999999, now)  # its requester died
    freeze("b.service", os.getpid(), now - int(MAX_FROZEN_S) - 1)  # held past the cap
    freeze("c.service", os.getpid(), now)  # in force
    freeze("gone.service", 99999999, now)  # stopped while frozen: nothing left to wake
    assert {r.unit for r in lease.wake_abandoned_freezes()} == {"a.service", "b.service", "gone.service"}
    assert systemd.frozen == {
        "a.service": False,
        "b.service": False,
        "c.service": True,
        "stuck.service": False,
    }
    assert [f.unit for _, f in lease.freezes()] == ["c.service"]

    freeze("stuck.service", 99999999, now)
    systemd.stuck.add("stuck.service")
    with pytest.raises(FreezeError, match=r"stuck\.service"):
        lease.wake_abandoned_freezes()
    assert "stuck.service" in {f.unit for _, f in lease.freezes()}  # kept for the next loop to try again
    systemd.stuck.clear()
    systemd.no_bus = True  # a wake that cannot be checked is not a wake
    with pytest.raises(FreezeError):
        lease.wake_abandoned_freezes()
    assert "stuck.service" in {f.unit for _, f in lease.freezes()}


def test_two_cards_waking_one_unit_at_once_still_wake_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    systemd = _FakeSystemd("nq.service")
    systemd.frozen["nq.service"] = True
    monkeypatch.setattr("localmodels.lease._systemctl", systemd)
    real_in_force = lease_module._in_force

    def slow_in_force(
        record: FreezeRecord, now: float
    ) -> bool:  # widens the gap between the check and the unlink
        time.sleep(0.2)
        return real_in_force(record, now)

    monkeypatch.setattr("localmodels.lease._in_force", slow_in_force)
    lease = GpuLease(tmp_path, project="tc")
    paths = [
        lease.write_freeze(
            FreezeRecord(
                pid=os.getpid(),
                project="tc",
                target="nq",
                unit="nq.service",
                frozen_at=int(time.time()),
                reason=r,
            )
        )
        for r in ("a", "b")
    ]
    threads = [threading.Thread(target=lease.wake, args=(path, "nq.service")) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert lease.freezes() == []
    assert systemd.frozen["nq.service"] is False


@pytest.mark.skipif(
    subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, check=False).returncode
    > 1,
    reason="no user systemd here",
)
def test_freeze_card_freezes_a_real_systemd_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = f"localmodels-test-freeze-{os.getpid()}.service"
    subprocess.run(  # noqa: S603 - fixed argv
        ["systemd-run", "--user", f"--unit={unit}", "--collect", "sleep", "30"],
        check=True,
        capture_output=True,
    )
    _memory(monkeypatch, 40.0)

    def state() -> str:
        shown = subprocess.run(  # noqa: S603 - fixed argv
            ["systemctl", "--user", "show", "-p", "FreezerState", "--value", unit],
            capture_output=True,
            text=True,
            check=True,
        )
        return shown.stdout.strip()

    try:
        service = FreezableRecord(pid=os.getpid(), project="svc", unit=unit)
        with FreezeCard(GpuLease(tmp_path, project="t"), service, need_gib=1, reason="t"):
            assert state() == "frozen"
        assert state() == "running"
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], check=False, capture_output=True)  # noqa: S603
