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
from typing import TYPE_CHECKING

import pytest

from localmodels.lease import GpuLease, LeaseRecord

if TYPE_CHECKING:
    from pathlib import Path


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


def test_live_others_purges_dead_pid_and_ignores_garbage(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_a")

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


def test_live_others_ignores_own_lease(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_a")
    lease.write(LeaseRecord(pid=os.getpid(), project="service_a"))

    others = lease.live_others()
    assert others == []
    assert lease.requested_by_other() is None


def _service_c_bash_lease_json(*, pid: int, port: str = "", model_path: str = "", started_at: str = "",
                                 job_id: str = "", lane: str = "", requested_at: str = "") -> str:
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
