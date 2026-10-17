"""Unit tests for `localmodels lease {status,acquire,wait,begin-wait,admitted,release}`.

Carries B12.3's DEVIATION: the `lease` subcommands' documented exit codes live in this task
(B12.4), in `cli.py`, so service_c's bash queue can call them instead of inline `python3 -c`.
Each exit code below is asserted directly against `main()`, the real CLI entry point -- these
are integration tests of the argparse wiring, not of `GpuLease` itself (B12.3 already covers
that in `tests/test_lease.py`).
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

from localmodels.cli import (
    EXIT_ERROR,
    EXIT_LEASE_CONFLICT,
    EXIT_LEASE_TIMEOUT,
    EXIT_OK,
    main,
)
from localmodels.lease import GpuLease, LeaseRecord

if TYPE_CHECKING:
    from pathlib import Path


def test_lease_acquire_succeeds_when_no_other_lease(tmp_path: Path) -> None:
    code = main(
        [
            "lease",
            "acquire",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--pid",
            str(os.getpid()),
            "--port",
            "8091",
            "--model-path",
            "/m.gguf",
        ]
    )
    assert code == EXIT_OK
    data = json.loads((tmp_path / "service_a.json").read_text())
    assert data["pid"] == os.getpid()
    assert data["port"] == 8091


def test_lease_acquire_refuses_when_another_lease_is_live(tmp_path: Path) -> None:
    other = GpuLease(tmp_path, project="service_c")
    other.write(LeaseRecord(pid=os.getpid(), project="service_c", port=8080))

    code = main(
        [
            "lease",
            "acquire",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--pid",
            str(os.getpid()),
        ]
    )
    assert code == EXIT_LEASE_CONFLICT
    assert not (tmp_path / "service_a.json").exists()


def test_lease_wait_returns_ok_immediately_when_free(tmp_path: Path) -> None:
    code = main(
        [
            "lease",
            "wait",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--timeout",
            "1",
            "--poll",
            "0.01",
        ]
    )
    assert code == EXIT_OK


def test_lease_wait_times_out_while_another_lease_is_live(tmp_path: Path) -> None:
    other = GpuLease(tmp_path, project="service_c")
    other.write(LeaseRecord(pid=os.getpid(), project="service_c", port=8080))

    code = main(
        [
            "lease",
            "wait",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--timeout",
            "0.05",
            "--poll",
            "0.01",
        ]
    )
    assert code == EXIT_LEASE_TIMEOUT


def test_lease_wait_after_begin_wait_is_not_blocked_by_a_later_waiter(tmp_path: Path) -> None:
    lease_args = ["--project", "service_a", "--lease-dir", str(tmp_path)]
    assert main(["lease", "begin-wait", *lease_args, "--pid", str(os.getpid())]) == EXIT_OK
    later = GpuLease(tmp_path, project="service_c")
    later.write(LeaseRecord(pid=os.getpid(), project="service_c", requested_at=int(time.time()) + 100))

    assert main(["lease", "wait", *lease_args, "--timeout", "0.05", "--poll", "0.01"]) == EXIT_OK


def test_lease_wait_succeeds_once_the_other_lease_clears(tmp_path: Path) -> None:
    """A lease whose pid is dead is stale and purged by `live_others()` -- `wait` should not
    need the holder to call `release` itself for a crashed holder to be seen as free."""
    dead_pid = 999999  # exceedingly unlikely to be a live pid on this host
    (tmp_path / "service_c.json").write_text(
        json.dumps({"pid": dead_pid, "project": "service_c", "port": 8080})
    )
    code = main(
        [
            "lease",
            "wait",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--timeout",
            "1",
            "--poll",
            "0.01",
        ]
    )
    assert code == EXIT_OK


def test_lease_begin_wait_writes_requested_at(tmp_path: Path) -> None:
    code = main(
        [
            "lease",
            "begin-wait",
            "--project",
            "service_c",
            "--lease-dir",
            str(tmp_path),
            "--pid",
            str(os.getpid()),
            "--job-id",
            "job-1",
            "--lane",
            "fast",
        ]
    )
    assert code == EXIT_OK
    data = json.loads((tmp_path / "service_c.json").read_text())
    assert data["requested_at"] is not None
    assert data["job_id"] == "job-1"
    assert data["lane"] == "fast"


def test_lease_admitted_rewrites_without_requested_at(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_c")
    lease.write(LeaseRecord(pid=os.getpid(), project="service_c", requested_at=int(time.time())))

    code = main(
        [
            "lease",
            "admitted",
            "--project",
            "service_c",
            "--lease-dir",
            str(tmp_path),
            "--pid",
            str(os.getpid()),
            "--port",
            "8080",
            "--model-path",
            "/models/m.gguf",
        ]
    )
    assert code == EXIT_OK
    data = json.loads((tmp_path / "service_c.json").read_text())
    assert "requested_at" not in data
    assert data["port"] == 8080


def test_lease_release_removes_the_file(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="service_a")
    lease.write(LeaseRecord(pid=os.getpid(), project="service_a", port=8091))

    code = main(["lease", "release", "--project", "service_a", "--lease-dir", str(tmp_path)])
    assert code == EXIT_OK
    assert not (tmp_path / "service_a.json").exists()


def test_lease_release_is_idempotent_when_nothing_to_release(tmp_path: Path) -> None:
    code = main(["lease", "release", "--project", "service_a", "--lease-dir", str(tmp_path)])
    assert code == EXIT_OK


def test_lease_status_reports_mine_and_others(tmp_path: Path) -> None:
    mine = GpuLease(tmp_path, project="service_a")
    mine.write(LeaseRecord(pid=os.getpid(), project="service_a", port=8091))
    other = GpuLease(tmp_path, project="service_c")
    other.write(LeaseRecord(pid=os.getpid(), project="service_c", port=8080))

    code = main(
        [
            "lease",
            "status",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--json-out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == EXIT_OK
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["mine"]["port"] == 8091
    assert len(report["others"]) == 1
    assert report["others"][0]["project"] == "service_c"


def test_lease_status_mine_is_none_when_no_lease_written(tmp_path: Path) -> None:
    code = main(
        [
            "lease",
            "status",
            "--project",
            "service_a",
            "--lease-dir",
            str(tmp_path),
            "--json-out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code == EXIT_OK
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["mine"] is None


def test_main_with_no_args_returns_error() -> None:
    assert main([]) == EXIT_ERROR
