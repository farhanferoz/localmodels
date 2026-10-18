"""GPU lease protocol: file-based cooperative leasing shared with service_c's bash queue.

The file format is the shared contract documented in service_c's
``docs/2026-09-15-gpu-lease-protocol.md`` and is not this package's to change unilaterally:
one JSON file per project in a shared directory, written atomically (temp file, then rename),
readable by any project that knows the contract — including service_c's ``run_gpu_queue.sh``,
which writes and reads it through inline ``python3 -c`` rather than importing this package.

Stay-up mode (``Contention.STAY_UP``) keeps its records in two subfolders, ``resident/`` and ``quiet/``, which
every reader of the top-level ``*.json`` files (this module's, service_c's, service_a's) never opens.
One file per service (``resident/<project>-<port>.json``) and per quiet card
(``quiet/<project>-<pid>-<token>.json``), so two of one project never overwrite each other's record.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from .atomic import write_text_atomically
from .process import LocalModelsError

if TYPE_CHECKING:
    from collections.abc import Callable

#: The shared lease directory every project on a host uses unless told otherwise; ``GPU_LEASE_DIR``
#: overrides it (slurm's tools on gpu-node-1 read the same variable).
DEFAULT_LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "~/.local/state/gpu-lease")).expanduser()

_RESIDENT_SUBDIR = "resident"
_QUIET_SUBDIR = "quiet"
_QUIET_TIMEOUT_S = 120.0
_QUIET_POLL_S = 0.5


class LeaseRecord(BaseModel):
    """A lease record in the shared GPU lease directory.

    ``extra="ignore"``: a reader must not break on a field only the other project's writer
    knows about (the contract's own rule — "ignore unknown fields"). ``job_id``/``lane`` are
    service_c's queue fields, optional here since this package's own callers do not use them.
    """

    model_config = ConfigDict(extra="ignore")

    pid: int
    project: str
    port: int | None = None
    model_path: str | None = None
    started_at: int | None = None
    requested_at: int | None = None
    job_id: str | None = None
    lane: str | None = None


class ResidentRecord(BaseModel):
    """A stay-up service's ``resident/`` record: its memory allowance and whether its server is loaded."""

    model_config = ConfigDict(extra="ignore")

    pid: int
    project: str
    port: int
    model_path: str
    allowance_gib: float
    loaded: bool
    started_at: int


class QuietRequest(BaseModel):
    """A ``quiet/`` request that every stay-up service unload until it is withdrawn (a benchmark, a tune)."""

    model_config = ConfigDict(extra="ignore")

    pid: int
    project: str
    requested_at: int
    reason: str


class QuietCardTimeoutError(LocalModelsError):
    """Raised when residents are still loaded after a quiet card's timeout."""


def _read_lease_data(path: Path) -> dict[str, object] | None:
    """The JSON object in a lease file, or None if the file is unreadable, not JSON, or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # Unreadable or invalid JSON is left in place (contract: "skip any file they cannot parse")
        return None
    return data if isinstance(data, dict) else None


def _holder_is_live(pid: object, path: Path) -> bool:
    """Whether `pid` names a live process; a dead one deletes `path` as stale.

    A pid that is not a positive integer the OS accepts is unparsable, so False and the file is
    left in place: kill(0|-1, 0) would succeed and read as live forever, and it is not ours to judge.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OverflowError:
        return False
    except ProcessLookupError:
        # Dead PID is stale; delete the file (contract: "stale leases ... ignore it and delete it")
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return False
    except PermissionError:
        # Live process owned by another user
        pass
    return True


def _check_record_file[R: BaseModel](path: Path, model: type[R]) -> R | None:
    """Parse a record file as ``model`` and check whether its holder process is alive."""
    data = _read_lease_data(path)
    if data is None or not _holder_is_live(data.get("pid"), path):
        return None

    if not data.get("project"):
        data["project"] = path.stem

    try:
        return model(**data)
    except ValidationError:
        # A field of the wrong type: unparsable, left in place (contract: "skip any file they cannot parse")
        return None


def _boot_time() -> float | None:
    """When this boot began (``btime`` in /proc/stat); None where there is no such file."""
    try:
        lines = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return next((float(line.split()[1]) for line in lines if line.startswith("btime ")), None)


def _written_before_boot(path: Path, booted_at: float | None) -> bool:
    """Whether ``path`` was last written before this boot (then deleted: its pid may name another process).

    Stay-up records are rewritten on every change, so an older one is always stale; slurm's lease keeper does
    the same for the top-level leases at its start.
    """
    try:
        stale = booted_at is not None and path.stat().st_mtime < booted_at
    except OSError:
        return False
    if stale:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
    return stale


def _live_records[R: BaseModel](directory: Path, model: type[R]) -> list[R]:
    """The ``model`` records in ``directory`` whose holder is alive; a dead or pre-boot record is purged."""
    if not directory.exists():
        return []
    booted_at = _boot_time()
    paths = [path for path in directory.glob("*.json") if not _written_before_boot(path, booted_at)]
    records = (_check_record_file(path, model) for path in paths)
    return [record for record in records if record is not None]


class GpuLease:
    """Manages cooperative GPU leasing via atomic JSON lock files."""

    def __init__(self, directory: Path | str, project: str) -> None:
        self.directory = Path(directory).expanduser()
        self.project = project

    def lease_path(self) -> Path:
        """The canonical path of this project's lease file."""
        return self.directory / f"{self.project}.json"

    def write(self, record: LeaseRecord) -> Path:
        """Write the lease record atomically (a temporary file in the lease directory, then a rename)."""
        return write_text_atomically(self.lease_path(), json.dumps(record.model_dump(exclude_none=True)))

    def read_own(self) -> LeaseRecord | None:
        """This project's own lease record; None when there is none or it cannot be parsed.

        No liveness check: the record names the caller's own process, which is running.
        """
        data = _read_lease_data(self.lease_path())
        if data is None:
            return None
        try:
            return LeaseRecord(**data)
        except ValidationError:
            return None

    def remove(self) -> None:
        """Remove this project's lease file if present."""
        target = self.lease_path()
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)

    def live_others(self) -> list[LeaseRecord]:
        """Read all other lease records, purging stale (dead PID) files."""
        if not self.directory.exists():
            return []

        live: list[LeaseRecord] = []
        my_filename = f"{self.project}.json"

        for path in self.directory.glob("*.json"):
            if path.name == my_filename:
                continue
            record = _check_record_file(path, LeaseRecord)
            if record is not None:
                live.append(record)

        return live

    def _resident_path(self, port: int) -> Path:
        return self.directory / _RESIDENT_SUBDIR / f"{self.project}-{port}.json"

    def write_resident(self, record: ResidentRecord) -> Path:
        """Write the ``resident/`` record of this project's service on ``record.port`` atomically."""
        return write_text_atomically(self._resident_path(record.port), json.dumps(record.model_dump()))

    def remove_resident(self, port: int) -> None:
        """Remove the ``resident/`` record of this project's service on ``port`` if present."""
        self._resident_path(port).unlink(missing_ok=True)

    def residents(self) -> list[ResidentRecord]:
        """Every stay-up service registered by a live process, this project's own included."""
        return _live_records(self.directory / _RESIDENT_SUBDIR, ResidentRecord)

    def write_quiet(self, request: QuietRequest) -> Path:
        """Write a new ``quiet/`` request atomically, in a file of its own; its path withdraws it."""
        name = f"{self.project}-{request.pid}-{secrets.token_hex(4)}.json"
        return write_text_atomically(self.directory / _QUIET_SUBDIR / name, json.dumps(request.model_dump()))

    @staticmethod
    def remove_quiet(path: Path) -> None:
        """Withdraw the quiet request at ``path`` (as ``write_quiet`` returned it) if present."""
        path.unlink(missing_ok=True)

    def quiet_requests(self) -> list[QuietRequest]:
        """Every quiet request whose holder is alive: a holder that died stops blocking the residents."""
        return _live_records(self.directory / _QUIET_SUBDIR, QuietRequest)

    def blockers(self, requested_at: int | None) -> tuple[list[LeaseRecord], bool]:
        """Other live leases this project must wait for, and whether any of them is itself only waiting.

        Holders (no ``requested_at``) always block. Another waiter blocks only if it asked first, ties
        broken by project name; later waiters wait for us. ``requested_at=None`` (not waiting) is
        blocked by every other lease. A reader that treats every other lease as busy, as service_c's
        queue does, would deadlock with a second waiter doing the same; this order cannot. The caller
        withdraws its own request while an earlier waiter exists, so that waiter is not blocked by it.
        Ported from service_b's GPU worker (``service_b/jobs/lease.py``, ``_blockers``).
        """
        mine = None if requested_at is None else (requested_at, self.project)
        blocking: list[LeaseRecord] = []
        earlier_waiter = False
        for record in self.live_others():
            if record.requested_at is None:
                blocking.append(record)
            elif mine is None or (record.requested_at, record.project) < mine:
                blocking.append(record)
                earlier_waiter = True
        return blocking, earlier_waiter

    def requested_by_other(self) -> str | None:
        """Return the project name of the first live other lease requesting the GPU."""
        for record in self.live_others():
            if record.requested_at is not None:
                return record.project
        return None


class GpuLeaseClient:
    """One process's turn at the GPU: ask for it, wait in fair order, hold it, give it back.

    Waiting follows ``GpuLease.blockers``: holders always block, another waiter only if it asked first. While
    an earlier waiter is queued this process withdraws its own request, so that a reader treating every other
    lease as busy (service_c's queue) is not blocked by it, and publishes it again with the original request
    time once that waiter has gone through. Moved here from service_b's GPU worker (``jobs/lease.py``).
    """

    def __init__(
        self, lease: GpuLease, *, pid: int | None = None, port: int | None = None, lane: str | None = None
    ) -> None:
        """Act for ``lease``'s project; ``pid`` (default: this process) owns the lease."""
        self.lease = lease
        self._pid = pid
        self._port = port
        self._lane = lane
        self._job_id: str | None = None
        self._requested_at: int | None = None
        self._admitted_at: int | None = None

    @classmethod
    def resuming(cls, lease: GpuLease, record: LeaseRecord) -> GpuLeaseClient:
        """A client continuing the request in ``record``: this project's lease, written elsewhere."""
        client = cls(lease, pid=record.pid, port=record.port, lane=record.lane)
        client._job_id = record.job_id
        client._requested_at = record.requested_at
        return client

    def _write(self, **fields: int | str | None) -> None:
        self.lease.write(
            LeaseRecord(
                pid=self._pid or os.getpid(),
                project=self.lease.project,
                port=self._port,
                lane=self._lane,
                job_id=self._job_id,
                **fields,
            )
        )

    def begin_wait(self, job_id: str | None = None) -> None:
        """Publish that this project wants the GPU; the request time fixes its place among the waiters."""
        self._job_id = job_id
        self._requested_at = int(time.time())
        self._write(requested_at=self._requested_at)

    def republish(self) -> None:
        """Write this project's request again if it was withdrawn (no-op when not waiting)."""
        if self._requested_at is not None and not self.lease.lease_path().exists():
            self._write(requested_at=self._requested_at)

    def step(self) -> list[LeaseRecord]:
        """One look at the lease directory: the leases this process must still wait for (empty: its turn)."""
        blocking, earlier_waiter = self.lease.blockers(self._requested_at)
        if earlier_waiter and self._requested_at is not None:
            self.lease.remove()
        elif not earlier_waiter:
            self.republish()
        return blocking

    def wait_turn(
        self,
        poll_s: float,
        should_stop: Callable[[], bool] = lambda: False,
        on_poll: Callable[[list[LeaseRecord]], None] = lambda _blocking: None,
    ) -> bool:
        """Block until nothing blocks this process (True), or ``should_stop()`` says give up (False)."""
        while True:
            blocking = self.step()
            on_poll(blocking)
            if not blocking:
                return True
            if should_stop():
                self.republish()
                return False
            time.sleep(poll_s)

    def blocking(self) -> list[LeaseRecord]:
        """The leases this process is waiting for now, read without changing its own record."""
        return self.lease.blockers(self._requested_at)[0]

    def admit(self, *, model_path: str | None = None, job_id: str | None = None) -> None:
        """Record that this project now holds the GPU (again, for each new job while it stays loaded)."""
        if job_id is not None:
            self._job_id = job_id
        self._requested_at = None
        self._admitted_at = int(time.time())
        self._write(started_at=self._admitted_at, model_path=model_path)

    def holding(self) -> list[LeaseRecord]:
        """Every live lease that holds the GPU now, this project's own included."""
        records = [record for record in self.lease.live_others() if record.requested_at is None]
        if self._admitted_at is not None:
            pid = self._pid or os.getpid()
            records.append(LeaseRecord(pid=pid, project=self.lease.project, started_at=self._admitted_at))
        return records

    def requested_by_other(self) -> str | None:
        """The project of another live lease that is waiting for the GPU, if any."""
        return self.lease.requested_by_other()

    def release(self) -> None:
        """Remove this project's lease and forget its request."""
        self._job_id = self._requested_at = self._admitted_at = None
        self.lease.remove()


class QuietCard:
    """While held, every stay-up service has unloaded its server, for a measurement they would disturb.

    Entering writes the quiet request, then waits until each live resident shows ``loaded=false``; leaving
    withdraws the request, also when the measurement raised. Other quiet cards do not block each other: the
    residents stay unloaded until none remains.
    """

    _path: Path

    def __init__(
        self,
        lease: GpuLease,
        *,
        reason: str,
        timeout_s: float = _QUIET_TIMEOUT_S,
        poll_s: float = _QUIET_POLL_S,
    ) -> None:
        self.lease = lease
        self.reason = reason
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _wait_for_residents(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while loaded := [r for r in self.lease.residents() if r.loaded]:
            if time.monotonic() >= deadline:
                names = ", ".join(r.project for r in loaded)
                msg = f"still loaded after {self.timeout_s:g}s, so not measuring: {names}"
                raise QuietCardTimeoutError(msg)
            time.sleep(self.poll_s)

    def __enter__(self) -> Self:
        """Ask the residents to unload and wait until they have."""
        request = QuietRequest(
            pid=os.getpid(), project=self.lease.project, requested_at=int(time.time()), reason=self.reason
        )
        self._path = self.lease.write_quiet(request)
        try:
            self._wait_for_residents()
        except BaseException:
            self.lease.remove_quiet(self._path)
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Withdraw this card's request (only its own: a card nested in another keeps the outer one's)."""
        self.lease.remove_quiet(self._path)
