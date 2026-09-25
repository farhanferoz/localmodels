"""GPU lease protocol: file-based cooperative leasing shared with service_c's bash queue.

The file format is the shared contract documented in service_c's
``docs/2026-09-15-gpu-lease-protocol.md`` and is not this package's to change unilaterally:
one JSON file per project in a shared directory, written atomically (temp file, then rename),
readable by any project that knows the contract — including service_c's ``run_gpu_queue.sh``,
which writes and reads it through inline ``python3 -c`` rather than importing this package.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict


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


def _check_other_lease_file(path: Path) -> LeaseRecord | None:
    """Parse an external lease file and check whether its holder process is alive."""
    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # Unreadable or invalid JSON is left in place (contract: "skip any file they cannot parse")
        return None

    if not isinstance(data, dict):
        return None
    pid_raw = data.get("pid")
    if not isinstance(pid_raw, int):
        return None

    try:
        os.kill(pid_raw, 0)
    except ProcessLookupError:
        # Dead PID is stale; delete the file (contract: "stale leases ... ignore it and delete it")
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return None
    except PermissionError:
        # Live process owned by another user
        pass

    if not data.get("project"):
        data["project"] = path.stem

    return LeaseRecord(**data)


class GpuLease:
    """Manages cooperative GPU leasing via atomic JSON lock files."""

    def __init__(self, directory: Path | str, project: str) -> None:
        self.directory = Path(directory).expanduser()
        self.project = project

    def lease_path(self) -> Path:
        """The canonical path of this project's lease file."""
        return self.directory / f"{self.project}.json"

    def write(self, record: LeaseRecord) -> Path:
        """Write the lease record atomically via a temporary file in the lease directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.lease_path()
        fd, temp_path = tempfile.mkstemp(dir=self.directory, prefix=f"{self.project}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            payload = record.model_dump(exclude_none=True)
            json.dump(payload, handle)
        Path(temp_path).replace(target)
        return target

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
            record = _check_other_lease_file(path)
            if record is not None:
                live.append(record)

        return live

    def requested_by_other(self) -> str | None:
        """Return the project name of the first live other lease requesting the GPU."""
        for record in self.live_others():
            if record.requested_at is not None:
                return record.project
        return None
