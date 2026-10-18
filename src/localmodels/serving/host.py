"""Keep one model server warm on a shared GPU, holding the GPU lease only while a model is loaded.

The pattern of service_b's GPU worker, for any caller that runs jobs one at a time: wait for the GPU in
request order (``GpuLeaseClient``), keep the server loaded between jobs so the next one skips the load,
unload after ``idle_unload_s`` without a job, and unload at once when another project asks for the GPU.
Starting a different model (or a new version of one) stops the old server first: one model at a time.

``StayUp`` is the opt-in alternative to holding the lease: the host keeps loading beside other projects'
leases, registered in the lease directory's ``resident/`` folder (``lease.ResidentRecord``), and steps aside
only for a quiet card (``lease.QuietCard``) or when available memory runs short.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from localmodels.hardware import available_memory
from localmodels.lease import LeaseRecord, ResidentRecord
from localmodels.serving.client import ModelServerProcess
from localmodels.serving.errors import ServingError, ServingErrorCode
from localmodels.stack import Engine, LoadOptions, ServerStamp, fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from localmodels.lease import GpuLeaseClient
    from localmodels.serving.protocol import LoadMessage

LOW_MEMORY: Final = "low available memory"
"""What ``yield_if_requested`` returns, instead of a project, when a stay-up host unloads for memory."""

_GIB = 2**30


class StayUp(BaseModel):
    """Opt-in stay-up mode for a ``ModelHost``: what it promises the other projects on the machine.

    ``floor_gib``: the host unloads when available memory falls below it. ``allowance_gib``: the memory the
    model needs, so the host loads again only with at least the floor plus this available. ``oom_score_adj``:
    the server's OOM score, so the kernel's killer picks it first (a score below the current one needs
    privileges, and the load is then refused).
    """

    model_config = ConfigDict(frozen=True)

    floor_gib: float = Field(gt=0)
    allowance_gib: float = Field(gt=0)
    oom_score_adj: int = Field(ge=-1000, le=1000)


def _available_gib() -> float | None:
    """Available memory for a CUDA model in GiB (``available_memory``); None when unreadable."""
    reading = available_memory("cuda")
    return None if reading is None else reading.bytes / _GIB


class ModelHost:
    """One warm model server and the GPU lease that goes with it."""

    def __init__(
        self,
        lease: GpuLeaseClient,
        *,
        idle_unload_s: float,
        poll_s: float,
        clock: Callable[[], float] = time.monotonic,
        stay_up: StayUp | None = None,
        available_gib: Callable[[], float | None] = _available_gib,
    ) -> None:
        """``poll_s`` is how often a wait re-reads the lease directory.

        With ``stay_up`` the host never takes the top-level lease: see ``StayUp``. ``available_gib`` reads the
        memory it compares with the floor (never the driver's free count: ``hardware.available_memory``).
        """
        self.lease = lease
        self._idle_unload_s = idle_unload_s
        self._poll_s = poll_s
        self._clock = clock
        self._stay_up = stay_up
        self._available_gib = available_gib
        self._model_path: str | None = None
        self._registered_at: int | None = None  # stay-up: when the resident record was first written
        self._server: ModelServerProcess | None = None
        self._load: LoadMessage | None = None
        self._holding = False
        self._last_use = clock()

    @property
    def loaded(self) -> ModelServerProcess | None:
        """The warm server process, if one is running."""
        return self._server

    @property
    def holding(self) -> bool:
        """Whether this host holds the GPU (a model is loaded or about to be)."""
        return self._holding

    def acquire(
        self,
        *,
        job_id: str,
        model_path: str,
        should_stop: Callable[[], bool],
        on_wait: Callable[[list[LeaseRecord]], None],
    ) -> bool:
        """Hold the GPU for one job: wait for its turn unless already held, then record it as held.

        False when ``should_stop()`` turned true while waiting; the request is then withdrawn.

        In stay-up mode the wait is for no quiet card and enough available memory (floor plus allowance), and
        ``on_wait`` gets the quiet cards' projects (none when only memory is short). A host that is loaded
        but finds pressure unloads first and waits like any other.
        """
        if self._stay_up is not None:
            return self._acquire_stay_up(model_path, should_stop, on_wait)
        if not self._holding:
            self.lease.begin_wait(job_id)
            if not self.lease.wait_turn(self._poll_s, should_stop, on_wait):
                self.lease.release()
                return False
        self.lease.admit(model_path=model_path, job_id=job_id)
        self._holding = True
        return True

    def _acquire_stay_up(
        self, model_path: str, should_stop: Callable[[], bool], on_wait: Callable[[list[LeaseRecord]], None]
    ) -> bool:
        if self._holding and self._pressure() is not None:
            self.unload()
        while not self._holding:
            quiet = self.lease.lease.quiet_requests()
            if not quiet and self._memory_to_load():
                # Publish first, look second: a card writes its request and then looks at the residents, so
                # one of the two sees the other and they cannot both go ahead.
                self._set_loaded(model_path, loaded=True)
                if not self.lease.lease.quiet_requests():
                    self._holding = True
                    break
                self._set_loaded(model_path, loaded=False)
            on_wait([LeaseRecord(pid=q.pid, project=q.project, requested_at=q.requested_at) for q in quiet])
            if should_stop():
                return False
            time.sleep(self._poll_s)
        return True

    def _memory_to_load(self) -> bool:
        assert self._stay_up is not None  # noqa: S101 - narrowing for mypy
        free = self._available_gib()
        return free is not None and free >= self._stay_up.floor_gib + self._stay_up.allowance_gib

    def _pressure(self) -> str | None:
        """Why a loaded stay-up host should unload now: a quiet card's project, or ``LOW_MEMORY``."""
        assert self._stay_up is not None  # noqa: S101 - narrowing for mypy
        if quiet := self.lease.lease.quiet_requests():
            return quiet[0].project
        free = self._available_gib()
        return LOW_MEMORY if free is not None and free < self._stay_up.floor_gib else None

    def _set_loaded(self, model_path: str | None, *, loaded: bool) -> None:
        """Publish whether the server is loaded, in the ``resident/`` record (once a path is known)."""
        assert self._stay_up is not None  # noqa: S101 - narrowing for mypy
        self._model_path = model_path or self._model_path
        if self._model_path is None:
            return
        self._registered_at = self._registered_at or int(time.time())
        self.lease.lease.write_resident(
            ResidentRecord(
                pid=os.getpid(),
                project=self.lease.lease.project,
                port=0,  # a model server speaks over stdin and stdout, it has no port
                model_path=self._model_path,
                allowance_gib=self._stay_up.allowance_gib,
                loaded=loaded,
                started_at=self._registered_at,
            )
        )

    def _set_oom_score(self, server: ModelServerProcess) -> None:
        """Set the OOM score of the just-started server; it has allocated nothing yet, as no load was sent."""
        assert self._stay_up is not None  # noqa: S101 - narrowing for mypy
        score = self._stay_up.oom_score_adj
        try:
            Path(f"/proc/{server.pid}/oom_score_adj").write_text(str(score))
        except OSError as exc:
            msg = (
                f"The model server's OOM score could not be set to {score} ({exc.strerror}); not loading it."
            )
            raise ServingError(ServingErrorCode.INTERNAL, msg) from exc

    def server(
        self,
        identity: Hashable,
        argv: Sequence[str],
        env: Mapping[str, str],
        load: LoadMessage,
        cancel_check: Callable[[], bool],
    ) -> ModelServerProcess:
        """The warm server of the model ``identity`` names; another model's server is stopped first."""
        if self._server is not None and self._server.serves(identity):
            return self._server
        self._stop_server()
        server = ModelServerProcess(argv, env, identity)
        self._server = server
        self._load = load
        try:
            server.start()
            if self._stay_up is not None:
                self._set_oom_score(server)
            server.load(load, cancel_check)
        except BaseException:
            self._stop_server()
            raise
        return server

    def stamp(self, model_sha256: str | None = None) -> ServerStamp | None:
        """How the warm server was made (its load's model directory, precision, options); None with no server.

        ``model_sha256`` is the model's hash when the caller knows it; it is never computed here.
        """
        if self._server is None or self._load is None:
            return None
        return ServerStamp(
            engine=Engine.PYTORCH,
            build=self._server.server_version or "unknown",
            model_path=self._load.model_dir,
            model_sha256=model_sha256,
            load=LoadOptions(dtype=self._load.dtype, options=self._load.options),
            host=fingerprint(),
        )

    def finished(self) -> None:
        """A job ended: restart the idle clock, and give the GPU back if the server is gone."""
        self._last_use = self._clock()
        if self._server is None or not self._server.alive:
            self.unload()

    def unload_if_idle(self) -> None:
        """Unload when no job has run for ``idle_unload_s``."""
        if self._holding and self._clock() - self._last_use >= self._idle_unload_s:
            self.unload()

    def yield_if_requested(self) -> str | None:
        """When another project waits for the GPU: unload and return that project; None otherwise.

        A stay-up host ignores ordinary requests; it unloads for a quiet card (returning its project) or when
        available memory is below the floor (returning ``LOW_MEMORY``). The worker polls this between jobs.
        """
        other = self._pressure() if self._stay_up is not None else self.lease.requested_by_other()
        if other is not None:
            self.unload()
        return other

    def unload(self) -> None:
        """Stop the server and remove the lease (also a waiting record a failed job left behind).

        In stay-up mode there is no lease: the ``resident/`` record stays, with ``loaded`` false.
        """
        self._stop_server()
        if self._stay_up is None:
            self.lease.release()
        else:
            self._set_loaded(None, loaded=False)
        self._holding = False

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
