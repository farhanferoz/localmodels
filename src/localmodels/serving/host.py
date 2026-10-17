"""Keep one model server warm on a shared GPU, holding the GPU lease only while a model is loaded.

The pattern of service_b's GPU worker, for any caller that runs jobs one at a time: wait for the GPU in
request order (``GpuLeaseClient``), keep the server loaded between jobs so the next one skips the load,
unload after ``idle_unload_s`` without a job, and unload at once when another project asks for the GPU.
Starting a different model (or a new version of one) stops the old server first: one model at a time.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from localmodels.serving.client import ModelServerProcess

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from localmodels.lease import GpuLeaseClient, LeaseRecord
    from localmodels.serving.protocol import LoadMessage


class ModelHost:
    """One warm model server and the GPU lease that goes with it."""

    def __init__(
        self,
        lease: GpuLeaseClient,
        *,
        idle_unload_s: float,
        poll_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``poll_s`` is how often a wait re-reads the lease directory."""
        self.lease = lease
        self._idle_unload_s = idle_unload_s
        self._poll_s = poll_s
        self._clock = clock
        self._server: ModelServerProcess | None = None
        self._holding = False
        self._last_use = clock()

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
        """
        if not self._holding:
            self.lease.begin_wait(job_id)
            if not self.lease.wait_turn(self._poll_s, should_stop, on_wait):
                self.lease.release()
                return False
        self.lease.admit(model_path=model_path, job_id=job_id)
        self._holding = True
        return True

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
        try:
            server.start()
            server.load(load, cancel_check)
        except BaseException:
            self._stop_server()
            raise
        return server

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
        """When another project waits for the GPU: unload and return that project; None otherwise."""
        other = self.lease.requested_by_other()
        if other is not None:
            self.unload()
        return other

    def unload(self) -> None:
        """Stop the server and remove the lease (also a waiting record a failed job left behind)."""
        self._stop_server()
        self.lease.release()
        self._holding = False

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
