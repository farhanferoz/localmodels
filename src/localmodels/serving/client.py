"""The manager side: start a model server in its own environment, talk to it, and stop it. Torch-free.

The server runs as a child in its own process group (so a stop reaches everything it started), with the
protocol on its stdin and stdout. Every wait polls the caller's cancel check about once a second and has a
deadline, so a stuck model never blocks the caller: cancelling or timing out kills the process group.
"""

from __future__ import annotations

import contextlib
import logging
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeVar

from pydantic import ValidationError

from localmodels.hardware import network_filesystem
from localmodels.serving.errors import ServingCancelled, ServingError, ServingErrorCode
from localmodels.serving.protocol import (
    ErrorMessage,
    InferMessage,
    LoadedMessage,
    LoadMessage,
    Message,
    ProgressMessage,
    ReadyMessage,
    ResultMessage,
    ShutdownMessage,
    decode,
    encode,
)
from localmodels.version import package_version

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

POLL_S: Final = 1.0
#: A pause between two polls longer than this beyond ``POLL_S`` is time this process spent frozen.
FROZEN_GAP_S: Final = 5.0
GRACE_S: Final = 10.0
DEFAULT_INFER_TIMEOUT_S: Final = 900.0
DEFAULT_LOAD_TIMEOUT_S: Final = 1800.0
READ_CHUNK: Final = 65536

MODEL_SERVER_ENV: Final = {
    "HF_HUB_OFFLINE": "1",  # weights are pinned and already on disk: never reach the Hub at load time
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # less fragmentation across varied input sizes
}
"""Added to every model server's environment (the caller's ``env`` overrides it)."""

R = TypeVar("R", bound=Message)

log = logging.getLogger(__name__)


def _frozen_s(last: float, now: float) -> float:
    """Time this process spent frozen (D10), or the host asleep, between two polls: not the server's time."""
    gap = now - last - POLL_S
    return gap if gap > FROZEN_GAP_S else 0.0


class ModelServerProcess:
    """One model-server child process; ``identity`` says which model (and version) it serves."""

    def __init__(
        self, argv: Sequence[str], env: Mapping[str, str] | None = None, identity: Hashable = None
    ) -> None:
        """Remember what to start: ``argv`` runs the server, ``env`` adds to this process's environment."""
        self.argv = list(argv)
        self.env = {**os.environ, **MODEL_SERVER_ENV, **(env or {})}
        self.identity = identity
        self.server_version: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._buffer = b""

    @property
    def alive(self) -> bool:
        """Whether the child is running."""
        return self._proc is not None and self._proc.poll() is None

    def serves(self, identity: Hashable) -> bool:
        """Whether this running server holds the model ``identity`` names."""
        return self.alive and self.identity == identity

    @property
    def pid(self) -> int | None:
        """Process id once started."""
        return None if self._proc is None else self._proc.pid

    def start(self) -> None:
        """Start the child in a new session, its stdout read without blocking.

        Raises:
            ServingError: ``internal``; the server program could not be run (missing, not executable).
        """
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv list from the caller, no shell
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env, start_new_session=True
            )
        except OSError as exc:
            raise ServingError(
                ServingErrorCode.INTERNAL,
                f"The model server could not start: {self.argv[0]}: {exc.strerror}. "
                "Check the server command.",
            ) from exc
        assert self._proc.stdout is not None  # noqa: S101 - narrowing for mypy
        os.set_blocking(self._proc.stdout.fileno(), False)
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)
        self._buffer = b""
        self.server_version = None

    def load(
        self,
        request: LoadMessage,
        cancel_check: Callable[[], bool],
        timeout_s: float = DEFAULT_LOAD_TIMEOUT_S,
    ) -> LoadedMessage:
        """Load the model.

        Raises:
            ServingCancelled: ``cancel_check()`` turned true (the process has been stopped).
            ServingError: the server failed, timed out or exited; ``not_implemented`` when it runs another
                localmodels version than this process.
        """
        if (kind := network_filesystem(Path(request.model_dir))) is not None:
            log.warning(
                "model %s is on a network filesystem (%s): every cold load reads it at network speed; "
                "keep model files on a local disk",
                request.model_dir,
                kind,
            )
        self._send(request)
        return self._await(LoadedMessage, cancel_check, timeout_s, lambda _progress: None)

    def infer(
        self,
        request: InferMessage,
        cancel_check: Callable[[], bool],
        on_progress: Callable[[ProgressMessage], None],
        timeout_s: float = DEFAULT_INFER_TIMEOUT_S,
    ) -> ResultMessage:
        """Run one inference; raises as ``load`` does."""
        self._send(request)
        return self._await(ResultMessage, cancel_check, timeout_s, on_progress)

    def stop(self) -> None:
        """Ask the server to exit, then SIGTERM and SIGKILL its process group if it does not."""
        if self._proc is None:
            return
        if self.alive:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._send(ShutdownMessage())
            self._wait(GRACE_S)
        self._terminate()

    def _send(self, msg: LoadMessage | InferMessage | ShutdownMessage) -> None:
        assert self._proc is not None  # noqa: S101 - narrowing for mypy
        assert self._proc.stdin is not None  # noqa: S101
        self._proc.stdin.write(encode(msg))
        self._proc.stdin.flush()

    def _await(
        self,
        want: type[R],
        cancel_check: Callable[[], bool],
        timeout_s: float,
        on_progress: Callable[[ProgressMessage], None],
    ) -> R:
        last = time.monotonic()
        deadline = last + timeout_s
        while True:
            if cancel_check():
                self._terminate()
                raise ServingCancelled
            now = time.monotonic()  # one reading: a freeze since the last lands in the gap
            deadline += _frozen_s(last, now)
            last = now
            if now >= deadline:
                self._terminate()
                raise ServingError(ServingErrorCode.TIMEOUT, "The model server did not answer in time.")
            msg = self._next_message(min(POLL_S, max(deadline - now, 0.0)))
            if msg is None:
                continue
            if isinstance(msg, ReadyMessage):
                self._check_version(msg.version)
                continue
            if self.server_version is None:
                self._mismatch("an older localmodels (it did not say which version)")
            if isinstance(msg, ErrorMessage):
                raise ServingError(msg.code, msg.message)
            if isinstance(msg, ProgressMessage):
                on_progress(msg)
            elif isinstance(msg, want):
                return msg
            else:
                raise ServingError(ServingErrorCode.INTERNAL, "Unexpected message from the model server.")

    def _check_version(self, version: str) -> None:
        self.server_version = version
        if version != package_version():
            self._mismatch(f"localmodels {version}")

    def _mismatch(self, server: str) -> None:
        self._terminate()
        msg = (
            f"The model server runs {server} and this process runs localmodels {package_version()}: "
            "install the same version in the model's environment."
        )
        raise ServingError(ServingErrorCode.NOT_IMPLEMENTED, msg)

    def _next_message(self, wait_s: float) -> Message | None:
        """The next complete line from the server, or None if none arrived within ``wait_s``."""
        assert self._proc is not None  # noqa: S101 - narrowing for mypy
        assert self._proc.stdout is not None  # noqa: S101
        assert self._selector is not None  # noqa: S101
        if b"\n" not in self._buffer and self._selector.select(wait_s):
            chunk = os.read(self._proc.stdout.fileno(), READ_CHUNK)
            if not chunk:
                self._terminate()
                raise ServingError(ServingErrorCode.INTERNAL, "The model server exited unexpectedly.")
            self._buffer += chunk
        if b"\n" not in self._buffer:
            return None
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            return decode(line)
        except ValidationError:
            self._terminate()
            raise ServingError(
                ServingErrorCode.INTERNAL, "The model server sent a malformed message."
            ) from None

    def _wait(self, seconds: float) -> bool:
        """Wait for the child to exit; whether it did."""
        assert self._proc is not None  # noqa: S101 - narrowing for mypy
        try:
            self._proc.wait(seconds)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _terminate(self) -> None:
        """SIGTERM the process group, SIGKILL after the grace period, and reap the child."""
        assert self._proc is not None  # noqa: S101 - narrowing for mypy
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if self._proc.poll() is not None:
                break
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, sig)
            if self._wait(GRACE_S):
                break
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        for pipe in (self._proc.stdin, self._proc.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
