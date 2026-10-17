"""The model-server side: a child process that loads one model and answers its manager over stdin/stdout.

A caller's entry module builds its ``ModelServer`` and calls ``run``. The server loop never lets an exception
end the process: every request gets an answer, and an unexpected exception is reported by its class name
only, so no private input data reaches the manager's logs. ``block_network`` and ``open_load`` are the start
of every ``load``: no model library may reach the network, and a GPU must be visible. PyTorch is imported
only inside the child (``open_load``), never by the manager.
"""

from __future__ import annotations

import os
import socket
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, NoReturn

from pydantic import BaseModel, ValidationError

from localmodels.serving.errors import SERVER_ERROR_CODES, ServingError, ServingErrorCode
from localmodels.serving.protocol import (
    ErrorMessage,
    InferMessage,
    LoadedMessage,
    LoadMessage,
    ProgressMessage,
    ReadyMessage,
    ResultMessage,
    ShutdownMessage,
    decode,
    encode,
)
from localmodels.version import package_version

if TYPE_CHECKING:
    from typing import BinaryIO

ProgressEmitter = Callable[[ProgressMessage], None]

NETWORK_BLOCKED: Final = "network access is blocked in the model server"


class ModelServer(ABC):
    """Runs inside the model-server process; one instance per process."""

    @abstractmethod
    def load(self, msg: LoadMessage) -> LoadedMessage:
        """Load the model."""

    @abstractmethod
    def infer(self, msg: InferMessage, emit_progress: ProgressEmitter) -> ResultMessage:
        """Run one inference.

        Raises:
            ServingError: a known failure (its code becomes the ``error`` message's code).
        """


def reply_stream() -> BinaryIO:
    """A private handle on the protocol channel (fd 1); anything else the process prints goes to stderr."""
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return channel


def _error(code: str, message: str, request_id: str | None = None) -> ErrorMessage:
    """An error reply; a code outside the server's set is reported as ``internal``."""
    known = code in SERVER_ERROR_CODES
    return ErrorMessage(
        request_id=request_id,
        code=ServingErrorCode(code) if known else ServingErrorCode.INTERNAL,
        message=message,
    )


def _emit(out: BinaryIO, msg: ProgressMessage) -> None:
    out.write(encode(msg))


def serve(server: ModelServer, stdin: BinaryIO, out: BinaryIO) -> None:
    """Dispatch messages from ``stdin`` to ``server`` until ``shutdown`` or the end of the input."""
    for line in stdin:
        request_id: str | None = None
        try:
            msg = decode(line)
            if isinstance(msg, ShutdownMessage):
                return
            if isinstance(msg, LoadMessage):
                out.write(encode(server.load(msg)))
            elif isinstance(msg, InferMessage):
                request_id = msg.request_id
                out.write(encode(server.infer(msg, lambda progress: _emit(out, progress))))
            else:
                out.write(encode(_error(ServingErrorCode.BAD_INPUT, "unexpected message", request_id)))
        except ServingError as exc:
            out.write(encode(_error(exc.code, exc.message, request_id)))
        except ValidationError:
            out.write(encode(_error(ServingErrorCode.BAD_INPUT, "malformed message", request_id)))
        except Exception as exc:  # noqa: BLE001 - the process must answer, with the class name only
            out.write(encode(_error(ServingErrorCode.INTERNAL, type(exc).__name__, request_id)))


def run(server: ModelServer) -> int:
    """Announce this process's localmodels version, then serve ``server`` on stdin/stdout; the exit code."""
    out = reply_stream()
    out.write(encode(ReadyMessage(version=package_version())))
    serve(server, sys.stdin.buffer, out)
    return 0


def block_network() -> list[str]:
    """Make every non-local connection and name lookup fail in this process.

    Inputs never leave the machine: a model library that tries to reach the network (telemetry, a hub
    download) is refused. The returned list gets one entry per attempt, for the server to report.
    """
    attempts: list[str] = []
    connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family != socket.AF_UNIX:
            attempts.append("connect")
            raise OSError(NETWORK_BLOCKED)
        connect(self, address)

    def refused_lookup(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        attempts.append("lookup")
        raise OSError(NETWORK_BLOCKED)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = refused_lookup
    return attempts


def open_load[SpecT: BaseModel](spec: type[SpecT], msg: LoadMessage) -> tuple[SpecT, list[str]]:
    """Open a ``load``: ``msg.options`` read as ``spec``, and the attempts list of ``block_network``.

    The network is blocked before any model code runs; then a GPU must be visible.

    Raises:
        ServingError: ``bad_input`` for options that are not a ``spec``; ``internal`` when no GPU is visible.
    """
    import torch  # noqa: PLC0415 - only the model-server process imports PyTorch

    try:
        request = spec.model_validate(msg.options)
    except ValidationError:
        raise ServingError(ServingErrorCode.BAD_INPUT, "invalid load request") from None
    attempts = block_network()
    if not torch.cuda.is_available():
        raise ServingError(ServingErrorCode.INTERNAL, "no GPU is visible")
    return request, attempts
