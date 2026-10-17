"""What can go wrong between a model process manager and its model server."""

from __future__ import annotations

from enum import StrEnum


class ServingErrorCode(StrEnum):
    """Machine-readable failure of a model server or its manager (the values travel on the wire)."""

    CUDA_OOM = "cuda_oom"
    BAD_INPUT = "bad_input"
    INTERNAL = "internal"
    TIMEOUT = "timeout"
    NOT_IMPLEMENTED = "not_implemented"


SERVER_ERROR_CODES = frozenset(
    {ServingErrorCode.CUDA_OOM, ServingErrorCode.BAD_INPUT, ServingErrorCode.INTERNAL}
)
"""The codes a model server itself may send; the others are raised by the manager or the registry."""


class ServingError(Exception):
    """A model server or its manager failed: ``code`` is a ``ServingErrorCode`` value; ``message`` is shown.

    A caller may subclass it with its own code enum, as long as the values that cross the wire are
    ``ServingErrorCode`` values; a server reports any other code as ``internal``.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Record the code and its message (which must not carry private input data)."""
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class ServingCancelled(Exception):  # noqa: N818 - reads as a state, like KeyboardInterrupt
    """The caller's cancel check fired while waiting for the model server (which has been stopped)."""
