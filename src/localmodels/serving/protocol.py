"""Wire protocol between a model process manager and its model server: one JSON object per line.

One pydantic model per ``op``. The manager (``client.ModelServerProcess``) sends ``load``, ``infer`` and
``shutdown``; the server announces itself with ``ready`` (its localmodels version) and answers ``loaded``,
``progress``, ``result`` or ``error``. The manager refuses a server whose version differs from its own, so an
environment left on an older localmodels fails with a plain message, not a malformed-message error.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from localmodels.enums import Precision
from localmodels.serving.errors import SERVER_ERROR_CODES, ServingErrorCode

NEWLINE = b"\n"


class Op(StrEnum):
    """Discriminator of every protocol message."""

    READY = "ready"
    LOAD = "load"
    LOADED = "loaded"
    INFER = "infer"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ProgressStage(StrEnum):
    """Phase a generating model server reports."""

    PREFILL = "prefill"
    DECODE = "decode"


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadyMessage(_Message):
    """Server -> manager, first line on start: the localmodels version the server runs (checked)."""

    op: Literal[Op.READY] = Op.READY
    version: str


class LoadMessage(_Message):
    """Manager -> server: load the model in ``model_dir``; ``options`` carries what else the server needs."""

    op: Literal[Op.LOAD] = Op.LOAD
    model_dir: str
    revision: str
    dtype: Precision = Precision.BFLOAT16
    options: dict[str, Any] = Field(default_factory=dict)


class LoadedMessage(_Message):
    """Server -> manager: the model is loaded."""

    op: Literal[Op.LOADED] = Op.LOADED
    seconds: float
    torch_allocated_gib: float
    torch_reserved_gib: float
    #: Library versions the server runs (``server.LIBRARIES``); here, not in ``ready``, so that a client of
    #: an older localmodels still reads ``ready`` and refuses by version before it meets this field. That
    #: needs the version number to have moved: a client with the same version cannot read this field and
    #: reports a malformed message.
    libraries: dict[str, str] = Field(default_factory=dict)


class InferMessage(_Message):
    """Manager -> server: run one inference on the input in ``input_dir``."""

    op: Literal[Op.INFER] = Op.INFER
    request_id: str
    input_dir: str
    params: dict[str, Any] = Field(default_factory=dict)


class ProgressMessage(_Message):
    """Server -> manager: generation progress."""

    op: Literal[Op.PROGRESS] = Op.PROGRESS
    request_id: str
    stage: ProgressStage
    new_tokens: int


class ResultMessage(_Message):
    """Server -> manager: the inference output."""

    op: Literal[Op.RESULT] = Op.RESULT
    request_id: str
    raw_text: str
    native_outputs: dict[str, Any] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    peak_allocated_gib: float = 0.0


class ErrorMessage(_Message):
    """Server -> manager: a failure; ``message`` carries no private input data."""

    op: Literal[Op.ERROR] = Op.ERROR
    request_id: str | None = None
    code: ServingErrorCode
    message: str

    @field_validator("code")
    @classmethod
    def _sent_by_a_server(cls, code: ServingErrorCode) -> ServingErrorCode:
        if code not in SERVER_ERROR_CODES:
            msg = f"{code.value} is not a model-server error code"
            raise ValueError(msg)
        return code


class ShutdownMessage(_Message):
    """Manager -> server: exit."""

    op: Literal[Op.SHUTDOWN] = Op.SHUTDOWN


Message = Annotated[
    ReadyMessage
    | LoadMessage
    | LoadedMessage
    | InferMessage
    | ProgressMessage
    | ResultMessage
    | ErrorMessage
    | ShutdownMessage,
    Field(discriminator="op"),
]
_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)


def encode(msg: Message) -> bytes:
    """One UTF-8 JSON line ending in a newline."""
    return _ADAPTER.dump_json(msg) + NEWLINE


def decode(line: bytes | str) -> Message:
    """Parse one protocol line.

    Raises:
        pydantic.ValidationError: not valid JSON, an unknown ``op``, or fields that do not fit it.
    """
    return _ADAPTER.validate_json(line)
