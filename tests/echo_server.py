"""A model server for the tests that needs no model: ``python -m tests.echo_server``.

``infer`` echoes the input directory and acts on its params: ``progress`` (report that many decode steps),
``sleep_s``, ``fail`` (``"oom"``: a reported out-of-memory error; ``"leak"``: an unexpected exception whose
text must not reach the manager) and ``exit`` (the process dies mid-request).
Load options shape every inference after them: ``sleep_s`` (each
inference sleeps that long), ``text`` (``raw_text`` is that, not the input directory), ``peak_gib`` (reported
as ``peak_allocated_gib``) and ``fail_dtype`` (a load in that dtype is refused as not implemented).
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

from localmodels.serving import (
    InferMessage,
    LoadedMessage,
    LoadMessage,
    ModelServer,
    ProgressMessage,
    ProgressStage,
    ResultMessage,
    ServingError,
    ServingErrorCode,
    run,
)

if TYPE_CHECKING:
    from localmodels.serving import ProgressEmitter

CRASH_EXIT = 3


class EchoServer(ModelServer):
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def load(self, msg: LoadMessage) -> LoadedMessage:
        if msg.options.get("fail"):
            raise RuntimeError("private model path")
        if msg.options.get("fail_dtype") == msg.dtype:
            raise ServingError(ServingErrorCode.NOT_IMPLEMENTED, f"{msg.dtype} is not supported")
        self.options = msg.options
        return LoadedMessage(seconds=0.0, torch_allocated_gib=0.0, torch_reserved_gib=0.0)

    def infer(self, msg: InferMessage, emit_progress: ProgressEmitter) -> ResultMessage:
        params = msg.params
        if params.get("exit"):
            os._exit(CRASH_EXIT)
        if params.get("fail") == "oom":
            raise ServingError(ServingErrorCode.CUDA_OOM, "out of memory")
        if params.get("fail") == "leak":
            raise ValueError("private input text")
        time.sleep(params.get("sleep_s", self.options.get("sleep_s", 0)))
        for step in range(params.get("progress", 0)):
            emit_progress(
                ProgressMessage(request_id=msg.request_id, stage=ProgressStage.DECODE, new_tokens=step + 1)
            )
        return ResultMessage(
            request_id=msg.request_id,
            raw_text=self.options.get("text", msg.input_dir),
            peak_allocated_gib=self.options.get("peak_gib", 0.0),
        )


if __name__ == "__main__":
    sys.exit(run(EchoServer()))
