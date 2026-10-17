"""Serving a local model of any kind in its own process and Python environment, beside llama-server.

Each model runs as a child process (``client.ModelServerProcess``) started in the environment that has its
libraries, talking one JSON object per line (``protocol``). The child's side is ``server``; ``registry`` maps
a key to the class that implements it, imported on first use; ``host.ModelHost`` keeps one server warm on a
shared GPU under the cross-project lease. The managing side never imports PyTorch: only the child does.
Extracted from service_b, whose imaging models (MedGemma, NeuroVFM, Pillar-0) run this way.
"""

from __future__ import annotations

from .client import DEFAULT_INFER_TIMEOUT_S, DEFAULT_LOAD_TIMEOUT_S, MODEL_SERVER_ENV, ModelServerProcess
from .devices import DEVICE_NAMES, require_device
from .errors import SERVER_ERROR_CODES, ServingCancelled, ServingError, ServingErrorCode
from .host import ModelHost
from .protocol import (
    ErrorMessage,
    InferMessage,
    LoadedMessage,
    LoadMessage,
    Message,
    Op,
    ProgressMessage,
    ProgressStage,
    ReadyMessage,
    ResultMessage,
    ShutdownMessage,
    decode,
    encode,
)
from .registry import LazyRegistry
from .server import ModelServer, ProgressEmitter, block_network, open_load, reply_stream, run, serve
from .tune import (
    Agreement,
    QualityCheck,
    TuneCandidate,
    TuneError,
    TuneMeasurement,
    TuneRecord,
    embedding_cosine,
    hardware_key,
    load_choice,
    run_tune,
    save_record,
    score_agreement,
    text_agreement,
    untuned_message,
)

__all__ = [
    "DEFAULT_INFER_TIMEOUT_S",
    "DEFAULT_LOAD_TIMEOUT_S",
    "DEVICE_NAMES",
    "MODEL_SERVER_ENV",
    "SERVER_ERROR_CODES",
    "Agreement",
    "ErrorMessage",
    "InferMessage",
    "LazyRegistry",
    "LoadMessage",
    "LoadedMessage",
    "Message",
    "ModelHost",
    "ModelServer",
    "ModelServerProcess",
    "Op",
    "ProgressEmitter",
    "ProgressMessage",
    "ProgressStage",
    "QualityCheck",
    "ReadyMessage",
    "ResultMessage",
    "ServingCancelled",
    "ServingError",
    "ServingErrorCode",
    "ShutdownMessage",
    "TuneCandidate",
    "TuneError",
    "TuneMeasurement",
    "TuneRecord",
    "block_network",
    "decode",
    "embedding_cosine",
    "encode",
    "hardware_key",
    "load_choice",
    "open_load",
    "reply_stream",
    "require_device",
    "run",
    "run_tune",
    "save_record",
    "score_agreement",
    "serve",
    "text_agreement",
    "untuned_message",
]
