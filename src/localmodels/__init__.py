"""localmodels: hardware-aware local model serving, GPU-lease coordination and model catalogue.

Extracted from service_a and service_c (service_a DECISIONS D31), then service_b: torch-free on
the managing side, so any caller can depend on it without pulling in a CUDA build meant for a
different accelerator. Text models run as llama.cpp's ``llama-server`` (never on port 8080); any
other local model runs in its own process through ``localmodels.serving``.
"""

from __future__ import annotations

from .catalogue import LicencePolicy, ModelCategory, ModelSpec, licence_allowed, select
from .enums import DeviceTier, DeviceType, ModelSizeClass, Precision, PreflightVerdict, QuantLevel
from .hardware import (
    AvailableMemory,
    HardwareReport,
    available_memory,
    host_available_bytes,
    inspect_hardware,
    resolve_device,
)
from .launcher import (
    Contention,
    LeaseConflictError,
    LeaseWaitTimeoutError,
    LocalServerManager,
    PortHeldError,
    ReservedPortError,
    ServerStartError,
    StartupYieldedError,
    SupervisorExit,
    supervise,
)
from .lease import DEFAULT_LEASE_DIR, GpuLease, GpuLeaseClient, LeaseRecord
from .params import ServingParams, TierDecision, classify_tier, derive_serving_params
from .pinned import PinnedFile, PinnedModel, fetch_snapshot, hash_snapshot, verify_files
from .preflight import PreflightResult, preflight, print_preflight
from .process import (
    BinaryNotFoundError,
    LocalModelsError,
    ServerHandle,
    ServerSpec,
    ServerStatus,
    build_args,
    resolve_llama_server_bin,
)
from .pull import check_updates, model_status, pull_model, search_hub

__all__ = [
    "DEFAULT_LEASE_DIR",
    "AvailableMemory",
    "BinaryNotFoundError",
    "Contention",
    "DeviceTier",
    "DeviceType",
    "GpuLease",
    "GpuLeaseClient",
    "HardwareReport",
    "LeaseConflictError",
    "LeaseRecord",
    "LeaseWaitTimeoutError",
    "LicencePolicy",
    "LocalModelsError",
    "LocalServerManager",
    "ModelCategory",
    "ModelSizeClass",
    "ModelSpec",
    "PinnedFile",
    "PinnedModel",
    "PortHeldError",
    "Precision",
    "PreflightResult",
    "PreflightVerdict",
    "QuantLevel",
    "ReservedPortError",
    "ServerHandle",
    "ServerSpec",
    "ServerStartError",
    "ServerStatus",
    "ServingParams",
    "StartupYieldedError",
    "SupervisorExit",
    "TierDecision",
    "available_memory",
    "build_args",
    "check_updates",
    "classify_tier",
    "derive_serving_params",
    "fetch_snapshot",
    "hash_snapshot",
    "host_available_bytes",
    "inspect_hardware",
    "licence_allowed",
    "model_status",
    "preflight",
    "print_preflight",
    "pull_model",
    "resolve_device",
    "resolve_llama_server_bin",
    "search_hub",
    "select",
    "supervise",
    "verify_files",
]
