"""localmodels: hardware-aware local LLM serving, GPU-lease coordination and model catalogue.

Extracted from service_a and service_c (service_a DECISIONS D31): torch-free, so any caller
can depend on it without pulling in a CUDA build meant for a different accelerator. Local model
servers are always llama.cpp's ``llama-server``; this package never starts anything on port 8080.
"""

from __future__ import annotations

from .catalogue import LicencePolicy, ModelCategory, ModelSpec, licence_allowed, select
from .enums import DeviceTier, DeviceType, ModelSizeClass, PreflightVerdict, QuantLevel
from .hardware import HardwareReport, inspect_hardware, resolve_device
from .launcher import (
    Contention,
    LeaseConflictError,
    LeaseWaitTimeoutError,
    LocalServerManager,
    PortHeldError,
    ReservedPortError,
    ServerStartError,
    SupervisorExit,
    supervise,
)
from .lease import GpuLease, LeaseRecord
from .params import ServingParams, TierDecision, classify_tier, derive_serving_params
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
    "BinaryNotFoundError",
    "Contention",
    "DeviceTier",
    "DeviceType",
    "GpuLease",
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
    "PortHeldError",
    "PreflightResult",
    "PreflightVerdict",
    "QuantLevel",
    "ReservedPortError",
    "ServerHandle",
    "ServerSpec",
    "ServerStartError",
    "ServerStatus",
    "ServingParams",
    "SupervisorExit",
    "TierDecision",
    "build_args",
    "check_updates",
    "classify_tier",
    "derive_serving_params",
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
]
