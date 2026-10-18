"""Serving-tier and serving-parameter derivation, torch-free.

Invariant (inherited from service_c DECISIONS D20/D21, carried into service_a and now here):
serving parameters are a pure function of MEASURED device type and MEASURED total memory. Never
key on a card name. Every engineering-policy threshold is echoed in ``assumptions``/``warning``
so it is never mistaken for a measurement.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import DeviceTier, DeviceType, ModelSizeClass, QuantLevel, ServingRole

MIN_RUNNABLE_MEMORY_GB = 3.0
_FAST_VRAM_GB = 16.0
_MEDIUM_MEMORY_GB = _FAST_VRAM_GB / 2.0
_SMALL_MEMORY_GB = MIN_RUNNABLE_MEMORY_GB * 2.0
_REFERENCE_MEMORY_GB = _FAST_VRAM_GB
_REFERENCE_N_CTX = 8192
_REFERENCE_N_PARALLEL = 2
_MIN_N_CTX = 2048
_MAX_N_CTX = 32768
_MIN_N_PARALLEL = 1
_MAX_N_PARALLEL = 4
_ROLE_N_CTX = 32768
_ROLE_N_PARALLEL = {ServingRole.INTERACTIVE: 4, ServingRole.BULK: 16}
_ROLE_RULE = {ServingRole.INTERACTIVE: "K1.layout", ServingRole.BULK: "K2.layout"}


class TierDecision(BaseModel):
    """A serving tier plus the operator-facing verdict text that goes with it."""

    tier: DeviceTier
    is_slow: bool
    warning: str | None
    recommendation: str | None


class ServingParams(BaseModel):
    """Serving parameters derived from measured hardware; assumptions always declared."""

    n_gpu_layers: int
    n_ctx: int
    n_parallel: int
    recommended_quant: QuantLevel | None
    recommended_size_class: ModelSizeClass
    kv_unified: bool = False
    assumptions: list[str] = Field(default_factory=list)


def classify_tier(device: DeviceType, total_mem_gb: float) -> TierDecision:
    """Map measured device and memory to a serving tier; never keyed on a card name."""
    if device is DeviceType.CUDA and total_mem_gb >= _FAST_VRAM_GB:
        return TierDecision(
            tier=DeviceTier.FAST_ACCELERATED, is_slow=False, warning=None, recommendation=None
        )
    if device in (DeviceType.CUDA, DeviceType.MPS):
        return TierDecision(
            tier=DeviceTier.MODERATE_ACCELERATED,
            is_slow=False,
            warning="Moderate accelerator detected; large model contexts may be constrained.",
            recommendation="Use 4-bit quantised models (Q4_K_M) or MoE architectures.",
        )
    return TierDecision(
        tier=DeviceTier.SLOW_CPU,
        is_slow=True,
        warning="No GPU accelerator detected (running on CPU). Local inference will be very slow.",
        recommendation="Configure a remote OpenAI-compatible endpoint, or run on a GPU-equipped machine.",
    )


def _clamp(value: float, lo: int, hi: int) -> int:
    return min(max(round(value), lo), hi)


def derive_serving_params(
    device: DeviceType,
    total_mem_gb: float,
    *,
    role: ServingRole | None = None,
    n_ctx_cap: int | None = None,
) -> ServingParams:
    """Derive llama-server serving parameters as a pure function of measured hardware.

    With a ``role`` the layout is the role's fixed one (rules K1.layout, K2.layout): a shared pool
    (``kv_unified``), the role's slot count and ``n_ctx_cap`` (default 32,768) as the pool, never
    derived from free memory. Without one the result is the memory-scaled layout.
    """
    params = _derive_scaled(device, total_mem_gb)
    if role is None:
        return params
    n_ctx = n_ctx_cap or _ROLE_N_CTX
    n_parallel = _ROLE_N_PARALLEL[role]
    params.assumptions.append(
        f"{role.value} role: shared pool (kv_unified), {n_parallel} slots, n_ctx {n_ctx} fixed "
        f"({_ROLE_RULE[role]}) (ASSUMPTION)."
    )
    return params.model_copy(update={"kv_unified": True, "n_ctx": n_ctx, "n_parallel": n_parallel})


def _derive_scaled(device: DeviceType, total_mem_gb: float) -> ServingParams:
    assumptions: list[str] = []
    if total_mem_gb < MIN_RUNNABLE_MEMORY_GB:
        assumptions.append(
            f"{total_mem_gb:.1f} GB is below the {MIN_RUNNABLE_MEMORY_GB:.1f} GB floor "
            "assumed for any local GGUF (ASSUMPTION)."
        )
        return ServingParams(
            n_gpu_layers=0,
            n_ctx=_MIN_N_CTX,
            n_parallel=_MIN_N_PARALLEL,
            recommended_quant=None,
            recommended_size_class=ModelSizeClass.REMOTE_RECOMMENDED,
            assumptions=assumptions,
        )
    scale = total_mem_gb / _REFERENCE_MEMORY_GB
    if device is DeviceType.CPU:
        n_gpu_layers, n_parallel = 0, _MIN_N_PARALLEL
        n_ctx = _clamp(_REFERENCE_N_CTX * scale, _MIN_N_CTX, _MAX_N_CTX // 4)
        assumptions.append("CPU tier: n_parallel fixed at 1 and n_ctx capped at a quarter (ASSUMPTION).")
    else:
        n_gpu_layers = -1
        n_ctx = _clamp(_REFERENCE_N_CTX * scale, _MIN_N_CTX, _MAX_N_CTX)
        n_parallel = _clamp(_REFERENCE_N_PARALLEL * scale, _MIN_N_PARALLEL, _MAX_N_PARALLEL)
    if total_mem_gb >= _FAST_VRAM_GB:
        quant, size = QuantLevel.Q4_K_M, ModelSizeClass.LARGE
    elif total_mem_gb >= _MEDIUM_MEMORY_GB:
        quant, size = QuantLevel.Q4_K_M, ModelSizeClass.MEDIUM
    elif total_mem_gb >= _SMALL_MEMORY_GB:
        quant, size = QuantLevel.Q3_K_M, ModelSizeClass.SMALL
    else:
        quant, size = QuantLevel.IQ2_XXS, ModelSizeClass.SMALL
    assumptions.append(
        f"Quant/size thresholds anchored on service_c's validated Q4_K_M at >= {_FAST_VRAM_GB:.0f} GB "
        "(ASSUMPTION)."
    )
    return ServingParams(
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_parallel=n_parallel,
        recommended_quant=quant,
        recommended_size_class=size,
        assumptions=assumptions,
    )
