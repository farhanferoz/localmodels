"""Preflight: can this host run local models at all, and how should it be configured.

service_c's gate + tuning contract (its ``env.py``), rehomed here torch-free: this is the more
capable of the two projects' checks (service_a had none) so it wins outright (commit body).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from .enums import PreflightVerdict
from .hardware import HardwareReport, inspect_hardware
from .params import MIN_RUNNABLE_MEMORY_GB, ServingParams, classify_tier, derive_serving_params

if TYPE_CHECKING:
    from typing import TextIO


class PreflightResult(BaseModel):
    """The verdict, the measurements behind it, and an actionable remedy when it is not READY."""

    verdict: PreflightVerdict
    report: HardwareReport
    serving_params: ServingParams
    message: str
    remedy: str | None


def preflight(report: HardwareReport | None = None, *, device_override: str | None = None) -> PreflightResult:
    """Gate + tuning in one call: "can this run, and if so, how should it be configured".

    Reuses ``classify_tier``'s own warning signal rather than re-deriving a second notion of
    "degraded". The only additional gate is LOCAL_MODELS_INFEASIBLE, for memory below
    ``MIN_RUNNABLE_MEMORY_GB`` regardless of device — distinct from DEGRADED because it means no
    local GGUF quantisation is practical at all, not merely slow.

    Remedies always name an actionable alternative, never a bypass flag.
    """
    if report is None:
        report = inspect_hardware(device_override)

    decision = classify_tier(report.device, report.total_memory_gb)
    params = derive_serving_params(report.device, report.total_memory_gb)

    remedy: str | None
    if params.recommended_quant is None:
        verdict = PreflightVerdict.LOCAL_MODELS_INFEASIBLE
        message = (
            f"{report.total_memory_gb:.1f} GB is below the {MIN_RUNNABLE_MEMORY_GB:.1f} GB "
            "floor assumed necessary for any local GGUF quantisation."
        )
        remedy = "Configure a remote OpenAI-compatible endpoint, or run on a machine with more memory."
    elif decision.warning is not None:
        verdict = PreflightVerdict.DEGRADED
        message = decision.warning
        remedy = decision.recommendation
    else:
        verdict = PreflightVerdict.READY
        message = f"Hardware supports full local model serving ({decision.tier.value})."
        remedy = None

    return PreflightResult(
        verdict=verdict, report=report, serving_params=params, message=message, remedy=remedy
    )


def print_preflight(result: PreflightResult, stream: TextIO | None = None) -> None:
    """Print a preflight verdict in service_c's existing ``[OK]``/``[WARNING]``/``[ADVICE]`` style."""
    import sys  # noqa: PLC0415 — only needed for the default stream

    out = stream or sys.stderr
    if result.verdict is PreflightVerdict.READY:
        print(f"\n[OK] {result.message}\n", file=out)
    else:
        print(f"\n[WARNING] {result.message}\n[ADVICE]  {result.remedy}\n", file=out)
