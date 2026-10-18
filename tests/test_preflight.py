import io

import pytest

from localmodels.enums import DeviceType, PreflightVerdict, QuantLevel
from localmodels.hardware import HardwareReport
from localmodels.params import classify_tier
from localmodels.preflight import preflight, print_preflight


def _fake_report(device: DeviceType, total_mem_gb: float) -> HardwareReport:
    """Build a HardwareReport the way inspect_hardware() would, without probing real hardware."""
    tier = classify_tier(device, total_mem_gb).tier
    return HardwareReport(
        platform="linux",
        arch="x86_64",
        device=device,
        device_name=f"fake-{device.value}",
        total_memory_gb=total_mem_gb,
        tier=tier,
        assumptions=[],
    )


def test_preflight_ready_on_fast_accelerated_cuda() -> None:
    result = preflight(report=_fake_report(DeviceType.CUDA, 32.0))
    assert result.verdict is PreflightVerdict.READY
    assert result.remedy is None
    assert result.serving_params.recommended_quant is QuantLevel.Q4_K_M


@pytest.mark.parametrize("device", [DeviceType.MPS, DeviceType.CPU])
def test_preflight_degraded_names_a_remedy_not_a_bypass(device: DeviceType) -> None:
    result = preflight(report=_fake_report(device, 8.0))
    assert result.verdict is PreflightVerdict.DEGRADED
    assert result.remedy is not None
    assert "--force" not in result.remedy
    assert "--skip" not in result.remedy
    if device is DeviceType.CPU:
        assert "remote" in result.remedy.lower()


@pytest.mark.parametrize("device", [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU])
def test_preflight_local_models_infeasible_below_floor(device: DeviceType) -> None:
    result = preflight(report=_fake_report(device, 1.0))
    assert result.verdict is PreflightVerdict.LOCAL_MODELS_INFEASIBLE
    assert result.serving_params.recommended_quant is None
    assert "remote" in result.remedy.lower()


def test_preflight_verdict_flips_at_min_runnable_memory_boundary() -> None:
    below = preflight(report=_fake_report(DeviceType.CUDA, 2.9))
    at_floor = preflight(report=_fake_report(DeviceType.CUDA, 3.0))
    assert below.verdict is PreflightVerdict.LOCAL_MODELS_INFEASIBLE
    assert at_floor.verdict is not PreflightVerdict.LOCAL_MODELS_INFEASIBLE


def test_print_preflight_tags_a_ready_and_a_degraded_verdict() -> None:
    ready, degraded = io.StringIO(), io.StringIO()
    print_preflight(preflight(report=_fake_report(DeviceType.CUDA, 32.0)), stream=ready)
    print_preflight(preflight(report=_fake_report(DeviceType.CPU, 8.0)), stream=degraded)
    assert "[OK]" in ready.getvalue()
    assert "[WARNING]" in degraded.getvalue()
    assert "[ADVICE]" in degraded.getvalue()
