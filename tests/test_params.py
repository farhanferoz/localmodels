import pytest

from localmodels.enums import DeviceTier, DeviceType, ModelSizeClass, QuantLevel, ServingRole
from localmodels.params import classify_tier, derive_serving_params


@pytest.mark.parametrize(
    ("device", "total_mem_gb", "tier", "advice"),
    [
        (DeviceType.CUDA, 121.7, DeviceTier.FAST_ACCELERATED, False),
        (DeviceType.CUDA, 8.0, DeviceTier.MODERATE_ACCELERATED, True),
        (DeviceType.MPS, 16.0, DeviceTier.MODERATE_ACCELERATED, True),
        (DeviceType.CPU, 64.0, DeviceTier.SLOW_CPU, True),
    ],
    ids=["gb10", "small-cuda", "mps", "cpu"],
)
def test_tier_follows_the_measured_device_and_memory(
    device: DeviceType, total_mem_gb: float, tier: DeviceTier, advice: bool
) -> None:
    decision = classify_tier(device, total_mem_gb)
    assert decision.tier is tier
    assert decision.is_slow is (tier is DeviceTier.SLOW_CPU)
    assert (decision.warning is not None, decision.recommendation is not None) == (advice, advice)


def test_below_the_floor_nothing_is_offloaded_even_on_cuda() -> None:
    assert derive_serving_params(DeviceType.CUDA, 2.9).n_gpu_layers == 0


@pytest.mark.parametrize(
    ("total_mem_gb", "expected_quant", "expected_size_class"),
    [
        (2.9, None, ModelSizeClass.REMOTE_RECOMMENDED),  # just below the floor
        (3.0, QuantLevel.IQ2_XXS, ModelSizeClass.SMALL),  # floor, inclusive
        (5.9, QuantLevel.IQ2_XXS, ModelSizeClass.SMALL),
        (6.0, QuantLevel.Q3_K_M, ModelSizeClass.SMALL),
        (7.9, QuantLevel.Q3_K_M, ModelSizeClass.SMALL),
        (8.0, QuantLevel.Q4_K_M, ModelSizeClass.MEDIUM),
        (15.9, QuantLevel.Q4_K_M, ModelSizeClass.MEDIUM),
        (16.0, QuantLevel.Q4_K_M, ModelSizeClass.LARGE),
        (121.7, QuantLevel.Q4_K_M, ModelSizeClass.LARGE),
    ],
)
def test_derive_serving_params_quant_boundaries_cuda(
    total_mem_gb, expected_quant, expected_size_class
) -> None:
    sp = derive_serving_params(DeviceType.CUDA, total_mem_gb)
    assert sp.recommended_quant is expected_quant
    assert sp.recommended_size_class is expected_size_class


@pytest.mark.parametrize("device", [DeviceType.CUDA, DeviceType.MPS])
def test_derive_serving_params_accelerated_full_offload(device: DeviceType) -> None:
    sp = derive_serving_params(device, 32.0)
    assert sp.n_gpu_layers == -1


def test_derive_serving_params_cpu_never_offloads_and_stays_single_slot() -> None:
    sp_small = derive_serving_params(DeviceType.CPU, 8.0)
    sp_large = derive_serving_params(DeviceType.CPU, 121.7)
    assert sp_small.n_gpu_layers == 0
    assert sp_large.n_gpu_layers == 0
    assert sp_small.n_parallel == 1
    assert sp_large.n_parallel == 1
    assert sp_large.n_ctx > sp_small.n_ctx


def test_derive_serving_params_respects_measured_ceilings() -> None:
    sp = derive_serving_params(DeviceType.CUDA, 1_000_000.0)
    assert sp.n_ctx == 32768
    assert sp.n_parallel == 4
    assert sp.recommended_quant is QuantLevel.Q4_K_M
    assert sp.assumptions  # engineering policy is always declared


def test_role_layouts_are_fixed_and_no_role_is_unchanged() -> None:
    plain = derive_serving_params(DeviceType.CUDA, 100.0)
    assert not plain.kv_unified
    assert derive_serving_params(DeviceType.CUDA, 100.0, role=None) == plain

    interactive = derive_serving_params(DeviceType.CUDA, 100.0, role=ServingRole.INTERACTIVE)
    assert (interactive.kv_unified, interactive.n_parallel, interactive.n_ctx) == (True, 4, 32768)
    assert any("K1.layout" in a and "interactive" in a for a in interactive.assumptions)
    bulk = derive_serving_params(DeviceType.CUDA, 20.0, role=ServingRole.BULK, n_ctx_cap=65536)
    assert (bulk.kv_unified, bulk.n_parallel, bulk.n_ctx) == (True, 16, 65536)
    assert any("K2.layout" in a for a in bulk.assumptions)
