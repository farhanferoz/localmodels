import pytest

from localmodels.enums import DeviceTier, DeviceType, ModelSizeClass, QuantLevel, ServingRole
from localmodels.params import classify_tier, derive_serving_params


def test_gb10_class_memory_is_fast_tier() -> None:
    assert classify_tier(DeviceType.CUDA, 121.7).tier is DeviceTier.FAST_ACCELERATED


def test_small_cuda_is_moderate() -> None:
    decision = classify_tier(DeviceType.CUDA, 8.0)
    assert decision.tier is DeviceTier.MODERATE_ACCELERATED
    assert decision.warning is not None
    assert decision.recommendation is not None


def test_cpu_is_slow() -> None:
    decision = classify_tier(DeviceType.CPU, 64.0)
    assert decision.tier is DeviceTier.SLOW_CPU
    assert decision.is_slow is True


def test_mps_is_moderate() -> None:
    assert classify_tier(DeviceType.MPS, 16.0).tier is DeviceTier.MODERATE_ACCELERATED


def test_serving_params_gb10() -> None:
    p = derive_serving_params(DeviceType.CUDA, 121.7)
    assert p.n_gpu_layers == -1
    assert p.n_ctx == 32768
    assert p.n_parallel == 4
    assert p.recommended_quant is QuantLevel.Q4_K_M
    assert p.recommended_size_class is ModelSizeClass.LARGE
    assert p.assumptions  # engineering policy is always declared


def test_tiny_memory_recommends_remote() -> None:
    p = derive_serving_params(DeviceType.CPU, 2.0)
    assert p.recommended_size_class is ModelSizeClass.REMOTE_RECOMMENDED
    assert p.recommended_quant is None
    assert p.n_gpu_layers == 0


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
