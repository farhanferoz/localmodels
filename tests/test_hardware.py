import subprocess
import sys

import pytest

import localmodels.hardware as hw_mod
from localmodels.enums import DeviceTier, DeviceType
from localmodels.hardware import inspect_hardware, resolve_device


def test_inspect_hardware_never_raises() -> None:
    report = inspect_hardware()
    assert report.total_memory_gb > 0


def _fake_cuda(monkeypatch, stdout: str, *, run_raises: bool = False) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hw_mod.platform, "system", lambda: "Linux")

    def _run(*_a, **_k):
        if run_raises:
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return type("R", (), {"stdout": stdout})()

    monkeypatch.setattr(hw_mod.subprocess, "run", _run)


def test_unified_memory_smi_uses_system_ram(monkeypatch) -> None:
    _fake_cuda(monkeypatch, "NVIDIA GB10, [N/A]\n")
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 121.7)
    r = inspect_hardware()
    assert r.device is DeviceType.CUDA
    assert r.tier is DeviceTier.FAST_ACCELERATED
    assert any("unified memory" in a for a in r.assumptions)


def test_smi_failure_is_cpu_not_fast_gpu(monkeypatch) -> None:
    _fake_cuda(monkeypatch, "", run_raises=True)
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 64.0)
    r = inspect_hardware()
    assert r.device is DeviceType.CPU


def test_nvml_memory_is_used_when_available(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(hw_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 64.0)
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: ("NVIDIA RTX 4090", 24.0))
    r = inspect_hardware()
    assert r.device is DeviceType.CUDA
    assert r.total_memory_gb == 24.0


def test_sysconf_failure_declares_8gb_fallback(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(hw_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: None)

    def boom(_: str) -> int:
        raise ValueError("no sysconf")

    monkeypatch.setattr(hw_mod.os, "sysconf", boom)
    r = inspect_hardware()
    assert r.total_memory_gb == 8.0
    assert any("8.0 GB" in a for a in r.assumptions)


def test_device_override_cpu_uses_system_ram(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 32.0)
    r = inspect_hardware(device_override="cpu")
    assert r.device is DeviceType.CPU
    assert r.total_memory_gb == 32.0


def test_device_override_cuda_without_real_gpu_falls_back_and_declares_assumption(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: None)
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 16.0)
    r = inspect_hardware(device_override="cuda")
    assert r.device is DeviceType.CUDA
    assert r.total_memory_gb == 16.0
    assert any("requested CUDA but no CUDA memory" in a for a in r.assumptions)


def test_device_override_cuda_with_index_suffix_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: ("NVIDIA RTX 4090", 24.0))
    r = inspect_hardware(device_override="cuda:1")
    assert r.device is DeviceType.CUDA
    assert r.total_memory_gb == 24.0


def test_device_override_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown device override"):
        inspect_hardware(device_override="tpu")


def test_resolve_device_with_override_preserves_index() -> None:
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_without_override_probes_real_hardware() -> None:
    assert resolve_device() in ("cuda", "mps", "cpu")
