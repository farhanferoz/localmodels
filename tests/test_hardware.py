import subprocess
import sys
import types

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


# --- available_memory: what an allocation can get, not the driver's free count
_MEMINFO = """MemTotal:       127600748 kB
MemFree:         22680460 kB
MemAvailable:   113377280 kB
HugePages_Total:       0
HugePages_Free:        0
Hugepagesize:       2048 kB
"""


def _meminfo(tmp_path, text: str = _MEMINFO) -> str:
    path = tmp_path / "meminfo"
    path.write_text(text)
    return str(path)


def test_host_available_bytes_is_memavailable_not_memfree(tmp_path) -> None:
    assert hw_mod.host_available_bytes(_meminfo(tmp_path)) == 113377280 * 1024


def test_host_available_bytes_with_huge_pages_is_the_free_huge_pages(tmp_path) -> None:
    text = _MEMINFO.replace("HugePages_Total:       0", "HugePages_Total:     100").replace(
        "HugePages_Free:        0", "HugePages_Free:       40"
    )
    assert hw_mod.host_available_bytes(_meminfo(tmp_path, text)) == 40 * 2048 * 1024


def test_host_available_bytes_unreadable_is_none(tmp_path) -> None:
    assert hw_mod.host_available_bytes(str(tmp_path / "absent")) is None


def test_unified_gpu_by_smi_gets_the_host_figure(monkeypatch, tmp_path) -> None:
    _fake_cuda(monkeypatch, "[N/A]\n")  # the GB10: nvidia-smi has no free figure of its own
    got = hw_mod.available_memory("cuda", _meminfo(tmp_path))
    assert got is not None
    assert got.unified
    assert got.bytes == 113377280 * 1024
    assert "MemAvailable" in got.source


def test_unified_gpu_by_nvml_not_supported_gets_the_host_figure(monkeypatch, tmp_path) -> None:
    class NotSupportedError(Exception):
        pass

    def raise_not_supported(_handle):
        raise NotSupportedError

    nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        NVMLError_NotSupported=NotSupportedError,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=raise_not_supported,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: None)
    got = hw_mod.available_memory("cuda:0", _meminfo(tmp_path))
    assert got is not None
    assert got.unified
    assert got.bytes == 113377280 * 1024


def test_discrete_gpu_gets_its_own_free_memory(monkeypatch, tmp_path) -> None:
    _fake_cuda(monkeypatch, "20000\n")
    got = hw_mod.available_memory("cuda:1", _meminfo(tmp_path))
    assert got is not None
    assert not got.unified
    assert got.bytes == 20000 * 1024**2


def test_discrete_gpu_by_nvml_gets_its_own_free_memory(monkeypatch, tmp_path) -> None:
    nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda _h: types.SimpleNamespace(free=7 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    got = hw_mod.available_memory("cuda", _meminfo(tmp_path))
    assert got is not None
    assert not got.unified
    assert got.bytes == 7 * 1024**3


def test_cpu_gets_the_host_figure_and_mps_and_an_unread_gpu_are_unknown(monkeypatch, tmp_path) -> None:
    got = hw_mod.available_memory("cpu", _meminfo(tmp_path))
    assert got is not None
    assert not got.unified
    assert got.bytes == 113377280 * 1024
    assert hw_mod.available_memory("mps", _meminfo(tmp_path)) is None
    _fake_cuda(monkeypatch, "", run_raises=True)
    assert hw_mod.available_memory("cuda", _meminfo(tmp_path)) is None


def test_available_memory_never_raises_on_this_host() -> None:
    hw_mod.available_memory("cuda")
    hw_mod.available_memory("cpu")
