import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import localmodels.hardware as hw_mod
from localmodels.enums import DeviceTier, DeviceType
from localmodels.hardware import inspect_hardware, network_filesystem, resolve_device


@pytest.fixture(autouse=True)
def _every_gpu_visible(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_inspect_hardware_never_raises() -> None:
    report = inspect_hardware()
    assert report.total_memory_gb > 0


def _fake_cuda(monkeypatch, stdout: str, *, run_raises: bool = False) -> list[str]:
    """Fake nvidia-smi; returns the list of GPUs (``-i``) it is asked about."""
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hw_mod.platform, "system", lambda: "Linux")
    asked: list[str] = []

    def _run(argv, *_a, **_k):
        asked.append(argv[argv.index("-i") + 1])
        if run_raises:
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return type("R", (), {"stdout": stdout})()

    monkeypatch.setattr(hw_mod.subprocess, "run", _run)
    return asked


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


_UUID = "GPU-d58339d6-8445-9785-5864-0815072c0f9a"


@pytest.mark.parametrize(
    ("visible", "probed"),
    [
        ("", None),
        ("-1", None),
        ("NoDevFiles", None),
        ("0,0", None),
        ("0abc", "0"),
        ("2,-1", "2"),
        (_UUID, _UUID),
    ],
)
def test_cuda_visible_devices_picks_the_gpu_probed_and_hiding_it_means_cpu(
    monkeypatch, tmp_path, visible: str, probed: str | None
) -> None:
    asked = _fake_cuda(monkeypatch, "NVIDIA GB10, [N/A]\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    report = inspect_hardware()
    if probed is None:  # NVML and nvidia-smi still see the GPU; CUDA, and so torch, does not
        assert report.device is DeviceType.CPU
        assert any("leaves this process no CUDA GPU" in a for a in report.assumptions)
        assert resolve_device() == "cpu"
        assert hw_mod.available_memory("cuda", _meminfo(tmp_path)) is None
        assert asked == []
    else:
        assert report.device is DeviceType.CUDA
        assert set(asked) == {probed}


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


def test_a_path_on_a_network_filesystem_is_recognised_by_its_closest_mount(tmp_path: Path) -> None:
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/nvme0n1p2 / ext4 rw 0 0\n"
        "nas:/share /mnt/nas nfs4 rw 0 0\n"
        "/dev/sdb1 /mnt/nas/cache ext4 rw 0 0\n"
    )
    assert network_filesystem(Path("/mnt/nas/models/m.gguf"), str(mounts)) == "nfs4"
    assert network_filesystem(Path("/mnt/nas/cache/m.gguf"), str(mounts)) is None  # the closest mount decides
    assert network_filesystem(Path("/home/user/m.gguf"), str(mounts)) is None
    assert network_filesystem(Path("/home/user/m.gguf"), str(tmp_path / "absent")) is None


def test_gpu_process_memory_is_per_process_and_none_without_a_reading(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    rows = "4242, 4468\n7, 100\n7, 50\n99, [N/A]\n"  # one process on two GPUs; one with no figure
    monkeypatch.setattr(hw_mod.subprocess, "run", lambda *_a, **_k: type("R", (), {"stdout": rows})())
    assert hw_mod.gpu_process_memory_mib() == {4242: 4468, 7: 150}
    monkeypatch.setattr(hw_mod.subprocess, "run", lambda *_a, **_k: type("R", (), {"stdout": ""})())
    assert hw_mod.gpu_process_memory_mib() == {}  # no process on the card is not "no reading"

    def fails(*_a, **_k):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(hw_mod.subprocess, "run", fails)
    assert hw_mod.gpu_process_memory_mib() is None
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: None)
    assert hw_mod.gpu_process_memory_mib() is None
    # the host side a process holds beside it, from /proc: this process has some, a gone one has no reading
    before = hw_mod.process_anon_mib(os.getpid())
    hog = bytearray(256 * 1024 * 1024)  # 256 MiB of anonymous memory, zeroed so it is resident
    after = hw_mod.process_anon_mib(os.getpid())
    del hog
    assert before is not None
    assert after is not None
    assert after - before >= 200  # the reading follows the process, not a constant
    assert hw_mod.process_anon_mib(2**22 + 1) is None  # above the kernel's pid limit
