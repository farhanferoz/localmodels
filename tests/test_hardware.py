import ctypes
import errno
import fcntl
import os
import struct
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath
from typing import NamedTuple

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
        ("9" * 5000, None),  # past Python's digit limit for int(): no GPU, never a ValueError
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


# --- AMD GPUs under ROCm ----------------------------------------------------------------------------------

_GIB_BYTES = 1024**3
_AMD_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_UID = (0x4B2C5E6F70810001, 0x4B2C5E6F70810002)  # two GPUs whose UUIDs share a prefix, as boards of a kind do


class FakeAmdGpu(NamedTuple):
    """One GPU of ``fake_rocm``. ``fusion`` None: the driver does not answer; ``openable`` False: no node."""

    fusion: bool | None = True
    vram_gib: float = 1.0
    vram_used_gib: float = 0.0
    gtt_gib: float = 109.0
    gtt_used_gib: float = 0.0
    unique_id: int = 0
    device_id: int = 0x1586
    gfx_target_version: int = 110501
    openable: bool = True


def fake_rocm(root, monkeypatch, *gpus: FakeAmdGpu, kfd_access: bool = True, runtime: bool = True) -> None:
    """KFD, /dev and the driver's answers for ``gpus`` (default: one M5 APU), behind a CPU node 0.

    KFD's local_mem_size is 0 for every GPU, as the kernel prints it, so only the driver's FUSION flag can
    tell an APU from a discrete card. A render node file holds what the fake driver answers: device id and
    flags.
    """
    gpus = gpus or (FakeAmdGpu(),)
    nodes, drm, dri = root / "nodes", root / "drm", root / "dri"
    (nodes / "0").mkdir(parents=True)
    (nodes / "0" / "properties").write_text("simd_count 0\nvendor_id 0\ngpu_id 0\n")
    dri.mkdir()
    for i, gpu in enumerate(gpus, start=1):
        minor = 127 + i
        (nodes / str(i)).mkdir()
        (nodes / str(i) / "properties").write_text(
            f"simd_count 80\ngfx_target_version {gpu.gfx_target_version}\nvendor_id 4098\n"
            f"device_id {gpu.device_id}\nlocal_mem_size 0\ndrm_render_minor {minor}\n"
            f"unique_id {gpu.unique_id}\n"
        )
        if gpu.openable:
            answer = "none" if gpu.fusion is None else f"{gpu.device_id} {int(gpu.fusion)}"
            (dri / f"renderD{minor}").write_text(answer)
        dev = drm / f"renderD{minor}" / "device"
        dev.mkdir(parents=True)
        for name, gib in (
            ("mem_info_gtt_total", gpu.gtt_gib),
            ("mem_info_gtt_used", gpu.gtt_used_gib),
            ("mem_info_vram_total", gpu.vram_gib),
            ("mem_info_vram_used", gpu.vram_used_gib),
        ):
            (dev / name).write_text(f"{int(gib * _GIB_BYTES)}\n")
    hw_mod._driver_fusion.cache_clear()  # kept per process; this tree is new
    kfd = root / "kfd"
    kfd.write_text("")
    kfd.chmod(0o600 if kfd_access else 0o000)

    def dev_info(fd: int) -> tuple[int, int] | None:
        words = os.pread(fd, 64, 0).decode().split()
        return None if words == ["none"] else (int(words[0]), int(words[1]))

    monkeypatch.setattr(hw_mod, "_KFD_NODES", nodes)
    monkeypatch.setattr(hw_mod, "_DRM", drm)
    monkeypatch.setattr(hw_mod, "_DRI", dri)
    monkeypatch.setattr(hw_mod, "_KFD_DEVICE", kfd)
    monkeypatch.setattr(hw_mod, "_amdgpu_dev_info", dev_info)
    monkeypatch.setattr(hw_mod, "_rocm_runtime", lambda: runtime)
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: None)  # no NVIDIA GPU
    monkeypatch.setattr(hw_mod.shutil, "which", lambda _: None)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(hw_mod, "_system_memory_gb", lambda: 122.0)
    for var in _AMD_VARS:
        monkeypatch.delenv(var, raising=False)


def _visible() -> list[int]:
    """The render minors of the GPUs this process sees, in HIP's order."""
    return [gpu.render_minor for i in range(4) if (gpu := hw_mod._amd_gpu(i)) is not None]


def _meminfo_gib(tmp_path, gib: float) -> str:
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: 134217728 kB\nMemAvailable: {int(gib * 1024**2)} kB\n")
    return str(path)


def test_the_real_paths_are_the_kernels(monkeypatch) -> None:
    monkeypatch.undo()  # conftest's no_real_amd_gpu points them at an empty tree for every test
    assert Path("/sys/class/kfd/kfd/topology/nodes") == hw_mod._KFD_NODES
    assert Path("/dev/kfd") == hw_mod._KFD_DEVICE
    assert Path("/dev/dri") == hw_mod._DRI
    assert Path("/sys/class/drm") == hw_mod._DRM


def test_gfx_name_follows_llvm_target_names() -> None:
    assert hw_mod._gfx_name(110501) == "gfx1151"
    assert hw_mod._gfx_name(90402) == "gfx942"
    assert hw_mod._gfx_name(90010) == "gfx90a"


def test_an_amd_apu_is_a_cuda_device_with_its_gtt_limit_as_memory(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    rep = inspect_hardware()
    assert rep.device == DeviceType.CUDA
    assert rep.device_name == "AMD GPU gfx1151 (1002:1586, ROCm)"
    assert rep.total_memory_gb == 109.0
    assert rep.tier == DeviceTier.FAST_ACCELERATED
    assert rep.assumptions == []
    assert resolve_device(None) == "cuda"
    assert hw_mod.is_rocm_device(rep.device_name)


def test_a_discrete_card_is_its_vram_though_kfd_reports_no_local_memory(tmp_path, monkeypatch) -> None:
    """A 24 GiB card with 20 GiB in use on a 128 GiB host: 24 GiB with 4 GiB free, not the GTT limit."""
    card = FakeAmdGpu(fusion=False, vram_gib=24.0, vram_used_gib=20.0, gtt_gib=62.0, device_id=0x744C)
    fake_rocm(tmp_path, monkeypatch, card)
    assert inspect_hardware().total_memory_gb == 24.0
    got = hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 120))
    assert got is not None
    assert not got.unified
    assert got.bytes == 4 * _GIB_BYTES


@pytest.mark.parametrize("answer", ["none", "29772 1"])  # no answer; an answer for another device than KFD's
def test_a_gpu_the_driver_does_not_vouch_for_is_taken_as_discrete(tmp_path, monkeypatch, answer) -> None:
    fake_rocm(tmp_path, monkeypatch, FakeAmdGpu(vram_gib=1.0, gtt_gib=109.0))
    (tmp_path / "dri" / "renderD128").write_text(answer)
    assert inspect_hardware().total_memory_gb == 1.0  # its VRAM: never the GTT limit on a guess


def test_a_missing_render_node_is_left_out_and_a_refused_one_leaves_no_gpu(tmp_path, monkeypatch) -> None:
    fake_rocm(
        tmp_path / "a", monkeypatch, FakeAmdGpu(openable=False), FakeAmdGpu(fusion=False, vram_gib=16.0)
    )
    assert _visible() == [129]  # ENOENT: the thunk skips the GPU and the next takes its number
    assert inspect_hardware().total_memory_gb == 16.0
    if os.geteuid() != 0:  # root opens a mode-000 file anyway
        fake_rocm(tmp_path / "b", monkeypatch, FakeAmdGpu(), FakeAmdGpu(fusion=False, vram_gib=16.0))
        (tmp_path / "b" / "dri" / "renderD128").chmod(0o000)
        assert _visible() == []  # EACCES fails the thunk's whole topology
    fake_rocm(tmp_path / "c", monkeypatch, FakeAmdGpu(), FakeAmdGpu(fusion=False, vram_gib=16.0))
    real_open = os.open

    def refused_by_cgroup(path, *args, **kwargs):
        if Path(path).name == "renderD128":
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(hw_mod.os, "open", refused_by_cgroup)
    assert _visible() == [129]  # EPERM (a device cgroup): skipped, as ENOENT is
    monkeypatch.setattr(hw_mod.os, "open", real_open)
    nodes = tmp_path / "c" / "nodes" / "1" / "properties"
    nodes.write_text(nodes.read_text().replace("drm_render_minor 128", "drm_render_minor 0"))
    assert _visible() == []  # a GPU without a render node fails the thunk's whole topology


def test_the_driver_is_asked_once_and_a_node_that_appears_later_is_found(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch, FakeAmdGpu(openable=False))
    assert _visible() == []
    (tmp_path / "dri" / "renderD128").write_text(f"{0x1586} 1")  # the node appears (a late udev rule)
    asked = []
    answer = hw_mod._amdgpu_dev_info
    monkeypatch.setattr(hw_mod, "_amdgpu_dev_info", lambda fd: asked.append(fd) or answer(fd))
    assert _visible() == [128]
    assert _visible() == [128]
    assert len(asked) == 1  # opening again would wake a runtime-suspended card


def test_no_access_to_kfd_or_no_rocm_runtime_means_cpu(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path / "a", monkeypatch, kfd_access=False)
    if os.geteuid() != 0:  # root opens a mode-000 file anyway
        assert inspect_hardware().device == DeviceType.CPU
    fake_rocm(tmp_path / "b", monkeypatch, runtime=False)
    assert inspect_hardware().device == DeviceType.CPU
    assert hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 100)) is None


def test_no_kfd_means_no_amd_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hw_mod, "_KFD_NODES", tmp_path / "absent")
    assert hw_mod._amd_gpus() == []
    (tmp_path / "properties").write_bytes(b"simd_count \xff\n")
    assert hw_mod._kfd_properties(tmp_path) == {}  # not what a kernel prints, and not an exception either


def test_the_ordinal_picks_the_gpu(tmp_path, monkeypatch) -> None:
    fake_rocm(
        tmp_path,
        monkeypatch,
        FakeAmdGpu(fusion=False, vram_gib=24.0, vram_used_gib=4.0),
        FakeAmdGpu(fusion=False, vram_gib=16.0, vram_used_gib=1.0),
    )
    meminfo = _meminfo_gib(tmp_path, 100)
    free = [hw_mod.available_memory(f"cuda:{i}", meminfo) for i in range(3)]
    assert [None if got is None else got.bytes for got in free] == [20 * _GIB_BYTES, 15 * _GIB_BYTES, None]


def test_available_memory_on_an_amd_apu_is_capped_by_the_gtt_room(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch, FakeAmdGpu(gtt_gib=109.0, gtt_used_gib=9.0))
    got = hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 120))
    assert got is not None
    assert got.unified
    assert got.bytes == 100 * _GIB_BYTES
    assert "GTT" in got.source


def test_available_memory_on_an_amd_apu_is_the_host_when_that_is_smaller(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    got = hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 10))
    assert got is not None
    assert got.bytes == 10 * _GIB_BYTES
    assert "MemAvailable" in got.source


_TWO = (FakeAmdGpu(unique_id=_UID[0]), FakeAmdGpu(unique_id=_UID[1]))


@pytest.mark.parametrize(
    ("value", "seen"),
    [
        (None, [128, 129]),
        ("", []),  # set but empty: ROCr shows no GPU
        ("1", [129]),
        ("1,0", [129, 128]),
        (" 1 ", [129]),  # entries are trimmed
        ("0x1", [129]),  # strtol, base 0
        ("01", [129]),  # octal
        ("0abc", []),  # strtol must read the whole entry
        ("0,0,1", [128]),  # a repeat ends the list
        ("0,5,1", [128]),  # so does an index past the last GPU
        ("0,1,junk", [128, 129]),  # entries past the number of GPUs are not read
        ("0,,1", []),  # an empty entry among those read: ROCr's parse throws, so no GPU
        ("0, ,1", []),
        ("9" * 5000, []),  # past Python's digit limit for int()
        (f"GPU-{_UID[1]:016x}"[:20], [129]),  # a UUID prefix, any case
        ("GPU-4B2C", []),  # a prefix of both UUIDs names neither
        ("GPU-", []),  # shorter than five characters
    ],
)
def test_rocr_visible_devices_follows_rocr(tmp_path, monkeypatch, value, seen) -> None:
    fake_rocm(tmp_path, monkeypatch, *_TWO)
    if value is not None:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", value)
    assert _visible() == seen


@pytest.mark.parametrize(
    ("hip", "cuda", "seen"),
    [
        (None, None, [128, 129]),
        ("", None, []),  # set but empty reads as " " in HIP's flags: no GPU (measured, HIP 7.16)
        (None, "", []),
        (" ", None, []),
        ("1", None, [129]),
        ("1,0", None, [129, 128]),
        ("0,0,1", None, [128, 129]),  # a repeat is skipped, the list goes on
        ("00", None, []),  # only the exact decimal index
        ("+1", None, []),
        ("0abc", None, []),
        ("0,x,1", None, [128]),  # the first bad entry ends the list
        ("9" * 5000, None, []),  # past Python's digit limit for int()
        ('"1"', None, [129]),  # HIP's flags drop surrounding double quotes
        (f"GPU-{_UID[1]:016x}", None, [129]),
        (f"{_UID[1]:016x}"[-8:], None, []),  # a piece of a UUID without "GPU-" is read as an index
        (f"GPU-{_UID[1]:016X}", None, []),  # HIP's match is case-sensitive, its UUIDs lower-case
        ("1", "0", [129]),  # HIP_VISIBLE_DEVICES wins; the two are never chained
        (None, "1", [129]),
        ("", "1", []),  # set, so HIP_VISIBLE_DEVICES still wins
    ],
)
def test_hip_and_cuda_visible_devices_follow_hip(tmp_path, monkeypatch, hip, cuda, seen) -> None:
    fake_rocm(tmp_path, monkeypatch, *_TWO)
    for var, value in (("HIP_VISIBLE_DEVICES", hip), ("CUDA_VISIBLE_DEVICES", cuda)):
        if value is not None:
            monkeypatch.setenv(var, value)
    assert _visible() == seen


def test_a_repeat_ends_rocrs_list_where_hip_skips_it(tmp_path, monkeypatch) -> None:
    fake_rocm(
        tmp_path, monkeypatch, FakeAmdGpu(), FakeAmdGpu(), FakeAmdGpu()
    )  # three, so "0,0,1" is read whole
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,0,1")
    assert _visible() == [128]
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,0,1")
    assert _visible() == [128, 129]


def test_hip_numbers_what_rocr_left(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch, *_TWO)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert _visible() == [129]
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert _visible() == []


def test_a_cuda_override_on_an_amd_gpu_measures_it(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    rep = inspect_hardware("cuda:0")
    assert rep.device_name.startswith("AMD GPU gfx1151")
    assert rep.total_memory_gb == 109.0
    assert rep.assumptions == []


# --- a host with both vendors: "cuda" is NVIDIA's, and AMD never answers for it


def test_nvidia_wins_on_a_host_with_both(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    _fake_cuda(monkeypatch, "NVIDIA RTX 4090, 24564\n")
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: ("NVIDIA RTX 4090", 24.0))
    assert inspect_hardware().device_name == "NVIDIA RTX 4090"
    assert hw_mod._amd_gpu(0) is None


def test_a_failing_nvidia_smi_is_not_replaced_by_the_amd_gpu(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    _fake_cuda(monkeypatch, "", run_raises=True)
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: None)
    assert inspect_hardware().device == DeviceType.CPU
    assert hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 100)) is None


def test_nvml_without_nvidia_smi_is_still_nvidia(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda _h: types.SimpleNamespace(free=7 * _GIB_BYTES),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    got = hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 100))
    assert got is not None
    assert got.bytes == 7 * _GIB_BYTES  # NVML's figure, not the AMD APU's


def test_a_unified_nvidia_gpu_is_not_capped_by_an_amd_apus_gtt(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch, FakeAmdGpu(gtt_gib=20.0, gtt_used_gib=10.0))
    _fake_cuda(monkeypatch, "[N/A]\n")  # a GB10: unified, no free figure of its own
    got = hw_mod.available_memory("cuda", _meminfo_gib(tmp_path, 100))
    assert got is not None
    assert got.bytes == 100 * _GIB_BYTES


def test_a_cuda_override_on_a_unified_nvidia_gpu_never_reports_the_amd_one(tmp_path, monkeypatch) -> None:
    fake_rocm(tmp_path, monkeypatch)
    _fake_cuda(monkeypatch, "NVIDIA GB10, [N/A]\n")
    monkeypatch.setattr(hw_mod, "_cuda_memory_gb", lambda: ("NVIDIA GB10", None))
    rep = inspect_hardware("cuda")
    assert not rep.device_name.startswith("AMD")
    assert rep.total_memory_gb == 122.0


# --- the ROCm runtime and the driver query


def test_a_rocm_runtime_is_found_where_rocm_puts_it(tmp_path, monkeypatch) -> None:
    versions: dict[str, str] = {}

    def version(dist: str) -> str:
        if dist not in versions:
            raise hw_mod.importlib.metadata.PackageNotFoundError(dist)
        return versions[dist]

    def files(dist: str) -> list[PurePosixPath]:
        if dist not in core_files:
            raise hw_mod.importlib.metadata.PackageNotFoundError(dist)
        return [PurePosixPath(name) for name in core_files[dist]]

    def found() -> bool:  # the answer is kept per process; each step here is another environment
        hw_mod._rocm_runtime.cache_clear()
        return hw_mod._rocm_runtime()

    core_files: dict[str, list[str]] = {}
    monkeypatch.setattr(hw_mod.importlib.metadata, "version", version)
    monkeypatch.setattr(hw_mod.importlib.metadata, "files", files)
    monkeypatch.setattr(hw_mod, "_ROCM_SYSTEM_ROOT", tmp_path / "opt-rocm")
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(hw_mod.ctypes.util, "find_library", lambda _: None)
    assert not found()
    versions["torch"] = "2.14.0+cu130"
    assert not found()
    versions["torch"] = "2.14.0+rocm10.1.0"
    assert found()
    versions.clear()
    core_files["rocm-sdk-core"] = ["rocm_sdk_core-0.1.0.dist-info/METADATA"]  # PyPI's empty placeholder
    assert not found()
    core_files["rocm-sdk-core"].append("_rocm_sdk_core/lib/libamdhip64.so.7")  # TheRock's
    assert found()
    core_files.clear()
    lib = tmp_path / "rocm" / "lib"
    lib.mkdir(parents=True)
    (lib / "libamdhip64.so.7").write_text("")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    assert found()
    monkeypatch.delenv("ROCM_PATH")
    monkeypatch.setattr(hw_mod, "_ROCM_SYSTEM_ROOT", tmp_path / "rocm")
    assert found()
    monkeypatch.setattr(hw_mod, "_ROCM_SYSTEM_ROOT", tmp_path / "opt-rocm")
    searched: list[str] = []
    monkeypatch.setattr(
        hw_mod.ctypes.util, "find_library", lambda name: searched.append(name) or "libamdhip64.so.6"
    )
    assert found()
    assert hw_mod._rocm_runtime()
    assert searched == ["amdhip64"]  # kept: a miss runs ldconfig and gcc (~25 ms) on every call otherwise


def test_the_driver_query_is_laid_out_as_amdgpu_drm_h(monkeypatch) -> None:
    """CI has no AMD driver: the request and the reply offsets, against include/uapi/drm/amdgpu_drm.h."""

    def ioctl(_fd: int, request: int, arg: bytes) -> bytes:
        assert request == 0x40206445  # DRM_IOW(DRM_COMMAND_BASE 0x40 + DRM_AMDGPU_INFO 0x05), 32-byte struct
        assert len(arg) == 32  # struct drm_amdgpu_info: return_pointer, return_size, query, 16-byte union
        pointer, size, query = struct.unpack_from("<QII", arg)
        assert query == 0x16  # AMDGPU_INFO_DEV_INFO
        reply = (ctypes.c_char * size).from_address(pointer)
        struct.pack_into("<I", reply, 0, 0x1586)  # drm_amdgpu_info_device.device_id
        struct.pack_into("<Q", reply, 136, 0x1)  # .ids_flags: AMDGPU_IDS_FLAGS_FUSION
        return arg

    monkeypatch.setattr(fcntl, "ioctl", ioctl)
    assert hw_mod._amdgpu_dev_info(-1) == (0x1586, 0x1)


_REAL_KFD = Path("/sys/class/kfd/kfd/topology/nodes")


@pytest.mark.skipif(not Path("/dev/kfd").exists(), reason="needs an AMD GPU under KFD")
def test_the_driver_query_answers_for_the_device_kfd_names() -> None:
    """On real hardware: the ioctl's layout holds, device id for device id (the M5: 0x1586, integrated)."""
    checked = 0
    for node in sorted(p for p in _REAL_KFD.iterdir() if p.name.isdigit()):
        props = hw_mod._kfd_properties(node)
        if props.get("simd_count", 0) <= 0:
            continue
        try:
            fd = os.open(f"/dev/dri/renderD{props['drm_render_minor']}", os.O_RDWR | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            info = hw_mod._amdgpu_dev_info(fd)
        finally:
            os.close(fd)
        assert info is not None
        assert info[0] == props["device_id"]
        checked += 1
    if not checked:
        pytest.skip("no render node this process may open")
