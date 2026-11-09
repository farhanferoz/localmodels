"""Hardware inspection, torch-free: real measurement over NVML/nvidia-smi, ROCm's sysfs (AMD) and sysconf.

service_a's probing wins here over service_c's ``inspect_hardware`` (commit body): service_c's
version resolves a real device name/memory only through an optional ``import torch`` call, which
this package cannot carry (it must stay torch-free). service_a's NVML → nvidia-smi → sysconf
chain measures the host directly and never imports torch, so it is the more capable version for a
torch-free package. service_c's ``device_override``/``resolve_device`` ergonomics are kept, laid
over the same torch-free probing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .enums import DeviceTier, DeviceType
from .params import classify_tier

if TYPE_CHECKING:
    from collections.abc import Callable

_GIB = 1024**3
_FALLBACK_MEMORY_GB = 8.0  # declared when system RAM cannot be measured


class HardwareReport(BaseModel):
    """What inspect_hardware measured about the host, plus declared assumptions."""

    platform: str
    arch: str
    device: DeviceType
    device_name: str
    total_memory_gb: float
    tier: DeviceTier
    assumptions: list[str] = Field(default_factory=list)


def _parse_device(override: str) -> DeviceType:
    try:
        return DeviceType(override.split(":", 1)[0].lower())
    except ValueError as exc:
        known = [d.value for d in DeviceType]
        msg = f"unknown device override {override!r}; use one of {known}"
        raise ValueError(msg) from exc


_UUID_PREFIXES = ("GPU-", "MIG-")
_CUDA_INDEX = re.compile(r"\+?\d+")


def _cuda_gpu(ordinal: int) -> str | None:
    """How nvidia-smi and NVML name this process's CUDA device ``ordinal``; None when it is hidden.

    NVML and nvidia-smi number every GPU on the host whatever the variable says, so without this a process
    whose GPU was hidden on purpose (``CUDA_VISIBLE_DEVICES=""``) was told it had one, then failed in torch.
    CUDA reads the variable as a comma-separated list up to the first entry it cannot use, each entry an index
    read like ``strtoul`` (``"0abc"`` is 0) or a UUID, and a repeated entry hides every GPU (CUDA driver
    580.173.02 on a one-GPU host, 2026-10-22: no device for ``""``, ``" "``, ``"-1"``, ``"1"``,
    ``"NoDevFiles"``, ``"-1,0"`` and ``"0,0"``; GPU 0 for ``"0"``, ``"0abc"``, ``"+0"`` and ``"0,-1"``). An
    index past the host's last GPU and a shortened UUID (which CUDA accepts, nvidia-smi does not) are left for
    the probe to fail on.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(ordinal)
    gpus: list[str] = []
    for raw in visible.split(","):
        entry = raw.strip()
        if not entry.startswith(_UUID_PREFIXES):
            index = _CUDA_INDEX.match(entry)
            if index is None:
                break
            entry = str(int(index.group()))
        if entry in gpus:
            return None
        gpus.append(entry)
    return gpus[ordinal] if ordinal < len(gpus) else None


def _nvml_read[T](gpu: str, read: Callable[[Any, Any], T]) -> T:
    """``read(pynvml, handle)`` for NVML device ``gpu`` (an index or a UUID), between init and shutdown.

    Raises what NVML raises (ImportError without pynvml).
    """
    import pynvml  # noqa: PLC0415

    pynvml.nvmlInit()
    try:
        handle = (
            pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            if gpu.isdigit()
            else pynvml.nvmlDeviceGetHandleByUUID(gpu)
        )
        return read(pynvml, handle)
    finally:
        pynvml.nvmlShutdown()


def _smi_query(fields: str, gpu: str) -> list[str] | None:
    """The comma-separated ``fields`` nvidia-smi reports for ``gpu`` (an index or a UUID); None on failure."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = (
            subprocess.run(  # noqa: S603 — smi is a shutil.which() result, not a literal
                [smi, "-i", gpu, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (subprocess.SubprocessError, IndexError):
        return None
    return [part.strip() for part in out.split(",")]


def gpu_process_memory_mib() -> dict[int, int] | None:
    """The GPU memory each process holds now, in MiB by process id (``nvidia-smi --query-compute-apps``).

    A process on several GPUs counts once, the sum of its rows; a row without a number (``[N/A]``) is left
    out. None when nvidia-smi is missing or fails, an empty dict when no process uses the GPU. Per-process, so
    other jobs on the card do not move the figure, which ``MemAvailable`` cannot say.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 — smi is a shutil.which() result, not a literal
            [smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    held: dict[int, int] = {}
    for line in out.splitlines():
        pid, _, used = (part.strip() for part in line.partition(","))
        if pid.isdigit() and used.isdigit():
            held[int(pid)] = held.get(int(pid), 0) + int(used)
    return held


def process_anon_mib(pid: int) -> int | None:
    """A process's anonymous resident memory in MiB (``RssAnon`` in ``/proc/<pid>/status``), None unreadable.

    On a GPU that shares the host's RAM (a GB10) a server holds this beside what nvidia-smi counts: a
    llama-server of an 87 GiB model held 28.5 GiB here right after loading, 61 GiB on the GPU, and grew here
    while it served (2026-10-23).
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^RssAnon:\s+(\d+) kB$", status, re.MULTILINE)
    return None if match is None else int(match[1]) // 1024


def _cuda_memory_gb() -> tuple[str, float | None] | None:
    gpu = _cuda_gpu(0)
    if gpu is None:
        return None
    try:
        name, total = _nvml_read(
            gpu, lambda nv, h: (nv.nvmlDeviceGetName(h), nv.nvmlDeviceGetMemoryInfo(h).total)
        )
        if total and total > 0:
            return (name if isinstance(name, str) else name.decode(), total / _GIB)
    except Exception:  # noqa: BLE001,S110 — probe failure is not an error, it means "no CUDA"
        pass
    parts = _smi_query("name,memory.total", gpu)
    if parts is None or len(parts) != 2:  # noqa: PLR2004 — the two fields asked for
        return None
    name, mib = parts
    if mib.isdigit() and int(mib) > 0:
        return (name, int(mib) / 1024)
    if mib == "[N/A]":  # unified-memory GPUs report [N/A]; caller uses system RAM
        return (name, None)
    return None


# --- AMD GPUs under ROCm, found the way ROCm's own runtime finds them (torch-free) -----------------------
# ROCm's PyTorch addresses an AMD GPU as "cuda" (torch.cuda.is_available() is True), so one is reported
# as DeviceType.CUDA. Only on a host without NVIDIA's stack (`_nvidia_on_host`): there "cuda" means NVIDIA,
# and the two vendors never answer for each other. Each rule below follows ROCm's source (ROCm/rocm-systems,
# develop, read 2026-11-09): the thunk (projects/rocr-runtime/libhsakmt/src/topology.c), ROCr
# (projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_topology.cpp, amd_filter_device.cpp,
# amd_gpu_agent.cpp) and HIP (projects/clr/rocclr/device/rocm/rocdevice.cpp). Measured on a Bosgame M5
# (Ryzen AI Max+ 395, Radeon 8060S, gfx1151, kernel 7.0.0-43).
_AMD_PCI_VENDOR = 0x1002
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
_KFD_DEVICE = Path("/dev/kfd")
_DRI = Path("/dev/dri")
_DRM = Path("/sys/class/drm")
_ROCM_SYSTEM_ROOT = Path("/opt/rocm")
# Integrated (an APU, whose memory is the host's RAM) or discrete: the thunk sets `Integrated` from
# AMDGPU_IDS_FLAGS_FUSION in the amdgpu driver's device info (topology.c, amdgpu_query_gpu_info), asked of the
# render node with DRM_IOCTL_AMDGPU_INFO = _IOW('d', 0x45, struct drm_amdgpu_info) and AMDGPU_INFO_DEV_INFO;
# ids_flags is at byte 136 of struct drm_amdgpu_info_device (include/uapi/drm/amdgpu_drm.h). KFD's
# local_mem_size cannot tell: the kernel prints it as 0 for every GPU (kfd_topology.c).
_DRM_IOCTL_AMDGPU_INFO = 0x40206445
_AMDGPU_INFO_DEV_INFO = 0x16
_DEV_INFO_IDS_FLAGS_AT = 136
_AMDGPU_IDS_FLAGS_FUSION = 0x1
_DEV_INFO_BYTES = 1024  # room for the whole struct: the kernel writes the smaller of this and its own size
_HIP_INDEX = re.compile(r"0|[1-9][0-9]*")  # HIP keeps an entry only when it is std::to_string of its value
_STRTOL_INDEX = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)")  # strtol(s, &end, 0), all of s
_ROCR_UUID_CHARS = range(5, 21)  # ROCr reads a GPU-... entry of 5 to 20 characters


def _nvidia_on_host() -> bool:
    """Whether NVIDIA's stack is here (nvidia-smi on PATH, or NVML starts): "cuda" then means NVIDIA."""
    if shutil.which("nvidia-smi") is not None:
        return True
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 — no NVML, or no driver behind it, means no NVIDIA stack
        return False
    return True


class _AmdGpu(BaseModel):
    """One AMD GPU as ROCr lists it, with what the driver says about its memory."""

    gfx: str  # e.g. "gfx1151"
    pci_id: str  # vendor:device, e.g. "1002:1586"
    render_minor: int
    unique_id: int  # KFD's unique_id; 0 when the GPU has none (the M5's gfx1151 has none)
    unified: bool  # the driver's FUSION flag: an APU, whose memory is the host's RAM

    @property
    def name(self) -> str:
        return f"AMD GPU {self.gfx} ({self.pci_id}, ROCm)"

    def _counter(self, name: str) -> int | None:
        try:
            return int((_DRM / f"renderD{self.render_minor}" / "device" / name).read_text(encoding="ascii"))
        except (OSError, ValueError):
            return None

    def total_bytes(self) -> int | None:
        """What the GPU can use: its VRAM; for an APU the GTT limit, the most it may map from the host's RAM.

        The GTT limit is what ROCm's torch reports as an APU's total memory (on the M5 with a 1 GiB BIOS
        carve-out and ``amd-ttm --set 109``: 109.0 GiB, matching ``torch.cuda.get_device_properties(0)``).
        """
        vram = self._counter("mem_info_vram_total") or 0
        total = max(vram, self._counter("mem_info_gtt_total") or 0) if self.unified else vram
        return total or None

    def free_bytes(self) -> tuple[int | None, bool] | None:
        """(free bytes, unified), as ``_cuda_free_bytes`` reports an NVIDIA GPU; an APU's is the host's."""
        if self.unified:
            return None, True
        total, used = self._counter("mem_info_vram_total"), self._counter("mem_info_vram_used")
        if total is None or used is None:
            return None
        return max(total - used, 0), False

    def host_cap(self) -> int | None:
        """Bytes an APU may still map (GTT limit less GTT in use), a cap on the host's free RAM; else None."""
        if not self.unified:
            return None
        total, used = self._counter("mem_info_gtt_total"), self._counter("mem_info_gtt_used")
        return None if total is None or used is None else max(total - used, 0)


def _kfd_properties(node: Path) -> dict[str, int]:
    props: dict[str, int] = {}
    try:
        for line in (node / "properties").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(" ")
            if value.strip().lstrip("-").isdigit():
                props[key] = int(value)
    except OSError:
        pass
    return props


def _gfx_name(target_version: int) -> str:
    """KFD's ``gfx_target_version`` as LLVM names the target: 110501 is gfx1151, 90402 is gfx942."""
    major, minor, step = target_version // 10000, (target_version // 100) % 100, target_version % 100
    return f"gfx{major}{minor}{step:x}"


def _amdgpu_dev_info(fd: int) -> tuple[int, int] | None:
    """(device_id, ids_flags) the amdgpu driver gives for open render node ``fd``; None without an answer."""
    import fcntl  # noqa: PLC0415 — POSIX only, and only reached where KFD exists

    reply = ctypes.create_string_buffer(_DEV_INFO_BYTES)
    request = struct.pack("<QII16x", ctypes.addressof(reply), _DEV_INFO_BYTES, _AMDGPU_INFO_DEV_INFO)
    try:
        fcntl.ioctl(fd, _DRM_IOCTL_AMDGPU_INFO, request)
    except OSError:
        return None
    (device_id,) = struct.unpack_from("<I", reply.raw, 0)
    (flags,) = struct.unpack_from("<Q", reply.raw, _DEV_INFO_IDS_FLAGS_AT)
    return device_id, flags


def _render_node_fusion(minor: int, device_id: int) -> bool | None:
    """Open render node ``minor`` read-write, as ROCm does, and ask the driver whether the GPU is integrated.

    None when this process cannot open it (no such node, or no permission: outside the render and video
    groups, a container started without /dev/dri), which is when the thunk leaves the GPU out. False when the
    driver does not answer, or answers for another device than KFD's ``device_id``: a discrete card's figures
    are its own VRAM, so an unknown never overstates.
    """
    try:
        fd = os.open(_DRI / f"renderD{minor}", os.O_RDWR | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        info = _amdgpu_dev_info(fd)
    finally:
        os.close(fd)
    return info is not None and info[0] == device_id and bool(info[1] & _AMDGPU_IDS_FLAGS_FUSION)


def _rocm_runtime() -> bool:
    """Whether this environment has a ROCm runtime to run an AMD GPU with.

    KFD lists every GPU the amdgpu driver runs, with or without ROCm installed, so without this an AMD laptop
    with ordinary PyTorch would be told "cuda". Found: TheRock's wheels (``rocm-sdk-core``, which carries
    libamdhip64), a ROCm build of torch (its version ends ``+rocmX``; PyTorch's ROCm wheels bundle HIP), or a
    system HIP runtime (``$ROCM_PATH``, /opt/rocm, or on the loader's path).
    """
    for dist, mark in (("rocm-sdk-core", ""), ("torch", "+rocm")):
        try:
            if mark in importlib.metadata.version(dist):
                return True
        except importlib.metadata.PackageNotFoundError:
            pass
    roots = [Path(os.environ["ROCM_PATH"])] if os.environ.get("ROCM_PATH") else []
    if any(next((root / "lib").glob("libamdhip64.so*"), None) for root in [*roots, _ROCM_SYSTEM_ROOT]):
        return True
    return ctypes.util.find_library("amdhip64") is not None


def _amd_gpus() -> list[_AmdGpu]:
    """The AMD GPUs ROCr enumerates for this process, in its order, before any visibility variable.

    Empty without KFD, without read-write access to /dev/kfd, or without a ROCm runtime (``_rocm_runtime``).
    A GPU whose render node this process cannot open is left out, as the thunk leaves it out
    (``topology_sysfs_check_node_supported``). Not checked: whether ROCr knows the GPU's ISA (it skips one it
    does not) or whether the installed runtime has kernels for it; such a GPU is reported as an NVIDIA GPU
    is whose compute capability the installed torch lacks.
    """
    try:
        nodes = sorted((p for p in _KFD_NODES.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError:
        return []
    found = [props for node in nodes if (props := _kfd_properties(node)).get("simd_count", 0) > 0]
    found = [props for props in found if props.get("vendor_id") == _AMD_PCI_VENDOR]
    if not found or not os.access(_KFD_DEVICE, os.R_OK | os.W_OK) or not _rocm_runtime():
        return []
    gpus = []
    for props in found:
        minor, device_id = props.get("drm_render_minor", 0), props.get("device_id", 0)
        unified = _render_node_fusion(minor, device_id) if minor > 0 else None
        if unified is not None:
            gpus.append(
                _AmdGpu(
                    gfx=_gfx_name(props.get("gfx_target_version", 0)),
                    pci_id=f"{_AMD_PCI_VENDOR:04x}:{device_id:04x}",
                    render_minor=minor,
                    unique_id=props.get("unique_id", 0),
                    unified=unified,
                )
            )
    return gpus


def _strtol_index(token: str) -> int | None:
    """``strtol(token, &end, 0)`` when it reads all of ``token`` (ROCr's test), else None.

    "0x1" is 1, "010" is 8, "0abc" and "1 " are None.
    """
    if not _STRTOL_INDEX.fullmatch(token):
        return None
    digits = token.lstrip("+-")
    base = 16 if digits[:2].lower() == "0x" else 8 if digits.startswith("0") else 10
    return -int(digits, base) if token.startswith("-") else int(digits, base)


def _rocr_visible(gpus: list[_AmdGpu]) -> list[_AmdGpu]:
    """ROCR_VISIBLE_DEVICES as ROCr applies it (``RvdFilter``), to the GPUs it enumerates.

    Unset: every GPU. Set, even empty: only those it names, in its order, so "" hides them all. Entries are
    trimmed and upper-cased; one is an index read by ``strtol`` in base 0 that must use the whole entry, or a
    ``GPU-`` prefix (5 to 20 characters) of exactly one GPU's ``GPU-%016X`` UUID. The first entry that is
    invalid, out of range or repeated ends the list; entries past the number of GPUs are not read.
    """
    value = os.environ.get("ROCR_VISIBLE_DEVICES")
    if value is None:
        return gpus
    uuids = [f"GPU-{gpu.unique_id:016X}" if gpu.unique_id else "Invalid-UUID" for gpu in gpus]
    entries = value.split(",")[:-1] if value.endswith(",") else value.split(",")  # getline drops a last ""
    picked: list[int] = []
    for raw in entries[: len(gpus)] if value else []:
        entry = raw.strip().upper()
        if entry.startswith("G"):
            fits = len(entry) in _ROCR_UUID_CHARS
            named = [i for i, uuid in enumerate(uuids) if fits and uuid.startswith(entry)]
            index = named[0] if len(named) == 1 else None
        else:
            index = _strtol_index(entry)
        if index is None or not 0 <= index < len(gpus) or index in picked:
            break
        picked.append(index)
    return [gpus[i] for i in picked]


def _hip_flag(name: str) -> str:
    """Environment variable ``name`` as HIP's flag loader reads it (``rocclr/utils/flags.cpp``).

    Unset is ""; set but empty is " " (so it counts as set); a value in double quotes loses them.
    """
    value = os.environ.get(name)
    if value is None:
        return ""
    if not value:
        return " "
    quoted = value.lstrip(" ").rstrip(" ")
    return quoted[1:-1] if len(quoted) > 1 and quoted.startswith('"') and quoted.endswith('"') else value


def _hip_visible(gpus: list[_AmdGpu]) -> list[_AmdGpu]:
    """HIP_VISIBLE_DEVICES, or CUDA_VISIBLE_DEVICES when that is unset, applied to ROCr's list as HIP does.

    HIP never chains the two (``rocdevice.cpp``). Only unset filters nothing: a variable set to "" reads as
    " " (``_hip_flag``), an invalid entry that hides every GPU. An entry is kept only when it is exactly the
    decimal index of a GPU, or names a GPU by a piece of its ``GPU-%016x`` UUID (``GPU-XX`` when it has none);
    the first entry that is not ends the list, and a repeated one is skipped. Measured with HIP 7.16 (TheRock
    10.1, gfx1151, one GPU): "" and " " hide it; "0", "0,0", "0,1", "GPU-" and '"0"' keep it; "00", "+0",
    "0abc" and "1,0" hide it; HIP_VISIBLE_DEVICES=0 keeps it whatever CUDA_VISIBLE_DEVICES says.
    """
    ordinals = _hip_flag("HIP_VISIBLE_DEVICES") or _hip_flag("CUDA_VISIBLE_DEVICES")
    if not ordinals:
        return gpus
    uuids = [f"GPU-{gpu.unique_id:016x}" if gpu.unique_id else "GPU-XX" for gpu in gpus]
    picked: list[int] = []
    for raw in ordinals.split(","):
        entry = raw
        if "GPU-" in entry:
            entry = next((str(i) for i, uuid in enumerate(uuids) if entry in uuid), entry)
        if not _HIP_INDEX.fullmatch(entry) or int(entry) >= len(gpus):
            break
        if int(entry) not in picked:
            picked.append(int(entry))
    return [gpus[i] for i in picked]


def _amd_gpu(ordinal: int) -> _AmdGpu | None:
    """AMD GPU ``ordinal`` in HIP's numbering for this process.

    None when there is none, it is hidden, or the host has NVIDIA's stack.
    """
    if _nvidia_on_host():
        return None
    visible = _hip_visible(_rocr_visible(_amd_gpus()))
    return visible[ordinal] if 0 <= ordinal < len(visible) else None


def is_rocm_device(device_name: str) -> bool:
    """Whether ``inspect_hardware`` named an AMD GPU under ROCm (DeviceType.CUDA either way)."""
    return device_name.startswith("AMD GPU ") and device_name.endswith(", ROCm)")


def _system_memory_gb() -> float | None:
    if platform.system() == "Darwin":
        try:
            raw = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            return int(raw.strip()) / _GIB
        except (subprocess.SubprocessError, ValueError):
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GIB
    except (AttributeError, ValueError, OSError):
        return None


def _system_memory_or_fallback(assumptions: list[str]) -> float:
    """Measured system RAM, or the declared fallback when the platform will not report it.

    Testing the measurement for None rather than comparing the result with the fallback matters:
    a machine with exactly 8 GB of RAM is measured, not assumed.
    """
    measured = _system_memory_gb()
    if measured:
        return measured
    assumptions.append("system RAM could not be measured; falling back to 8.0 GB (ASSUMPTION).")
    return _FALLBACK_MEMORY_GB


def inspect_hardware(device_override: str | None = None) -> HardwareReport:
    """Probe the host (NVML → nvidia-smi → ROCm's sysfs → sysctl) and report measured hardware; never raises.

    ``device_override`` forces the device family (``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``)
    without skipping measurement: memory is still probed for the forced device where possible
    (real CUDA memory for a ``"cuda"`` override, system RAM otherwise), and every unmeasured value
    is still declared in ``assumptions``. An AMD GPU under ROCm is a CUDA device, measured only on a
    host without NVIDIA's stack (``_amd_gpu``).
    """
    assumptions: list[str] = []
    system, arch = platform.system().lower(), platform.machine().lower()
    if _cuda_gpu(0) is None and _amd_gpu(0) is None:
        assumptions.append(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} leaves this process no CUDA GPU."
        )

    if device_override is not None:
        device = _parse_device(device_override)
        name = f"override ({device_override})"
        if device is DeviceType.CUDA:
            name, mem = _cuda_override_memory(device_override, name, assumptions)
        else:
            mem = _system_memory_or_fallback(assumptions)
    else:
        device, name, mem = DeviceType.CPU, "CPU", 0.0
        cuda = _cuda_memory_gb()
        if cuda is not None:
            device, name = DeviceType.CUDA, cuda[0]
            reported = cuda[1]
            if reported is None:
                assumptions.append(
                    "nvidia-smi reports [N/A] memory.total (unified memory); using system RAM as GPU memory."
                )
                mem = _system_memory_or_fallback(assumptions)
            else:
                mem = reported
        elif (amd := _amd_memory_gb(0)) is not None:
            device, (name, mem) = DeviceType.CUDA, amd
        else:
            mem = _system_memory_or_fallback(assumptions)
            if system == "darwin" and arch == "arm64":
                device, name = DeviceType.MPS, "Apple Silicon"

    return HardwareReport(
        platform=system,
        arch=arch,
        device=device,
        device_name=name,
        total_memory_gb=round(mem, 1),
        tier=classify_tier(device, mem).tier,
        assumptions=assumptions,
    )


def _cuda_override_memory(device_override: str, name: str, assumptions: list[str]) -> tuple[str, float]:
    """(name, GiB) for a ``"cuda"`` override: NVIDIA's measured memory, else an AMD GPU's, else system RAM."""
    cuda = _cuda_memory_gb()
    if cuda is not None and cuda[1] is not None:
        return cuda[0], cuda[1]
    amd = _amd_memory_gb(0) if cuda is None else None
    if amd is not None:
        return amd
    mem = _system_memory_or_fallback(assumptions)
    assumptions.append(
        f"device_override={device_override!r} requested CUDA but no CUDA memory "
        "was probed; using system RAM (ASSUMPTION)."
    )
    return name, mem


def _amd_memory_gb(ordinal: int) -> tuple[str, float] | None:
    """(name, GiB it can use) of AMD GPU ``ordinal`` (``_amd_gpu``); None when absent or unmeasured."""
    gpu = _amd_gpu(ordinal)
    total = None if gpu is None else gpu.total_bytes()
    return None if gpu is None or total is None else (gpu.name, total / _GIB)


_MEMINFO = "/proc/meminfo"
_MEMINFO_FIELDS = frozenset({"MemAvailable", "HugePages_Total", "HugePages_Free", "Hugepagesize"})


class AvailableMemory(BaseModel):
    """What a new allocation on a device can get now (``available_memory``)."""

    bytes: int
    device: DeviceType
    unified: bool  # the device's memory is the host's RAM
    source: str  # where the figure comes from, in words, for a log line


def host_available_bytes(meminfo_path: str = _MEMINFO) -> int | None:
    """What the host can allocate without swapping; None where /proc/meminfo cannot be read (not Linux).

    NVIDIA's rule for unified memory (DGX Spark User Guide, Known Issues, "Guidance for reporting memory
    resources with unified memory architecture"): ``MemAvailable``, which is free memory plus the page cache
    and other memory the kernel gives back on demand, or ``HugePages_Free`` x ``Hugepagesize`` when huge pages
    are configured (``HugePages_Total`` above 0).
    """
    fields: dict[str, int] = {}
    try:
        with Path(meminfo_path).open(encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in _MEMINFO_FIELDS:
                    fields[key] = int(rest.split()[0])  # kB, except the two HugePages_ counts
    except (OSError, ValueError, IndexError):
        return None
    if fields.get("HugePages_Total", 0) > 0 and {"HugePages_Free", "Hugepagesize"} <= fields.keys():
        return fields["HugePages_Free"] * fields["Hugepagesize"] * 1024
    return fields["MemAvailable"] * 1024 if "MemAvailable" in fields else None


def _cuda_free_bytes(ordinal: int) -> tuple[int | None, bool] | None:
    """(free bytes, unified) of CUDA device ``ordinal`` by NVML, then nvidia-smi; None when neither reads it.

    Unified (the device's memory is the host's RAM, free bytes None) when NVML reports the memory as not
    supported or nvidia-smi as ``[N/A]``: the GB10 of a DGX Spark does both. None too when
    CUDA_VISIBLE_DEVICES hides the device (``_cuda_gpu``).
    """
    gpu = _cuda_gpu(ordinal)
    if gpu is None:
        return None

    def free(nv: Any, handle: Any) -> tuple[int | None, bool]:
        try:
            return nv.nvmlDeviceGetMemoryInfo(handle).free, False
        except Exception as exc:
            if isinstance(exc, getattr(nv, "NVMLError_NotSupported", ())):
                return None, True
            raise

    try:
        return _nvml_read(gpu, free)
    except Exception:  # noqa: BLE001,S110 — probe failure is not an error, it means "try nvidia-smi"
        pass
    parts = _smi_query("memory.free", gpu)
    if parts is None:
        return None
    if parts[0].isdigit():
        return int(parts[0]) * 1024**2, False
    if parts[0] == "[N/A]":
        return None, True
    return None


def available_memory(device: str = "cuda", meminfo_path: str = _MEMINFO) -> AvailableMemory | None:
    """How much a new allocation on ``device`` (``"cuda"``, ``"cuda:1"``, ``"cpu"``, ``"mps"``) can get now.

    None when it cannot be measured or CUDA_VISIBLE_DEVICES hides the GPU, so a caller can tell "unknown" from
    a number. A GPU with its own memory: that memory's free bytes. A GPU that shares the host's RAM (such as a
    DGX Spark's GB10): ``host_available_bytes``, because the driver's own free count (``cudaMemGetInfo``,
    ``torch.cuda.mem_get_info``, nvidia-smi) leaves out the page cache the kernel gives back on demand, so
    after a large file read it reports far less than an allocation can get (NVIDIA, DGX Spark porting guide:
    "the cudaMemGetInfo API does not account for memory that could potentially be reclaimed"; measured on a
    GB10 on 2026-10-17: 55.9 GB by the driver, 108.1 GB available). Decide whether something fits from this,
    never from the driver's count. The CPU: ``host_available_bytes``. Apple silicon (MPS): None, macOS has
    no /proc/meminfo.
    """
    kind = _parse_device(device)
    host = None if kind is DeviceType.MPS else host_available_bytes(meminfo_path)
    unified, source = False, "the host's available memory (MemAvailable)"
    if kind is DeviceType.CUDA:
        _, _, index = device.partition(":")
        ordinal = int(index) if index else 0
        amd = _amd_gpu(ordinal)
        free, unified = (amd.free_bytes() if amd is not None else _cuda_free_bytes(ordinal)) or (None, False)
        if not unified:
            if free is None:
                return None
            return AvailableMemory(bytes=free, device=kind, unified=False, source="the GPU's own free memory")
        source += ", as this GPU shares the host's RAM and the driver's free count leaves out the page cache"
        cap = None if amd is None else amd.host_cap()
        if host is not None and cap is not None and cap < host:
            host = cap
            source = "the GPU's GTT limit less what it holds (an AMD APU cannot map more of the host's RAM)"
    if host is None:
        return None
    return AvailableMemory(bytes=host, device=kind, unified=unified, source=source)


def resolve_device(device_override: str | None = None) -> str:
    """Resolve a compute device string (``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``).

    With an override, validates it and preserves any index suffix (e.g. ``"cuda:0"``) verbatim.
    Otherwise inspects hardware for real.
    """
    if device_override:
        _parse_device(device_override)
        return device_override.lower()
    return inspect_hardware().device.value


_MOUNTS = "/proc/mounts"
_MOUNT_FIELDS = 3  # device, mount point, filesystem type (then options)
_NETWORK_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "fuse.sshfs", "ceph", "glusterfs"}
)


def network_filesystem(path: Path, mounts_path: str = _MOUNTS) -> str | None:
    """The filesystem type when ``path`` is on a network filesystem (NFS, SMB, ...); None otherwise.

    A model file read over the network loads at network speed on every cold start (measured on gpu-node-1 on
    2026-10-17: about 0.05 GB/s from its NFS share, 1.1 GB/s from local disk), so callers warn about it.
    """
    try:
        lines = Path(mounts_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    target = Path(path).resolve()
    best: Path | None = None
    kind = None
    for line in lines:
        fields = line.split()
        if len(fields) < _MOUNT_FIELDS:
            continue
        mount = Path(fields[1].replace("\\040", " "))
        if target.is_relative_to(mount) and (best is None or len(mount.parts) > len(best.parts)):
            best, kind = mount, fields[2]
    return kind if kind in _NETWORK_FILESYSTEMS else None
