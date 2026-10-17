"""Hardware inspection, torch-free: real measurement over NVML/nvidia-smi/sysconf.

service_a's probing wins here over service_c's ``inspect_hardware`` (commit body): service_c's
version resolves a real device name/memory only through an optional ``import torch`` call, which
this package cannot carry (it must stay torch-free). service_a's NVML → nvidia-smi → sysconf
chain measures the host directly and never imports torch, so it is the more capable version for a
torch-free package. service_c's ``device_override``/``resolve_device`` ergonomics are kept, laid
over the same torch-free probing.
"""

from __future__ import annotations

import os
import platform
import shutil
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


def _nvml_read[T](index: int, read: Callable[[Any, Any], T]) -> T:
    """``read(pynvml, handle)`` for NVML device ``index``, between NVML's init and shutdown.

    Raises what NVML raises (ImportError without pynvml).
    """
    import pynvml  # noqa: PLC0415

    pynvml.nvmlInit()
    try:
        return read(pynvml, pynvml.nvmlDeviceGetHandleByIndex(index))
    finally:
        pynvml.nvmlShutdown()


def _smi_query(fields: str, index: int = 0) -> list[str] | None:
    """The comma-separated ``fields`` nvidia-smi reports for GPU ``index``; None when it fails."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = (
            subprocess.run(  # noqa: S603 — smi is a shutil.which() result, not a literal
                [smi, "-i", str(index), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
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


def _cuda_memory_gb() -> tuple[str, float | None] | None:
    try:
        name, total = _nvml_read(
            0, lambda nv, h: (nv.nvmlDeviceGetName(h), nv.nvmlDeviceGetMemoryInfo(h).total)
        )
        if total and total > 0:
            return (name if isinstance(name, str) else name.decode(), total / _GIB)
    except Exception:  # noqa: BLE001,S110 — probe failure is not an error, it means "no CUDA"
        pass
    parts = _smi_query("name,memory.total")
    if parts is None or len(parts) != 2:  # noqa: PLR2004 — the two fields asked for
        return None
    name, mib = parts
    if mib.isdigit() and int(mib) > 0:
        return (name, int(mib) / 1024)
    if mib == "[N/A]":  # unified-memory GPUs report [N/A]; caller uses system RAM
        return (name, None)
    return None


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
    """Probe the host (NVML → nvidia-smi → sysctl) and report measured hardware; never raises.

    ``device_override`` forces the device family (``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``)
    without skipping measurement: memory is still probed for the forced device where possible
    (real CUDA memory for a ``"cuda"`` override, system RAM otherwise), and every unmeasured value
    is still declared in ``assumptions``.
    """
    assumptions: list[str] = []
    system, arch = platform.system().lower(), platform.machine().lower()

    if device_override is not None:
        device = _parse_device(device_override)
        name = f"override ({device_override})"
        if device is DeviceType.CUDA:
            cuda = _cuda_memory_gb()
            if cuda is not None and cuda[1] is not None:
                name, mem = cuda[0], cuda[1]
            else:
                mem = _system_memory_or_fallback(assumptions)
                assumptions.append(
                    f"device_override={device_override!r} requested CUDA but no CUDA memory "
                    "was probed; using system RAM (ASSUMPTION)."
                )
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


def _cuda_free_bytes(index: int) -> tuple[int | None, bool] | None:
    """(free bytes, unified) of CUDA device ``index`` by NVML, then nvidia-smi; None when neither reads it.

    Unified (the device's memory is the host's RAM, free bytes None) when NVML reports the memory as not
    supported or nvidia-smi as ``[N/A]``: the GB10 of a DGX Spark does both.
    """

    def free(nv: Any, handle: Any) -> tuple[int | None, bool]:
        try:
            return nv.nvmlDeviceGetMemoryInfo(handle).free, False
        except Exception as exc:
            if isinstance(exc, getattr(nv, "NVMLError_NotSupported", ())):
                return None, True
            raise

    try:
        return _nvml_read(index, free)
    except Exception:  # noqa: BLE001,S110 — probe failure is not an error, it means "try nvidia-smi"
        pass
    parts = _smi_query("memory.free", index)
    if parts is None:
        return None
    if parts[0].isdigit():
        return int(parts[0]) * 1024**2, False
    if parts[0] == "[N/A]":
        return None, True
    return None


def available_memory(device: str = "cuda", meminfo_path: str = _MEMINFO) -> AvailableMemory | None:
    """How much a new allocation on ``device`` (``"cuda"``, ``"cuda:1"``, ``"cpu"``, ``"mps"``) can get now.

    None when it cannot be measured, so a caller can tell "unknown" from a number. A GPU with its own
    memory: that memory's free bytes. A GPU that shares the host's RAM (such as a DGX Spark's GB10):
    ``host_available_bytes``, because the driver's own free count (``cudaMemGetInfo``,
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
        free, unified = _cuda_free_bytes(int(index) if index else 0) or (None, False)
        if not unified:
            if free is None:
                return None
            return AvailableMemory(bytes=free, device=kind, unified=False, source="the GPU's own free memory")
        source += ", as this GPU shares the host's RAM and the driver's free count leaves out the page cache"
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
