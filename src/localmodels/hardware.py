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

from pydantic import BaseModel, Field

from .enums import DeviceTier, DeviceType
from .params import classify_tier

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


def _cuda_memory_gb() -> tuple[str, float | None] | None:
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            if total and total > 0:
                return (name if isinstance(name, str) else name.decode(), total / _GIB)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001,S110 — probe failure is not an error, it means "no CUDA"
        pass
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = (
            subprocess.run(  # noqa: S603 — smi is a shutil.which() result, not a literal
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, mib = (part.strip() for part in out.split(","))
        if mib.isdigit() and int(mib) > 0:
            return (name, int(mib) / 1024)
        if mib == "[N/A]":  # unified-memory GPUs report [N/A]; caller uses system RAM
            return (name, None)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None
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


def resolve_device(device_override: str | None = None) -> str:
    """Resolve a compute device string (``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``).

    With an override, validates it and preserves any index suffix (e.g. ``"cuda:0"``) verbatim.
    Otherwise inspects hardware for real.
    """
    if device_override:
        _parse_device(device_override)
        return device_override.lower()
    return inspect_hardware().device.value
