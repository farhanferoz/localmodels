r"""llama-server compute-backend detection, torch-free.

Distinguishes "no accelerator library is linked into this build" (``Backend.CPU``) from
"the probe itself could not be read" (``Backend.UNKNOWN``), so a caller never mistakes a
detection failure for a genuine CPU-only build and refuses one it was never built to refuse.

``--list-devices`` is tried first: it is the authoritative runtime answer (it lists the
devices the binary would actually serve on), verified against this DGX's real build
(2026-09-25): ``Available devices:\n  CUDA0: NVIDIA GB10 (124610 MiB, 86812 MiB free)``,
exit 0, no model load, no GPU memory touched. ``ldd`` (link-time evidence, ``libggml-cuda.so.0``
on this build) is the fallback, used only when ``--list-devices`` cannot be run at all.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_LIST_DEVICES_TIMEOUT_S = 10.0
_LDD_TIMEOUT_S = 10.0


class Backend(StrEnum):
    """Compute backend actually linked into a llama-server build."""

    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    CPU = "cpu"
    UNKNOWN = "unknown"


_LIST_DEVICES_PATTERNS: tuple[tuple[re.Pattern[str], Backend], ...] = (
    (re.compile(r"\bCUDA\d*\s*:", re.IGNORECASE), Backend.CUDA),
    (re.compile(r"\bMetal\d*\s*:", re.IGNORECASE), Backend.METAL),
    (re.compile(r"\bVulkan\d*\s*:", re.IGNORECASE), Backend.VULKAN),
)

_LDD_LIBRARY_PATTERNS: tuple[tuple[re.Pattern[str], Backend], ...] = (
    (re.compile(r"libggml-cuda\.so"), Backend.CUDA),
    (re.compile(r"libggml-metal\.so"), Backend.METAL),
    (re.compile(r"libggml-vulkan\.so"), Backend.VULKAN),
)


def _backend_from_list_devices(output: str) -> Backend | None:
    """Parse ``--list-devices`` output; ``None`` means the output was not recognisable at all."""
    for pattern, backend in _LIST_DEVICES_PATTERNS:
        if pattern.search(output):
            return backend
    if "Available devices:" in output:
        return Backend.CPU  # ran cleanly, listed none -- a CPU-only build says exactly this
    return None


def _backend_from_ldd(binary: Path) -> Backend:
    """Link-time fallback: which ``libggml-*`` backend library is linked into ``binary``."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return Backend.UNKNOWN
    try:
        out = subprocess.run(  # noqa: S603 — ldd is a shutil.which() result, not a literal
            [ldd, str(binary)],
            capture_output=True,
            text=True,
            timeout=_LDD_TIMEOUT_S,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return Backend.UNKNOWN
    for pattern, backend in _LDD_LIBRARY_PATTERNS:
        if pattern.search(out):
            return backend
    return Backend.CPU if "libggml" in out else Backend.UNKNOWN


def detect_backend(binary: Path) -> Backend:
    """Detect the compute backend ``binary`` (a llama-server build) actually serves on.

    Read-only: never loads a model or allocates GPU memory, so it is always safe to call,
    including under ``--dry-run`` and while another process holds the GPU.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — binary path is caller-resolved, not user input
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=_LIST_DEVICES_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _backend_from_ldd(binary)

    backend = _backend_from_list_devices(proc.stdout + proc.stderr)
    if backend is not None:
        return backend
    return _backend_from_ldd(binary)
