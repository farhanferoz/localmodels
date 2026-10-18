"""Unit tests for llama-server backend detection.

`--list-devices` output shapes below are verbatim from this DGX's real build (verified
2026-09-25, `llama-server --list-devices` -> "Available devices:\\n  CUDA0: NVIDIA GB10 ...")
and from llama.cpp's device-listing source for the Metal/Vulkan/CPU-only shapes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import localmodels.backend as backend_mod
from localmodels.backend import Backend, detect_backend

_CUDA_LIST_DEVICES = "Available devices:\n  CUDA0: NVIDIA GB10 (124610 MiB, 86812 MiB free)\n"
_METAL_LIST_DEVICES = "Available devices:\n  Metal0: Apple M2 Pro (24576 MiB, 20000 MiB free)\n"
_VULKAN_LIST_DEVICES = "Available devices:\n  Vulkan0: AMD Radeon (16384 MiB, 12000 MiB free)\n"
_CPU_ONLY_LIST_DEVICES = "Available devices:\n"  # a CPU-only build lists no accelerator devices


def _fake_run(monkeypatch, *, stdout: str = "", returncode: int = 0, raises: bool = False) -> None:
    def _run(*_a, **_k):
        if raises:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=10)
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)


@pytest.mark.parametrize(
    ("listing", "backend"),
    [
        (_CUDA_LIST_DEVICES, Backend.CUDA),
        (_METAL_LIST_DEVICES, Backend.METAL),
        (_VULKAN_LIST_DEVICES, Backend.VULKAN),
        (_CPU_ONLY_LIST_DEVICES, Backend.CPU),
    ],
    ids=["cuda", "metal", "vulkan", "cpu-only"],
)
def test_backend_detected_from_list_devices(monkeypatch, listing: str, backend: Backend) -> None:
    _fake_run(monkeypatch, stdout=listing)
    assert detect_backend(Path("/fake/llama-server")) is backend


def test_falls_back_to_ldd_when_list_devices_cannot_run(monkeypatch) -> None:
    _fake_run(monkeypatch, raises=True)
    monkeypatch.setattr(
        backend_mod,
        "_backend_from_ldd",
        lambda _binary: Backend.CUDA,
    )
    assert detect_backend(Path("/fake/llama-server")) is Backend.CUDA


@pytest.mark.parametrize(
    ("accelerator_library", "backend"),
    [("cuda", Backend.CUDA), ("cpu", Backend.CPU)],
    ids=["cuda", "cpu-only"],
)
def test_ldd_reads_the_linked_ggml_backend_library(
    monkeypatch, accelerator_library: str, backend: Backend
) -> None:
    library = f"libggml-{accelerator_library}.so.0"
    ldd_out = f"\tlibggml.so.0 => /x/libggml.so.0\n\t{library} => /x/{library}\n"

    def _run(*_a, **_k):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=ldd_out, stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    assert backend_mod._backend_from_ldd(Path("/fake/llama-server")) is backend


def test_ldd_unknown_when_it_cannot_run_either(monkeypatch) -> None:
    def _run(*_a, **_k):
        raise OSError("ldd not found")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    assert backend_mod._backend_from_ldd(Path("/fake/llama-server")) is Backend.UNKNOWN
