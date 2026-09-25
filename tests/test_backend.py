"""Unit tests for llama-server backend detection.

`--list-devices` output shapes below are verbatim from this DGX's real build (verified
2026-09-25, `llama-server --list-devices` -> "Available devices:\\n  CUDA0: NVIDIA GB10 ...")
and from llama.cpp's device-listing source for the Metal/Vulkan/CPU-only shapes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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


def test_cuda_detected_from_list_devices(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout=_CUDA_LIST_DEVICES)
    assert detect_backend(Path("/fake/llama-server")) is Backend.CUDA


def test_metal_detected_from_list_devices(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout=_METAL_LIST_DEVICES)
    assert detect_backend(Path("/fake/llama-server")) is Backend.METAL


def test_vulkan_detected_from_list_devices(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout=_VULKAN_LIST_DEVICES)
    assert detect_backend(Path("/fake/llama-server")) is Backend.VULKAN


def test_cpu_only_build_detected_from_list_devices(monkeypatch) -> None:
    _fake_run(monkeypatch, stdout=_CPU_ONLY_LIST_DEVICES)
    assert detect_backend(Path("/fake/llama-server")) is Backend.CPU


def test_falls_back_to_ldd_when_list_devices_cannot_run(monkeypatch) -> None:
    _fake_run(monkeypatch, raises=True)
    monkeypatch.setattr(
        backend_mod,
        "_backend_from_ldd",
        lambda _binary: Backend.CUDA,
    )
    assert detect_backend(Path("/fake/llama-server")) is Backend.CUDA


def test_ldd_detects_cuda_library(monkeypatch) -> None:
    ldd_out = "\tlibggml.so.0 => /x/libggml.so.0\n\tlibggml-cuda.so.0 => /x/libggml-cuda.so.0\n"

    def _run(*_a, **_k):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=ldd_out, stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    assert backend_mod._backend_from_ldd(Path("/fake/llama-server")) is Backend.CUDA


def test_ldd_cpu_only_when_no_accelerator_library_linked(monkeypatch) -> None:
    ldd_out = "\tlibggml.so.0 => /x/libggml.so.0\n\tlibggml-cpu.so.0 => /x/libggml-cpu.so.0\n"

    def _run(*_a, **_k):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=ldd_out, stderr="")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    assert backend_mod._backend_from_ldd(Path("/fake/llama-server")) is Backend.CPU


def test_ldd_unknown_when_it_cannot_run_either(monkeypatch) -> None:
    def _run(*_a, **_k):
        raise OSError("ldd not found")

    monkeypatch.setattr(backend_mod.subprocess, "run", _run)
    assert backend_mod._backend_from_ldd(Path("/fake/llama-server")) is Backend.UNKNOWN


def test_real_binary_on_this_dgx_reports_cuda() -> None:
    """Live check against the DGX-verified fact: this build is CUDA-only."""
    real_binary = Path("/home/user1/dev/llama.cpp/build/bin/llama-server")
    if not real_binary.exists():
        return
    assert detect_backend(real_binary) is Backend.CUDA
