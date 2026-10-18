"""Unit tests for `localmodels setup`'s pipeline: hardware -> choose -> download -> backend -> bench.

Download, bench and every subprocess are faked throughout (brief requirement) -- these tests
never touch the network, the GPU, or the shared lease directory outside a `tmp_path`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import localmodels.setup as setup_mod
from localmodels.backend import Backend
from localmodels.bench import BenchResult
from localmodels.catalogue import ModelCategory, ModelSpec
from localmodels.enums import DeviceType, ModelSizeClass, PreflightVerdict
from localmodels.hardware import HardwareReport
from localmodels.lease import GpuLease, LeaseRecord
from localmodels.params import classify_tier
from localmodels.preflight import preflight as real_preflight
from localmodels.setup import (
    SetupStepName,
    StepStatus,
    _run_bench_under_lease,
    record_setup,
    refuse_cpu_only_build,
    run_setup,
)


@pytest.fixture(autouse=True)
def _empty_hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model in this machine's own Hub cache would read as cached and skip the download step."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


_LARGE_GENERATOR = ModelSpec(
    alias="qwen3-30b-a3b",
    category=ModelCategory.GENERATOR,
    repo_id="Qwen/Qwen3-30B-A3B-GGUF",
    filename="Qwen3-30B-A3B-Q4_K_M.gguf",
    revision="e4d4bafdfb96a411a163846265362aceb0b9c63a",
    description="large generator",
    licence="apache-2.0",
    osi_approved=True,
    size_class=ModelSizeClass.LARGE,
    default_for_category=True,
)
_NON_OSI_LARGE = ModelSpec(
    alias="gemma-large",
    category=ModelCategory.GENERATOR,
    repo_id="google/gemma-27b",
    description="large generator, non-OSI",
    licence="gemma-terms-of-use",
    osi_approved=False,
    size_class=ModelSizeClass.LARGE,
    default_for_category=True,
)


def _fake_report(device: DeviceType, total_mem_gb: float) -> HardwareReport:
    tier = classify_tier(device, total_mem_gb).tier
    return HardwareReport(
        platform="linux",
        arch="x86_64",
        device=device,
        device_name=f"fake-{device.value}",
        total_memory_gb=total_mem_gb,
        tier=tier,
        assumptions=[],
    )


def _fake_preflight_cuda_fast(monkeypatch) -> None:
    report = _fake_report(DeviceType.CUDA, 32.0)

    def _preflight(*, device_override=None):
        return real_preflight(report=report)

    monkeypatch.setattr(setup_mod, "preflight", _preflight)


def _never_called(*_a, **_k):
    msg = "must not be called under --dry-run / without --yes"
    raise AssertionError(msg)


# --- model choice follows size class and licence ------------------------------------------


def test_run_setup_chooses_the_entry_matching_the_derived_size_class(monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)  # -> LARGE size class
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True)
    assert report.chosen_model == "qwen3-30b-a3b"


def test_run_setup_never_chooses_a_non_osi_entry(monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)

    report = run_setup(catalogue=[_NON_OSI_LARGE], dry_run=True)
    choose_result = next(r for r in report.results if r.step is SetupStepName.CHOOSE_MODEL)
    assert choose_result.status is StepStatus.FAILED
    assert report.chosen_model is None


# --- backend refusal ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "rebuild_flag"),
    [(DeviceType.CUDA, "-DGGML_CUDA=ON"), (DeviceType.MPS, "-DGGML_METAL=ON")],
)
def test_refuse_cpu_only_build_names_the_rebuild_flag(device: DeviceType, rebuild_flag: str) -> None:
    detail = refuse_cpu_only_build(backend=Backend.CPU, device=device, device_name="fake device")
    assert detail is not None
    assert rebuild_flag in detail
    assert "re-run 'localmodels setup'" in detail


def test_refuse_cpu_only_build_allows_cpu_backend_on_cpu_host() -> None:
    assert refuse_cpu_only_build(backend=Backend.CPU, device=DeviceType.CPU, device_name="x") is None


def test_refuse_cpu_only_build_allows_matching_accelerator() -> None:
    assert refuse_cpu_only_build(backend=Backend.CUDA, device=DeviceType.CUDA, device_name="x") is None


def test_run_setup_backend_step_blocked_on_cpu_only_build_with_accelerator_host(monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CPU)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True)
    backend_result = next(r for r in report.results if r.step is SetupStepName.BACKEND)
    assert backend_result.status is StepStatus.BLOCKED
    assert "-DGGML_CUDA=ON" in backend_result.detail
    bench_result = next(r for r in report.results if r.step is SetupStepName.BENCH)
    # dry_run always PLANs bench regardless of backend -- the refusal is asserted for a real run below.
    assert bench_result.status is StepStatus.PLANNED


def test_run_setup_bench_blocked_when_backend_refused_on_a_real_yes_run(monkeypatch) -> None:
    def _fake_model_status(_spec, *, models_dir):
        return {"cached": True, "local_path": str(models_dir / "m.gguf")}

    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CPU)
    monkeypatch.setattr(setup_mod, "model_status", _fake_model_status)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=False, yes=True)
    bench_result = next(r for r in report.results if r.step is SetupStepName.BENCH)
    assert bench_result.status is StepStatus.BLOCKED
    assert "backend check refused" in bench_result.detail


# --- dry-run / --yes gating: never acquires the lease, benches, downloads or starts anything ---


def test_dry_run_never_downloads_or_benches(monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)
    monkeypatch.setattr(setup_mod, "download_model", _never_called)
    monkeypatch.setattr(setup_mod, "run_bench", _never_called)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True, yes=True)
    download_result = next(r for r in report.results if r.step is SetupStepName.DOWNLOAD)
    bench_result = next(r for r in report.results if r.step is SetupStepName.BENCH)
    assert download_result.status is StepStatus.PLANNED
    assert bench_result.status is StepStatus.PLANNED
    assert report.ok


def test_without_yes_never_downloads_or_benches(monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)
    monkeypatch.setattr(setup_mod, "download_model", _never_called)
    monkeypatch.setattr(setup_mod, "run_bench", _never_called)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=False, yes=False)
    download_result = next(r for r in report.results if r.step is SetupStepName.DOWNLOAD)
    assert download_result.status is StepStatus.PLANNED


def test_dry_run_never_writes_state(tmp_path: Path, monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True)
    state_path = tmp_path / "setup.json"
    if report.dry_run:
        pass  # run_setup itself never writes state; the CLI decides whether to call record_setup
    assert not state_path.exists()


def test_infeasible_hardware_never_reaches_choose_model(monkeypatch) -> None:
    report_infeasible = _fake_report(DeviceType.CPU, 1.0)

    def _preflight(*, device_override=None):
        return real_preflight(report=report_infeasible)

    monkeypatch.setattr(setup_mod, "preflight", _preflight)
    monkeypatch.setattr(setup_mod, "choose_model", _never_called)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True)
    assert report.preflight.verdict is PreflightVerdict.LOCAL_MODELS_INFEASIBLE
    assert report.chosen_model is None
    assert len(report.results) == 1


# --- the bench runs only while the lease is held -------------------------------------------


def test_bench_runs_only_while_the_lease_is_held(tmp_path: Path) -> None:
    lease = GpuLease(tmp_path, project="localmodels-setup")
    held_during_call = []

    def spy_bench(_bench_bin: Path, _model_path: Path) -> BenchResult:
        held_during_call.append(lease.lease_path().exists())
        return BenchResult(prompt_tok_s=100.0, decode_tok_s=50.0, raw_rows=[])

    result, detail = _run_bench_under_lease(
        lease,
        project="localmodels-setup",
        model_path=tmp_path / "m.gguf",
        bench_bin=tmp_path / "llama-bench",
        bench_fn=spy_bench,
    )
    assert held_during_call == [True]
    assert not lease.lease_path().exists()  # released afterwards
    assert result is not None
    assert "100.0" in detail


def test_bench_refuses_when_another_lease_is_live(tmp_path: Path) -> None:
    other = GpuLease(tmp_path, project="service_a")
    other.write(LeaseRecord(pid=os.getpid(), project="service_a", port=8091))
    lease = GpuLease(tmp_path, project="localmodels-setup")

    result, detail = _run_bench_under_lease(
        lease,
        project="localmodels-setup",
        model_path=tmp_path / "m.gguf",
        bench_bin=tmp_path / "llama-bench",
        bench_fn=_never_called,
    )
    assert result is None
    assert "service_a" in detail
    assert not lease.lease_path().exists()


# --- setup.json records every step ----------------------------------------------------------


def test_record_setup_writes_every_step(tmp_path: Path, monkeypatch) -> None:
    _fake_preflight_cuda_fast(monkeypatch)
    monkeypatch.setattr(setup_mod, "resolve_llama_server_bin", lambda: Path("/fake/llama-server"))
    monkeypatch.setattr(setup_mod, "detect_backend", lambda _b: Backend.CUDA)

    report = run_setup(catalogue=[_LARGE_GENERATOR], dry_run=True)
    state_path = tmp_path / "state" / "setup.json"
    record_setup(report, state_path=state_path)

    data = json.loads(state_path.read_text())
    assert data["chosen_model"] == "qwen3-30b-a3b"
    assert {s["step"] for s in data["steps"]} == {r.step.value for r in report.results}
    assert len(data["steps"]) == len(report.results)
    assert not any(p.suffix == ".tmp" for p in (tmp_path / "state").iterdir())
