"""`localmodels setup`: one-command hardware-aware bootstrap for a local chat model.

Composed from this package's own pieces (`preflight`, `pull_model`) plus this task's
`backend`/`bench`/`download` modules, mirroring service_c's own `setup.py` step shape
(idempotent steps, a `StepResult` per step, `--dry-run`-able) without importing service_c --
this package is torch-free and depends on neither service_a nor service_c.

Every step below is real, including under `--dry-run`, EXCEPT download (step 3) and lease+bench
(step 5): hardware inspection, model selection and backend detection are all pure/read-only (no
network, no GPU mutation, no subprocess call that loads a model or blocks on a lock file), so a
`--dry-run` run still reports real measurements. Only the two steps that would move bytes or
touch the shared GPU lease are gated by `--dry-run`/`--yes`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .backend import Backend, detect_backend
from .bench import BenchError, BenchResult, run_bench
from .catalogue import LicencePolicy, ModelCategory, ModelSpec
from .download import NoCatalogueMatchError, choose_model, download_model
from .enums import DeviceType, ModelSizeClass, PreflightVerdict
from .lease import GpuLease, LeaseRecord
from .preflight import PreflightResult, preflight
from .process import resolve_llama_server_bin
from .pull import model_status

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

DEFAULT_MODELS_DIR = Path("~/.local/share/localmodels/models").expanduser()
DEFAULT_STATE_PATH = Path("~/.local/state/localmodels/setup.json").expanduser()
DEFAULT_LEASE_DIR = Path("~/.local/state/gpu-lease").expanduser()
DEFAULT_PROJECT = "localmodels-setup"

__version__ = "0.2.0"


class Role(StrEnum):
    """Closed set of catalogue roles `localmodels setup` understands (extend as roles are added)."""

    CHAT = "chat"


ROLE_TO_CATEGORY: dict[Role, ModelCategory] = {Role.CHAT: ModelCategory.GENERATOR}


# Bootstrapping default (2026-09-25): `localmodels setup`'s own --dry-run verify command omits
# --catalogue, so this command must work without one. This is deliberately NOT "the" catalogue
# `localmodels.catalogue`'s docstring says this package never ships (service_a's OSI-gated
# production list and service_c's, including non-commercial detectors, each stay in their own
# repository) -- it exists only so this command is usable standalone, and --catalogue overrides
# it entirely. Entries and pinned revisions are carried from service_a's own already-verified
# registry (models/registry.py KNOWN_MODELS, apache-2.0/mit, both OSI-approved, measured on this
# DGX 2026-09-08) rather than invented here. `default_for_category` marks the one pick per size
# class: phi-4-mini over qwen3-4b for SMALL because service_a's own registry records it as the
# one measured on CPU (evidence="schema-valid classification in the D20 contract probe").
DEFAULT_CATALOGUE: tuple[ModelSpec, ...] = (
    ModelSpec(
        alias="qwen3-30b-a3b",
        category=ModelCategory.GENERATOR,
        repo_id="Qwen/Qwen3-30B-A3B-GGUF",
        filename="Qwen3-30B-A3B-Q4_K_M.gguf",
        revision="e4d4bafdfb96a411a163846265362aceb0b9c63a",
        description="workhorse MoE chat/generation model",
        licence="apache-2.0",
        osi_approved=True,
        size_class=ModelSizeClass.LARGE,
        default_for_category=True,
        local=True,
    ),
    ModelSpec(
        alias="qwen3-8b",
        category=ModelCategory.GENERATOR,
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        revision="7c41481f57cb95916b40956ab2f0b139b296d974",
        description="mid-size chat/generation model",
        licence="apache-2.0",
        osi_approved=True,
        size_class=ModelSizeClass.MEDIUM,
        default_for_category=True,
        local=True,
    ),
    ModelSpec(
        alias="phi-4-mini",
        category=ModelCategory.GENERATOR,
        repo_id="unsloth/Phi-4-mini-instruct-GGUF",
        filename="Phi-4-mini-instruct-Q4_K_M.gguf",
        revision="78eb92a46fc37e6b524df991ed9aca9bc6aa7b80",
        description="small chat/generation model, CPU-viable",
        licence="mit",
        osi_approved=True,
        size_class=ModelSizeClass.SMALL,
        default_for_category=True,
        local=True,
    ),
)


class SetupStepName(StrEnum):
    """Closed set of `localmodels setup` steps, in execution order."""

    HARDWARE = "hardware"
    CHOOSE_MODEL = "choose_model"
    DOWNLOAD = "download"
    BACKEND = "backend"
    BENCH = "bench"


class StepStatus(StrEnum):
    """Closed set of outcomes for one setup step."""

    DONE = "done"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    PLANNED = "planned"
    FAILED = "failed"


class StepResult(BaseModel):
    """One step's outcome: what happened (or would happen), and why."""

    step: SetupStepName
    name: str
    status: StepStatus
    detail: str


class SetupReport(BaseModel):
    """The full `localmodels setup` report: every step, plus what was measured/chosen."""

    dry_run: bool
    preflight: PreflightResult | None = None
    chosen_model: str | None = None
    backend: Backend | None = None
    bench: BenchResult | None = None
    results: list[StepResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every step that ran (or was planned) is free of a FAILED result."""
        return not any(r.status is StepStatus.FAILED for r in self.results)


def refuse_cpu_only_build(*, backend: Backend, device: DeviceType, device_name: str) -> str | None:
    """A refusal detail naming the rebuild flag, or ``None`` if this combination is fine.

    Refuses only a CPU-only *build* running on a host that measured an accelerator -- a CPU-only
    build on a CPU-only host is correct, not a refusal (`Backend.CPU` alone is not the trigger,
    the mismatch with measured hardware is).
    """
    if backend is not Backend.CPU or device is DeviceType.CPU:
        return None
    rebuild_flag = {
        DeviceType.CUDA: "-DGGML_CUDA=ON",
        DeviceType.MPS: "-DGGML_METAL=ON",
    }.get(device, "-DGGML_VULKAN=ON")
    return (
        f"llama-server is a CPU-only build, but this host has a {device.value} accelerator "
        f"({device_name}); rebuild with {rebuild_flag} and re-run 'localmodels setup'."
    )


def _run_bench_under_lease(
    lease: GpuLease,
    *,
    project: str,
    model_path: Path,
    bench_bin: Path,
    bench_fn: Callable[[Path, Path], BenchResult] = run_bench,
) -> tuple[BenchResult | None, str]:
    """Acquire ``lease``, benchmark ``model_path``, then always release it.

    Refuses (returns ``None`` plus a detail naming the holder) when another project's lease is
    already live -- this step never contends with a running server; it simply reports "try
    again once it clears" rather than blocking `setup` on someone else's server.
    """
    others = lease.live_others()
    if others:
        holders = ", ".join(o.project for o in others)
        return None, f"GPU lease held by: {holders}; run 'localmodels setup' again once it clears"

    lease.write(
        LeaseRecord(pid=os.getpid(), project=project, model_path=str(model_path), started_at=int(time.time()))
    )
    try:
        result = bench_fn(bench_bin, model_path)
    except BenchError as exc:
        return None, str(exc)
    finally:
        lease.remove()
    return result, f"prompt={result.prompt_tok_s} tok/s, decode={result.decode_tok_s} tok/s"


def _hardware_step(*, device_override: str | None) -> tuple[StepResult, PreflightResult]:
    pf = preflight(device_override=device_override)
    blocked = pf.verdict is PreflightVerdict.LOCAL_MODELS_INFEASIBLE
    remedy = f" -- remedy: {pf.remedy}" if pf.remedy else ""
    result = StepResult(
        step=SetupStepName.HARDWARE,
        name="preflight",
        status=StepStatus.BLOCKED if blocked else StepStatus.DONE,
        detail=f"{pf.verdict.value}: {pf.message}{remedy}",
    )
    return result, pf


def _choose_model_step(
    catalogue_specs: list[ModelSpec], *, role: Role, size_class: ModelSizeClass
) -> tuple[StepResult, ModelSpec | None]:
    name = SetupStepName.CHOOSE_MODEL
    category = ROLE_TO_CATEGORY[role]
    try:
        spec = choose_model(
            catalogue_specs, category=category, size_class=size_class, policy=LicencePolicy.OSI_ONLY
        )
    except NoCatalogueMatchError as exc:
        result = StepResult(step=name, name=role.value, status=StepStatus.FAILED, detail=str(exc))
        return result, None
    detail = (
        f"{spec.repo_id}@{spec.revision} ({spec.filename}), "
        f"size_class={size_class.value}, licence={spec.licence}"
    )
    result = StepResult(step=name, name=spec.alias, status=StepStatus.DONE, detail=detail)
    return result, spec


def _download_step(spec: ModelSpec, *, models_dir: Path, dry_run: bool, yes: bool) -> tuple[StepResult, Path]:
    """Download `spec` (gated by `dry_run`/`yes`); cached is always SKIPPED regardless of the gate."""
    step = SetupStepName.DOWNLOAD
    status_info = model_status(spec, models_dir=models_dir)
    default_path = models_dir / (spec.filename or spec.alias)
    model_path = Path(status_info["local_path"]) if status_info["local_path"] else default_path

    if status_info["cached"]:
        detail = f"already cached at {model_path}"
        return StepResult(step=step, name=spec.alias, status=StepStatus.SKIPPED, detail=detail), model_path
    if dry_run or not yes:
        detail = (
            f"would pull {spec.repo_id}@{spec.revision} to {models_dir} "
            "-- rerun with --yes (and without --dry-run) to download"
        )
        return StepResult(step=step, name=spec.alias, status=StepStatus.PLANNED, detail=detail), model_path
    try:
        model_path = download_model(spec, models_dir=models_dir)
    except Exception as exc:  # noqa: BLE001 — a download failure is reported, not a crash
        detail = f"{spec.repo_id}: {exc}"
        return StepResult(step=step, name=spec.alias, status=StepStatus.FAILED, detail=detail), model_path
    detail = f"downloaded to {model_path}"
    return StepResult(step=step, name=spec.alias, status=StepStatus.DONE, detail=detail), model_path


def _backend_step(
    *, binary: Path | None, device: DeviceType, device_name: str
) -> tuple[StepResult, Backend | None, Path | None]:
    """Resolve the llama-server binary and detect its backend -- always real, see module docstring."""
    try:
        resolved_binary = binary or resolve_llama_server_bin()
    except Exception as exc:  # noqa: BLE001 — reported as a step failure, not a crash
        step, detail = SetupStepName.BACKEND, str(exc)
        result = StepResult(step=step, name="detect", status=StepStatus.FAILED, detail=detail)
        return result, None, None

    backend = detect_backend(resolved_binary)
    refusal = refuse_cpu_only_build(backend=backend, device=device, device_name=device_name)
    name = "detect"
    if refusal is not None:
        result = StepResult(step=SetupStepName.BACKEND, name=name, status=StepStatus.BLOCKED, detail=refusal)
    else:
        detail = f"backend={backend.value}"
        result = StepResult(step=SetupStepName.BACKEND, name=name, status=StepStatus.DONE, detail=detail)
    return result, backend, resolved_binary


def _bench_step(
    *,
    dry_run: bool,
    yes: bool,
    backend_refused: bool,
    downloaded: bool,
    lease: GpuLease,
    model_path: Path,
    resolved_binary: Path | None,
    bench_fn: Callable[[Path, Path], BenchResult],
) -> tuple[StepResult, BenchResult | None]:
    """Acquire the GPU lease and bench -- gated, and only once step 3 put a model on disk."""
    name = "llama-bench"
    if dry_run or not yes:
        detail = "would acquire the GPU lease and run llama-bench -p 512 -n 128 -r 3 -o json, then release it"
        return StepResult(step=SetupStepName.BENCH, name=name, status=StepStatus.PLANNED, detail=detail), None
    if backend_refused:
        detail = "skipped: backend check refused"
        return StepResult(step=SetupStepName.BENCH, name=name, status=StepStatus.BLOCKED, detail=detail), None
    if not downloaded:
        detail = "skipped: no model on disk"
        return StepResult(step=SetupStepName.BENCH, name=name, status=StepStatus.BLOCKED, detail=detail), None

    bench_bin = (resolved_binary or resolve_llama_server_bin()).parent / "llama-bench"
    bench_result, detail = _run_bench_under_lease(
        lease, project=lease.project, model_path=model_path, bench_bin=bench_bin, bench_fn=bench_fn
    )
    status = StepStatus.DONE if bench_result is not None else StepStatus.BLOCKED
    return StepResult(step=SetupStepName.BENCH, name=name, status=status, detail=detail), bench_result


def run_setup(
    *,
    role: Role = Role.CHAT,
    models_dir: Path = DEFAULT_MODELS_DIR,
    catalogue: list[ModelSpec] | None = None,
    lease: GpuLease | None = None,
    dry_run: bool = False,
    yes: bool = False,
    device_override: str | None = None,
    binary: Path | None = None,
) -> SetupReport:
    """Run the bootstrap pipeline: hardware -> choose model -> download -> backend -> bench.

    Steps 1 (hardware), 2 (choose model) and 4 (backend) are always real -- see the module
    docstring. Step 3 (download) and step 5 (lease + bench) are the only steps `--dry-run`/
    without `--yes` turns into a plan rather than an action. Each step is its own function
    (`_hardware_step`, `_choose_model_step`, ...) so this orchestrator stays a short sequence.
    """
    catalogue_specs = list(catalogue) if catalogue is not None else list(DEFAULT_CATALOGUE)
    resolved_lease = lease or GpuLease(DEFAULT_LEASE_DIR, project=DEFAULT_PROJECT)
    results: list[StepResult] = []

    hardware_result, pf = _hardware_step(device_override=device_override)
    results.append(hardware_result)
    if pf.verdict is PreflightVerdict.LOCAL_MODELS_INFEASIBLE:
        return SetupReport(dry_run=dry_run, preflight=pf, results=results)

    choose_result, spec = _choose_model_step(
        catalogue_specs, role=role, size_class=pf.serving_params.recommended_size_class
    )
    results.append(choose_result)
    if spec is None:
        return SetupReport(dry_run=dry_run, preflight=pf, results=results)

    download_result, model_path = _download_step(spec, models_dir=models_dir, dry_run=dry_run, yes=yes)
    results.append(download_result)
    if download_result.status is StepStatus.FAILED:
        return SetupReport(dry_run=dry_run, preflight=pf, chosen_model=spec.alias, results=results)
    downloaded = download_result.status in (StepStatus.DONE, StepStatus.SKIPPED)

    backend_result, backend, resolved_binary = _backend_step(
        binary=binary, device=pf.report.device, device_name=pf.report.device_name
    )
    results.append(backend_result)
    backend_refused = backend_result.status is StepStatus.BLOCKED

    bench_result, bench = _bench_step(
        dry_run=dry_run,
        yes=yes,
        backend_refused=backend_refused,
        downloaded=downloaded,
        lease=resolved_lease,
        model_path=model_path,
        resolved_binary=resolved_binary,
        bench_fn=run_bench,
    )
    results.append(bench_result)

    return SetupReport(
        dry_run=dry_run, preflight=pf, chosen_model=spec.alias, backend=backend, bench=bench, results=results
    )


def _serialise_report(report: SetupReport) -> dict[str, object]:
    return {
        "recorded_at": int(time.time()),
        "localmodels_version": __version__,
        "hardware": report.preflight.report.model_dump(mode="json") if report.preflight else None,
        "preflight_verdict": report.preflight.verdict.value if report.preflight else None,
        "chosen_model": report.chosen_model,
        "backend": report.backend.value if report.backend else None,
        "bench": report.bench.model_dump(mode="json") if report.bench else None,
        "steps": [r.model_dump(mode="json") for r in report.results],
    }


def record_setup(report: SetupReport, *, state_path: Path = DEFAULT_STATE_PATH) -> None:
    """Persist ``report`` to ``state_path`` atomically.

    Temp file + rename, `GpuLease.write`'s own pattern -- never leaves a half-written state
    file behind.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialise_report(report)
    fd, tmp = tempfile.mkstemp(dir=state_path.parent, prefix="setup.", suffix=".json.tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    Path(tmp).replace(state_path)


_STATUS_MARKER: dict[StepStatus, str] = {
    StepStatus.DONE: "OK",
    StepStatus.SKIPPED: "SKIP",
    StepStatus.BLOCKED: "BLOCKED",
    StepStatus.PLANNED: "PLAN",
    StepStatus.FAILED: "FAIL",
}


def print_summary(report: SetupReport, stream: TextIO | None = None) -> None:
    """Print a plain-language explanation of what `localmodels setup` did (or would do)."""
    import sys  # noqa: PLC0415 — only needed for the default stream

    out = stream or sys.stdout
    print("\n=== localmodels setup summary ===", file=out)
    if report.preflight is not None:
        rep = report.preflight.report
        print(
            f"Hardware:  {rep.device.value} ({rep.device_name}), {rep.total_memory_gb:.1f} GB, "
            f"tier={rep.tier.value}, verdict={report.preflight.verdict.value}",
            file=out,
        )
    for r in report.results:
        marker = _STATUS_MARKER[r.status]
        print(f"  [{marker:<7}] {r.step.value:<12} {r.name:<22} {r.detail}", file=out)
    if report.dry_run:
        print("\n(dry run: no changes were made)", file=out)
    elif not report.ok:
        print("\nOne or more steps failed; see FAIL lines above.", file=out)
    print("\nTo start serving this model, run: localmodels serve --config <toml>", file=out)
