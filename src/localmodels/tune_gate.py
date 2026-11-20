"""`localmodels settings require`: refuse a long GPU job whose model has no measured settings on this machine.

A project calls it before submitting a GPU job (design
docs/research/in-process-tune-design-2026-11-20.md, section 2.7). A job expected to run no longer than
``--max-untuned-hours`` is allowed as it is. A longer one is allowed when ``settings.lookup`` answers from a
record of this machine: tuned, or stale, which D2 keeps in use until its confirm tune lands. It is refused
when only a rule or the caller's default applies, and the refusal names the tune to run, or the tune job
``lookup`` queued when the caller gave the loader's flags. Every decision, allows included, is one line on
stderr and one JSON line in ``<state>/settings/gate.jsonl``.
"""

from __future__ import annotations

import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from localmodels.enums import DeviceType, WorkloadKind
from localmodels.hardware import inspect_hardware
from localmodels.stack import (
    AMDGPU_VERSION_FILE,
    NVIDIA_VERSION_FILE,
    Engine,
    SettingsStatus,
    StoreState,
    TuneState,
)

if TYPE_CHECKING:
    import argparse

    from localmodels.hardware import HardwareReport
    from localmodels.settings import ModelIdentity, SettingsAnswer, TuneRecipe

#: owner-approved request of 2026-11-19 (tracecraft): "refuses a GPU job expected over ~1 h"; a flag's default
DEFAULT_MAX_UNTUNED_HOURS: Final = 1.0
GATE_LOG: Final = "gate.jsonl"  # in the settings store, beside the records
#: the engines a record can be measured for here: this package's in-process tune and llama_tune
GATE_ENGINES: Final = (Engine.PYTORCH, Engine.LLAMA_CPP)
LLAMA_TUNE_KINDS: Final = (
    WorkloadKind.K1,
    WorkloadKind.K2,
    WorkloadKind.K3,
)  # what llama_tune's --kind accepts
#: files that exist only on a host with a GPU driver loaded (``stack``'s driver readings)
DRIVER_FILES: Final = (NVIDIA_VERSION_FILE, AMDGPU_VERSION_FILE)
_ALLOWING: Final = frozenset({SettingsStatus.TUNED, SettingsStatus.STALE})
_JOB_QUEUED: Final = frozenset({TuneState.QUEUED, TuneState.ALREADY_QUEUED})


class GateDecision(BaseModel):
    """Whether a job may be submitted, why, and what to do when it may not."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str
    remedy: str = ""
    hardware: str = ""
    status: SettingsStatus | None = None
    record_id: str | None = None
    stale: list[str] = Field(default_factory=list)
    tune: TuneState | None = None
    tune_detail: str | None = None


def _tune_command(model: ModelIdentity, kind: WorkloadKind, engine: Engine, recipe: TuneRecipe | None) -> str:
    """The command that measures ``model`` here: the recipe's, else the tune's with placeholders to fill."""
    if recipe is not None:
        return shlex.join(recipe.command)
    if engine is Engine.LLAMA_CPP:
        files = " ".join(f"--file {shlex.quote(f.path)}" for f in model.files)
        return (
            f"python -m localmodels.llama_tune --repository {model.repository} --revision {model.revision} "
            f"{files} --kind {kind.value} --role <role> --prompts <prompts.jsonl> --prompt-tokens <n> "
            "--run-dir <dir>"
        )
    return (
        f"localmodels settings tune --project <project> --loader <module:function> --entry <entry> "
        f"--model {model.repository} --revision {model.revision} --kind {kind.value} --texts <texts.jsonl>"
    )


def _shape_problem(model: ModelIdentity, kind: WorkloadKind, engine: Engine) -> str:
    """Why this identity, kind and engine can never have a record the tunes write here; "" when they can."""
    if engine not in GATE_ENGINES:
        names = ", ".join(e.value for e in GATE_ENGINES)
        return (
            f"{engine.value} has no tune here, so no record can be measured for it; the check covers {names}"
        )
    if engine is Engine.PYTORCH and model.files:
        return "an in-process model's records are filed by its snapshot: give --revision without --file"
    if engine is Engine.LLAMA_CPP and not model.files:
        return "a llama.cpp model's records are filed by its files: give --file for each part"
    if engine is Engine.LLAMA_CPP and kind not in LLAMA_TUNE_KINDS:
        return f"llama_tune measures {', '.join(k.value for k in LLAMA_TUNE_KINDS)} only, not {kind.value}"
    return ""


def _hidden_gpu(report: HardwareReport) -> str:
    """The driver file of a GPU this process cannot see ("" when it sees one, or the host has none)."""
    if report.device is not DeviceType.CPU:
        return ""
    return next((str(path) for path in DRIVER_FILES if path.exists()), "")


def _why_refused(answer: SettingsAnswer, hardware: str) -> str:
    """Why no measured record answers ``answer``'s model: its status, the store's state, other identities."""
    from localmodels.settings import stored_identities  # noqa: PLC0415 - the lookup's module, loaded by the check

    name = f"{answer.model.repository} ({answer.kind.value}, {answer.engine.value})"
    parts = [f"no measured record for {name} on this machine ({hardware})"]
    if answer.rule is not None:
        measured = "this model" if answer.rule.applied else ", ".join(answer.rule.models) or "no named model"
        parts.append(
            f"only rule {answer.rule.id} applies, measured on {measured}, not on this file and machine"
        )
    if answer.store is not StoreState.OK:
        parts.append(f"the settings store is {answer.store.value}")
    others = [
        other.revision or ", ".join(f.path for f in other.files) or other.unpinned
        for _, other in stored_identities(answer.model.repository, answer.kind)
        if other.key != answer.model.key
    ]
    if others:
        parts.append(
            f"records exist under another identity ({'; '.join(others)}): name the model as its tune did"
        )
    return "; ".join(parts)


def _remedy(answer: SettingsAnswer, command: str) -> str:
    """What to do about a refusal, by what ``lookup`` did about the tune."""
    detail = answer.tune_detail or ""
    if answer.tune in _JOB_QUEUED and detail.isdigit():
        return f"wait for the tune, slurm job {detail}, to end; then submit again"
    why = f" (automatic tune: {answer.tune.value}{f', {detail}' if detail else ''})"
    return f"run the tune on this machine, through slurm with --gpu: {command}{why}"


def _decide(
    model: ModelIdentity,
    kind: WorkloadKind,
    engine: Engine,
    hours: float,
    max_untuned_hours: float,
    recipe: TuneRecipe | None,
) -> tuple[GateDecision, SettingsAnswer | None]:
    """The decision, and the lookup answer it was made from (None when no lookup was needed)."""
    from localmodels import settings  # noqa: PLC0415 - the lookup's module, loaded by the check
    from localmodels.serving.server import library_versions  # noqa: PLC0415
    from localmodels.serving.tune import hardware_key  # noqa: PLC0415

    if problem := _shape_problem(model, kind, engine):
        return GateDecision(allowed=False, reason=problem, remedy="fix the call"), None
    if hours <= max_untuned_hours:
        reason = f"expected {hours:g} h, at most {max_untuned_hours:g} h: no measured record is needed"
        return GateDecision(allowed=True, reason=reason), None
    report = inspect_hardware()
    hardware = hardware_key(report)
    if driver := _hidden_gpu(report):
        reason = f"this process sees no GPU though the host has one ({driver}), so no GPU record can be found"
        remedy = "run the check where the job's GPU is visible: the submitting shell, not a CPU-only job"
        return GateDecision(allowed=False, reason=reason, remedy=remedy, hardware=hardware), None
    answer = settings.lookup(model, kind, engine=engine, libraries=library_versions(), recipe=recipe)
    record_id = f"{answer.record.path}@{answer.record.measured_at.isoformat()}" if answer.record else None
    common: dict[str, Any] = {
        "hardware": hardware,
        "status": answer.status,
        "record_id": record_id,
        "stale": answer.stale,
        "tune": answer.tune,
        "tune_detail": answer.tune_detail,
    }
    if answer.status in _ALLOWING:
        stale = (
            f"; stale ({'; '.join(answer.stale)}), its confirm tune: {answer.tune.value}"
            if answer.stale
            else ""
        )
        reason = f"measured record {answer.record.candidate if answer.record else ''} on this machine{stale}"
        return GateDecision(allowed=True, reason=reason, **common), answer
    remedy = _remedy(answer, _tune_command(model, kind, engine, recipe))
    return GateDecision(allowed=False, reason=_why_refused(answer, hardware), remedy=remedy, **common), answer


def _log(decision: GateDecision, asked: dict[str, Any], answer: SettingsAnswer | None) -> None:
    """One line on stderr and one JSON line in the gate's log, for every decision."""
    from localmodels.settings import settings_dir  # noqa: PLC0415 - the lookup's module, loaded by the check

    verdict = "ALLOWED" if decision.allowed else "REFUSED"
    tail = f"; {decision.remedy}" if decision.remedy else ""
    sys.stderr.write(
        f"settings require: {verdict} {asked['model']} ({asked['kind']}, {asked['engine']}, "
        f"{asked['hours']:g} h): {decision.reason}{tail}\n"
    )
    line = {
        "time": datetime.now(UTC).isoformat(),
        **asked,
        **decision.model_dump(mode="json"),
        "stamp": answer.stamp().model_dump(mode="json") if answer is not None else None,
    }
    path = settings_dir() / GATE_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def require(
    model: ModelIdentity,
    kind: WorkloadKind,
    *,
    engine: Engine,
    hours: float,
    max_untuned_hours: float = DEFAULT_MAX_UNTUNED_HOURS,
    recipe: TuneRecipe | None = None,
) -> GateDecision:
    """Whether a GPU job expected to run ``hours`` may load ``model`` for ``kind`` work here; it never raises.

    ``recipe`` (the project's tune, ``inprocess_tune.in_process_recipe``) lets ``lookup`` queue a tune that is
    due (D2) and is the command a refusal names. The decision is logged, allows included.
    """
    decision, answer = _decide(model, kind, engine, hours, max_untuned_hours, recipe)
    asked = {
        "model": model.repository,
        "model_key": model.key,
        "kind": kind.value,
        "engine": engine.value,
        "hours": hours,
        "max_untuned_hours": max_untuned_hours,
    }
    _log(decision, asked, answer)
    return decision


def cmd_settings_require(args: argparse.Namespace) -> int:
    """`localmodels settings require`: exit 0 when the job may be submitted, 1 when it is refused."""
    from localmodels.settings import identity_of  # noqa: PLC0415 - the lookup's module, loaded by the check

    loader_flags = (args.project, args.loader, args.entry, args.texts)
    if any(loader_flags) and not all(loader_flags):
        sys.stderr.write(
            "settings require: give --project, --loader, --entry and --texts together, or none\n"
        )
        return 1
    try:
        model = identity_of(args.model, args.revision, args.file)
    except ValueError as exc:
        sys.stderr.write(f"settings require: cannot identify the model (give --revision or --file): {exc}\n")
        return 1
    kind, engine = WorkloadKind(args.kind), Engine(args.engine)
    recipe = _recipe(args, model, kind) if all(loader_flags) and engine is Engine.PYTORCH else None
    decision = require(
        model, kind, engine=engine, hours=args.hours, max_untuned_hours=args.max_untuned_hours, recipe=recipe
    )
    return 0 if decision.allowed else 1


def _recipe(args: argparse.Namespace, model: ModelIdentity, kind: WorkloadKind) -> TuneRecipe:
    """The project's in-process tune as a recipe, its reference named from the loader's own declaration."""
    from localmodels.inprocess_tune import in_process_recipe  # noqa: PLC0415
    from localmodels.serving.loader import resolve_factory  # noqa: PLC0415

    declaration = resolve_factory(args.loader)(args.entry).declare()
    return in_process_recipe(
        project=args.project,
        loader=args.loader,
        entry=args.entry,
        model=model,
        kind=kind,
        texts=Path(args.texts),
        workdir=Path.cwd(),
        reference=declaration.reference_name(),
    )


def register_require(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `settings require` to the `settings` group."""
    p = commands.add_parser("require", help="refuse a long GPU job whose model has no measured settings here")
    p.add_argument("--model", required=True, help="Hugging Face repository id")
    p.add_argument("--revision", default=None, help="the model's 40-character commit")
    p.add_argument(
        "--file", action="append", help="a GGUF file of a llama.cpp model (repeat for split parts)"
    )
    p.add_argument("--kind", required=True, choices=[k.value for k in WorkloadKind])
    p.add_argument("--engine", required=True, choices=[e.value for e in GATE_ENGINES])
    p.add_argument("--hours", type=float, required=True, help="the job's expected run time, in hours")
    p.add_argument(
        "--max-untuned-hours",
        type=float,
        default=DEFAULT_MAX_UNTUNED_HOURS,
        help="a job expected to run no longer needs no measured record",
    )
    p.add_argument("--project", default=None, help="with --loader, --entry and --texts: the in-process tune")
    p.add_argument("--loader", default=None, help="MODULE:ATTRIBUTE of the project's loader factory")
    p.add_argument("--entry", default=None, help="the project's name for the model")
    p.add_argument("--texts", default=None, help="the project's tune texts (JSONL)")
    p.set_defaults(fn=cmd_settings_require)
