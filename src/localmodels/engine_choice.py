"""Which engine serves a bulk job: llama.cpp, or a container engine a project has approved (design 6).

``choose`` applies the rules of docs/research/engine-registry-design-2026-11-08.md section 6 in order and
returns the engine, the rule that decided it, the numbers behind that and one line per approval considered.
A container engine is chosen only when every rule holds for an approval of the project: the approval exists,
the engine's `current` has a complete record of the approved file, llama.cpp has a same-client record of the
paired GGUF, the job has 16 or more requests in flight and saves at least twice the extra start time, and the
whole memory need fits. Anything else is llama.cpp's bulk layout. Nothing here starts an engine.

``settings`` imports this module for its types, so it stays light: the modules that read engine runs
(``container_run``, ``engine_refresh``, ``refresh``) are loaded inside ``choose`` when a rule needs them.
``import localmodels`` does not load this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from .engines import Approval, VersionRecord, approvals_for, read_record, versions_dir
from .enums import WorkloadKind
from .llama_builds import current_commit, link_target, llama_home
from .stack import Engine

if TYPE_CHECKING:
    from .settings import ModelIdentity, PinEntry

#: The figures are measured at 1 and 16 requests in flight and nothing between: below 16, llama.cpp
#: (at one request the container engines were slower, 0.82x, D16).
MIN_CONCURRENCY: Final = 16
#: The projected saving must be at least this many times the extra start time the container engine needs.
BREAK_EVEN_FACTOR: Final = 2.0
WRITE_FIGURE: Final = "write_tps_16_at_once"
READ_FIGURE: Final = re.compile(r"read_tps_prefill_(\d+)")
_LLAMA_KINDS: Final = (WorkloadKind.K1, WorkloadKind.K3)
_BULK_KIND: Final = WorkloadKind.K2


class JobShape(BaseModel):
    """What a bulk job will ask of the server: how much, and how many requests at once.

    ``prompt_tokens`` and ``output_tokens`` are typical per request. ``project`` names whose approvals count.
    """

    model_config = ConfigDict(frozen=True)

    project: str
    requests: int = Field(gt=0)
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(gt=0)
    concurrency: int = Field(gt=0)


class ChoiceRule(StrEnum):
    """Which rule decided the engine; for llama.cpp, the first rule an approval failed."""

    EXPLICIT = "explicit"
    PINNED = "pinned"
    INTERACTIVE_OR_SCORING = "interactive-or-scoring"  # K1, K3: llama.cpp (C3)
    PYTORCH_KIND = "pytorch-kind"  # K4-K10
    NO_JOB = "no-job"  # K2 without a JobShape
    NO_APPROVAL = "no-approval"
    NO_CURRENT = "no-current"
    NO_COMPLETE_RECORD = "no-complete-record"
    NO_SAME_CLIENT = "no-same-client"
    LOW_CONCURRENCY = "low-concurrency"
    SAVING_TOO_SMALL = "saving-too-small"
    MEMORY = "memory"
    CHOSEN = "chosen"


class EngineChoice(BaseModel):
    """The engine, the rule that decided it and its numbers; a container also has version, image, file."""

    engine: Engine
    rule: ChoiceRule
    version: str | None = None  # the container engine's tag
    image: str | None = None  # its digest, ``repo@sha256:...``
    file: str | None = None  # the container's model file, ``repo@revision``
    numbers: dict[str, float] = Field(default_factory=dict)
    considered: list[str] = Field(default_factory=list)

    def stamp(self) -> dict[str, str]:
        """The part of a result's stamp that says which engine was chosen and by which rule."""
        fields = {
            "engine": self.engine.value,
            "rule": self.rule.value,
            "version": self.version,
            "image": self.image,
            "file": self.file,
        }
        return {name: value for name, value in fields.items() if value is not None}

    def note(self) -> str:
        """One line for the answer's notes: engine, rule and numbers."""
        numbers = ", ".join(f"{name}={value:.4g}" for name, value in self.numbers.items())
        return f"engine {self.engine.value} ({self.rule.value}){': ' + numbers if numbers else ''}"


def pinned_choice(entry: PinEntry) -> EngineChoice:
    """The choice a pin entry holds: its engine, and for a container engine its version and file."""
    engine = entry.engine or Engine.LLAMA_CPP
    record = read_record(engine, entry.engine_version) if entry.engine_version else None
    return EngineChoice(
        engine=engine,
        rule=ChoiceRule.PINNED,
        version=entry.engine_version,
        image=record.image if record and record.image else None,
        file=entry.engine_file,
    )


@dataclass
class _Verdict:
    """One approval judged: the rule that failed (or CHOSEN), its numbers and the line for ``considered``."""

    approval: Approval
    rule: ChoiceRule
    why: str
    numbers: dict[str, float] = field(default_factory=dict)
    record: VersionRecord | None = None

    @property
    def line(self) -> str:
        """``considered``'s line for this approval."""
        verdict = "chosen" if self.rule is ChoiceRule.CHOSEN else f"not chosen ({self.rule.value})"
        return f"{self.approval.engine.value} {self.approval.file_pin}: {verdict}: {self.why}"


def _reading_seconds(figures: dict[str, float | None]) -> float | None:
    """Tokens per second of reading the prompt, from the largest prompt size measured; None when none.

    Prompt reading is measured on one request at a time (nothing is measured for it at 16 in flight), so
    this is the single-request figure, used for both sides alike.
    """
    sizes = {int(m[1]): name for name in figures if (m := READ_FIGURE.fullmatch(name))}
    value = figures.get(sizes[max(sizes)]) if sizes else None
    return value or None


def _writing(figures: dict[str, float | None]) -> float | None:
    value = figures.get(WRITE_FIGURE)
    return value or None


def _job_seconds(job: JobShape, write_tps: float, read_tps: float) -> float:
    """The job's time at these rates: writing at 16 at once, reading the prompts one request's rate."""
    return job.requests * job.output_tokens / write_tps + job.requests * job.prompt_tokens / read_tps


class _Run(NamedTuple):
    """The speed figures of one finished run and the seconds it took to start."""

    figures: dict[str, float | None]
    startup_s: float


def _complete_run(record: VersionRecord, model: str) -> _Run | None:
    """The newest result of ``model`` that finished every phase and holds the speed figures."""
    for entry in reversed(record.results):
        figures, startup = entry.get("figures"), entry.get("startup_s")
        if (
            entry.get("model") == model
            and entry.get("complete") is True
            and isinstance(figures, dict)
            and _writing(figures) is not None
            and _reading_seconds(figures) is not None
            and isinstance(startup, int | float)
        ):
            return _Run(figures, float(startup))
    return None


def _judge(approval: Approval, job: JobShape, available: float | None) -> _Verdict:  # noqa: PLR0911 - one return per rule
    from . import container_run, engine_refresh  # noqa: PLC0415 - see the module docstring
    from .refresh import Model  # noqa: PLC0415

    engine = approval.engine
    model = approval.file_repository
    tag = link_target(versions_dir(engine))
    record = read_record(engine, tag) if tag else None
    if record is None:
        return _Verdict(approval, ChoiceRule.NO_CURRENT, f"{engine.value} has no `current` version")
    run = _complete_run(record, model)
    if run is None:
        why = f"{engine.value} {record.tag} has no complete record of {model}"
        return _Verdict(approval, ChoiceRule.NO_COMPLETE_RECORD, why, record=record)
    gguf = Model(approval.gguf_repository, approval.gguf_revision, approval.gguf_file)
    commit = current_commit(llama_home())
    llama = engine_refresh.same_client_record(commit, gguf) if commit else None
    if llama is None or llama.startup_s is None or _writing(llama.figures) is None:
        why = f"llama.cpp has no same-client record of {gguf.filename} at its current build ({commit})"
        return _Verdict(approval, ChoiceRule.NO_SAME_CLIENT, why, record=record)
    if job.concurrency < MIN_CONCURRENCY:
        why = f"{job.concurrency} requests at once is under {MIN_CONCURRENCY}, the lowest measured"
        return _Verdict(approval, ChoiceRule.LOW_CONCURRENCY, why, record=record)
    t_container = _job_seconds(job, _writing(run.figures) or 0.0, _reading_seconds(run.figures) or 0.0)
    t_llama = _job_seconds(job, _writing(llama.figures) or 0.0, _reading_seconds(llama.figures) or 0.0)
    extra_start = max(0.0, run.startup_s - llama.startup_s)
    saving = t_llama - t_container
    numbers = {
        "t_llama_s": t_llama,
        "t_container_s": t_container,
        "saving_s": saving,
        "extra_start_s": extra_start,
    }
    if saving <= 0 or saving < BREAK_EVEN_FACTOR * extra_start:
        why = (
            f"saves {saving:.0f} s against {extra_start:.0f} s of extra start "
            f"(it must save at least {BREAK_EVEN_FACTOR:g} times that, and something)"
        )
        return _Verdict(approval, ChoiceRule.SAVING_TOO_SMALL, why, numbers, record)
    need = container_run.MemoryNeed.recorded(record, model)
    reserve = engine_refresh.HOST_RESERVE_GIB
    numbers.update({n: v for n, v in (("available_gib", available), ("need_gib", need)) if v is not None})
    if need is None or available is None or need + reserve > available:
        why = (
            f"needs {need if need is not None else 'an unrecorded amount of'} GiB plus {reserve:g} GiB "
            f"of host reserve, {available if available is not None else 'unreadable'} GiB available"
        )
        return _Verdict(approval, ChoiceRule.MEMORY, why, numbers, record)
    why = f"saves {saving:.0f} s for {extra_start:.0f} s of extra start"
    return _Verdict(approval, ChoiceRule.CHOSEN, why, numbers, record)


def choose(
    model: ModelIdentity,
    kind: WorkloadKind,
    job: JobShape | None,
    *,
    available_gib: float | None = None,
) -> EngineChoice:
    """The engine for ``model``'s ``kind`` work, by the rules of design 6 in order.

    ``available_gib`` is the memory a start can get now (default: the hardware reading; an unreadable
    one fails the memory rule). Never raises on missing records: each missing piece is a rule that fails.
    """
    if kind in _LLAMA_KINDS:
        return EngineChoice(engine=Engine.LLAMA_CPP, rule=ChoiceRule.INTERACTIVE_OR_SCORING)
    if kind is not _BULK_KIND:
        return EngineChoice(engine=Engine.PYTORCH, rule=ChoiceRule.PYTORCH_KIND)
    if job is None:
        return EngineChoice(engine=Engine.LLAMA_CPP, rule=ChoiceRule.NO_JOB)
    approvals = approvals_for(job.project, model.key)
    if not approvals:
        why = f"{job.project} has no approval for this model ({model.key})"
        return EngineChoice(engine=Engine.LLAMA_CPP, rule=ChoiceRule.NO_APPROVAL, considered=[why])
    from .container_run import available_gib as hardware_gib  # noqa: PLC0415 - see the module docstring

    available = hardware_gib() if available_gib is None else available_gib
    verdicts = [_judge(a, job, available) for a in approvals]
    passing = [v for v in verdicts if v.rule is ChoiceRule.CHOSEN]
    considered = [v.line for v in verdicts]
    if not passing:
        first = verdicts[0]
        return EngineChoice(
            engine=Engine.LLAMA_CPP, rule=first.rule, numbers=first.numbers, considered=considered
        )
    best = max(passing, key=lambda v: v.numbers["saving_s"])
    assert best.record is not None  # noqa: S101 - a chosen verdict always has its record
    return EngineChoice(
        engine=best.approval.engine,
        rule=ChoiceRule.CHOSEN,
        version=best.record.tag,
        image=best.record.image or None,
        file=best.approval.file_pin,
        numbers=best.numbers,
        considered=considered,
    )
